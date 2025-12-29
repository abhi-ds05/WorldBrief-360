# backend/app/integrations/un_client.py
"""
United Nations Data API client.
Provides access to UN datasets including:
- UNData API (data.un.org)
- UN Comtrade (international trade)
- UN SDG Indicators
- UN Population data
- UN Development Programme (UNDP) data
- UN World Food Programme (WFP) data
"""

import asyncio
import csv
import hashlib
import json
import logging
import re
from datetime import datetime, timedelta
from enum import Enum
from io import StringIO
from typing import Dict, List, Optional, Union, Any, Tuple
from urllib.parse import urlencode, quote

import aiohttp
import pandas as pd
from pydantic import BaseModel, Field, root_validator, root_validator, validator, HttpUrl
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.config import settings
from app.core.logging_config import get_logger
from app.cache.redis_client import get_redis_client
from app.utils.http_client import get_http_client

logger = get_logger(__name__)


# ============ Data Models ============

class UNDataAPI(str, Enum):
    """UN Data API endpoints."""
    UNDATA = "undata"           # data.un.org
    COMTRADE = "comtrade"       # comtrade.un.org
    SDG = "sdg"                # unstats.un.org/sdgs
    POPULATION = "population"   # population.un.org
    UNDP = "undp"              # hdr.undp.org
    WFP = "wfp"                # wfp.org
    UNHCR = "unhcr"            # data.unhcr.org (refugees)
    WHO = "who"                # who.int (health)
    UNESCO = "unesco"          # data.uis.unesco.org (education)
    FAO = "fao"                # fao.org (food/agriculture)


class UNIndicatorCategory(str, Enum):
    """UN indicator categories."""
    POPULATION = "population"
    ECONOMIC = "economic"
    SOCIAL = "social"
    ENVIRONMENTAL = "environmental"
    HEALTH = "health"
    EDUCATION = "education"
    GENDER = "gender"
    DEVELOPMENT = "development"
    TRADE = "trade"
    AGRICULTURE = "agriculture"
    ENERGY = "energy"
    TRANSPORT = "transport"
    TECHNOLOGY = "technology"
    GOVERNANCE = "governance"
    SECURITY = "security"


class UNRegion(str, Enum):
    """UN geographic regions."""
    WORLD = "world"
    AFRICA = "africa"
    ASIA = "asia"
    EUROPE = "europe"
    LATIN_AMERICA = "latin_america"
    NORTH_AMERICA = "north_america"
    OCEANIA = "oceania"
    MIDDLE_EAST = "middle_east"
    CARIBBEAN = "caribbean"
    DEVELOPED = "developed"
    DEVELOPING = "developing"
    LDC = "ldc"  # Least Developed Countries


class UNDataRequest(BaseModel):
    """UN Data API request model."""
    dataset: UNDataAPI
    indicator: Optional[str] = None
    country_code: Optional[str] = None  # ISO3 code
    country_name: Optional[str] = None
    region: Optional[UNRegion] = None
    year: Optional[int] = None
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    category: Optional[UNIndicatorCategory] = None
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = 0
    format: str = "json"  # json, csv, xml
    sort_by: Optional[str] = None
    sort_order: str = "desc"  # asc, desc
    include_metadata: bool = True
    
    @validator('year')
    def validate_year(cls, v):
        if v is not None:
            current_year = datetime.now().year
            if v < 1900 or v > current_year + 1:  # Allow future projections
                raise ValueError(f'Year must be between 1900 and {current_year + 1}')
        return v
    
    @validator('start_year', 'end_year')
    def validate_year_range(cls, v, values):
        if v is not None:
            current_year = datetime.now().year
            if v < 1900 or v > current_year + 1:
                raise ValueError(f'Year must be between 1900 and {current_year + 1}')
        return v
    
    @root_validator
    def validate_year_range_complete(cls, values):
        start_year = values.get('start_year')
        end_year = values.get('end_year')
        year = values.get('year')
        
        if year and (start_year or end_year):
            raise ValueError('Cannot specify both year and start_year/end_year')
        
        if start_year and end_year and start_year > end_year:
            raise ValueError('start_year must be less than or equal to end_year')
        
        return values


class UNDataResponse(BaseModel):
    """UN Data API response model."""
    request_id: str = Field(default_factory=lambda: f"un_{datetime.utcnow().timestamp()}")
    success: bool
    dataset: UNDataAPI
    data: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    pagination: Optional[Dict[str, Any]] = None
    total_count: Optional[int] = None
    cached: bool = False
    cache_key: Optional[str] = None
    execution_time_ms: Optional[float] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class UNIndicator(BaseModel):
    """UN indicator definition."""
    code: str
    name: str
    description: Optional[str] = None
    category: UNIndicatorCategory
    unit: Optional[str] = None
    source: UNDataAPI
    last_updated: Optional[datetime] = None
    frequency: Optional[str] = None  # annual, quarterly, monthly
    geographic_coverage: List[str] = Field(default_factory=list)
    time_coverage: Optional[str] = None  # e.g., "1960-2023"
    url: Optional[HttpUrl] = None
    sdg_goal: Optional[int] = None  # Related SDG goal (1-17)
    sdg_target: Optional[str] = None  # Related SDG target (e.g., "1.1")


class UNCountry(BaseModel):
    """UN country information."""
    iso3_code: str
    iso2_code: str
    name: str
    official_name: Optional[str] = None
    un_member: bool = True
    region: UNRegion
    subregion: Optional[str] = None
    population: Optional[int] = None
    area_km2: Optional[float] = None
    capital: Optional[str] = None
    currency: Optional[str] = None
    languages: List[str] = Field(default_factory=list)
    hdi_index: Optional[float] = None  # Human Development Index
    income_group: Optional[str] = None  # High, Upper-middle, Lower-middle, Low
    ldc: bool = False  # Least Developed Country
    lldc: bool = False  # Landlocked Developing Country
    sids: bool = False  # Small Island Developing State


class UNTradeData(BaseModel):
    """UN Comtrade trade data."""
    year: int
    reporter_code: str  # ISO3 code
    reporter_name: str
    partner_code: str  # ISO3 code
    partner_name: str
    commodity_code: str  # HS code
    commodity_name: str
    trade_flow: str  # Export, Import, Re-export, Re-import
    trade_value_usd: float
    quantity: Optional[float] = None
    quantity_unit: Optional[str] = None
    netweight_kg: Optional[float] = None


class UNPopulationData(BaseModel):
    """UN Population Division data."""
    year: int
    location_code: str  # ISO3 code or UN M49 code
    location_name: str
    variant: str  # Medium, High, Low, Constant fertility, etc.
    total_population: float
    male_population: Optional[float] = None
    female_population: Optional[float] = None
    population_density: Optional[float] = None  # persons per km²
    urban_population: Optional[float] = None
    rural_population: Optional[float] = None
    population_age_0_14: Optional[float] = None
    population_age_15_64: Optional[float] = None
    population_age_65_plus: Optional[float] = None
    median_age: Optional[float] = None
    life_expectancy: Optional[float] = None
    fertility_rate: Optional[float] = None
    mortality_rate: Optional[float] = None
    migration_rate: Optional[float] = None


class UNDevelopmentData(BaseModel):
    """UNDP Human Development data."""
    year: int
    country_code: str
    country_name: str
    hdi_index: float  # Human Development Index (0-1)
    life_expectancy: float
    education_index: float
    mean_years_schooling: float
    expected_years_schooling: float
    gni_per_capita: float
    gii_index: Optional[float] = None  # Gender Inequality Index
    gdi_index: Optional[float] = None  # Gender Development Index
    mpi_index: Optional[float] = None  # Multidimensional Poverty Index
    carbon_footprint: Optional[float] = None  # tons CO2 per capita
    ecological_footprint: Optional[float] = None


# ============ UN API Client ============

class UNClient:
    """
    United Nations Data API client with caching and rate limiting.
    """
    
    def __init__(self):
        self.http_client = get_http_client()
        self.redis = get_redis_client()
        self.cache_ttl = 86400  # 24 hours cache TTL
        self.initialized = False
        
        # API endpoints
        self.endpoints = {
            UNDataAPI.UNDATA: "https://data.un.org/api",
            UNDataAPI.COMTRADE: "https://comtrade.un.org/api",
            UNDataAPI.SDG: "https://unstats.un.org/sdgapi",
            UNDataAPI.POPULATION: "https://population.un.org/api",
            UNDataAPI.UNDP: "https://hdr.undp.org/api",
            UNDataAPI.WFP: "https://api.wfp.org/api",
            UNDataAPI.UNHCR: "https://data.unhcr.org/api",
            UNDataAPI.WHO: "https://who.int/api",
            UNDataAPI.UNESCO: "https://data.uis.unesco.org/api",
            UNDataAPI.FAO: "https://faostat.fao.org/api",
        }
        
        # API keys (if required)
        self.api_keys = {
            UNDataAPI.WFP: settings.WFP_API_KEY,
            UNDataAPI.UNHCR: settings.UNHCR_API_KEY,
            # Most UN APIs don't require keys
        }
        
        # Common indicators mapping
        self.indicator_mapping = self._load_indicator_mapping()
    
    def _load_indicator_mapping(self) -> Dict[str, UNIndicator]:
        """Load UN indicator mappings."""
        # This would be loaded from a database or configuration file
        # Here's a sample of important UN indicators
        return {
            # Population indicators
            "SP.POP.TOTL": UNIndicator(
                code="SP.POP.TOTL",
                name="Population, total",
                description="Total population",
                category=UNIndicatorCategory.POPULATION,
                unit="persons",
                source=UNDataAPI.POPULATION,
                frequency="annual",
                geographic_coverage=["world"],
                time_coverage="1950-2023",
                sdg_goal=1,
                sdg_target="1.1"
            ),
            "SP.URB.TOTL": UNIndicator(
                code="SP.URB.TOTL",
                name="Urban population",
                description="Urban population as percentage of total",
                category=UNIndicatorCategory.POPULATION,
                unit="% of total",
                source=UNDataAPI.POPULATION,
                frequency="annual"
            ),
            
            # Economic indicators
            "NY.GDP.MKTP.CD": UNIndicator(
                code="NY.GDP.MKTP.CD",
                name="GDP (current US$)",
                description="Gross Domestic Product in current US dollars",
                category=UNIndicatorCategory.ECONOMIC,
                unit="US$",
                source=UNDataAPI.UNDATA,
                frequency="annual"
            ),
            "NY.GDP.PCAP.CD": UNIndicator(
                code="NY.GDP.PCAP.CD",
                name="GDP per capita (current US$)",
                description="GDP per capita in current US dollars",
                category=UNIndicatorCategory.ECONOMIC,
                unit="US$",
                source=UNDataAPI.UNDATA,
                frequency="annual"
            ),
            
            # Social indicators
            "SP.DYN.LE00.IN": UNIndicator(
                code="SP.DYN.LE00.IN",
                name="Life expectancy at birth",
                description="Life expectancy at birth, total (years)",
                category=UNIndicatorCategory.HEALTH,
                unit="years",
                source=UNDataAPI.WHO,
                frequency="annual",
                sdg_goal=3,
                sdg_target="3.8"
            ),
            "SE.ADT.LITR.ZS": UNIndicator(
                code="SE.ADT.LITR.ZS",
                name="Literacy rate, adult total",
                description="Adult literacy rate, population 15+ years, both sexes (%)",
                category=UNIndicatorCategory.EDUCATION,
                unit="%",
                source=UNDataAPI.UNESCO,
                frequency="annual",
                sdg_goal=4,
                sdg_target="4.6"
            ),
            
            # SDG indicators
            "SI.POV.DDAY": UNIndicator(
                code="SI.POV.DDAY",
                name="Poverty headcount ratio at $1.90 a day",
                description="Percentage of population living on less than $1.90 a day",
                category=UNIndicatorCategory.SOCIAL,
                unit="% of population",
                source=UNDataAPI.SDG,
                frequency="annual",
                sdg_goal=1,
                sdg_target="1.1"
            ),
        }
    
    async def initialize(self) -> None:
        """Initialize the UN client."""
        if self.initialized:
            return
        
        logger.info("Initializing UN Data API client...")
        
        # Load country metadata
        await self._load_country_metadata()
        
        self.initialized = True
        logger.info("UN Data API client initialized")
    
    async def _load_country_metadata(self) -> None:
        """Load UN country metadata."""
        cache_key = "un:countries:metadata"
        
        # Try cache first
        cached = await self.redis.get(cache_key)
        if cached:
            self.countries = {c.iso3_code: c for c in [UNCountry(**json.loads(c)) for c in json.loads(cached)]}
            logger.info(f"Loaded {len(self.countries)} countries from cache")
            return
        
        # Load from UN API or static file
        # For now, we'll load a static list of UN member states
        # In production, this would come from UN API or a maintained dataset
        
        self.countries = {}
        logger.info("Country metadata loaded")
        
        # Cache for 7 days
        country_list = [c.dict() for c in self.countries.values()]
        await self.redis.setex(cache_key, 604800, json.dumps(country_list))
    
    async def _get_cache_key(self, request: UNDataRequest) -> str:
        """Generate cache key for request."""
        request_dict = request.dict(exclude={'include_metadata'})
        request_json = json.dumps(request_dict, sort_keys=True)
        request_hash = hashlib.md5(request_json.encode()).hexdigest()
        return f"un:data:{request.dataset.value}:{request_hash}"
    
    async def fetch_data(self, request: UNDataRequest) -> UNDataResponse:
        """
        Fetch data from UN APIs.
        
        Args:
            request: UN data request
            
        Returns:
            UNDataResponse with data
        """
        start_time = datetime.utcnow()
        
        # Check cache first
        cache_key = await self._get_cache_key(request)
        if request.include_metadata:
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                response_data = json.loads(cached_data)
                response_data['cached'] = True
                response_data['cache_key'] = cache_key
                response_data['execution_time_ms'] = (datetime.utcnow() - start_time).total_seconds() * 1000
                return UNDataResponse(**response_data)
        
        try:
            # Route to appropriate API handler
            handler_map = {
                UNDataAPI.UNDATA: self._fetch_undata,
                UNDataAPI.COMTRADE: self._fetch_comtrade,
                UNDataAPI.SDG: self._fetch_sdg,
                UNDataAPI.POPULATION: self._fetch_population,
                UNDataAPI.UNDP: self._fetch_undp,
                UNDataAPI.WFP: self._fetch_wfp,
                UNDataAPI.UNHCR: self._fetch_unhcr,
                UNDataAPI.WHO: self._fetch_who,
                UNDataAPI.UNESCO: self._fetch_unesco,
                UNDataAPI.FAO: self._fetch_fao,
            }
            
            handler = handler_map.get(request.dataset)
            if not handler:
                raise ValueError(f"Unsupported dataset: {request.dataset}")
            
            data = await handler(request)
            
            # Create response
            response = UNDataResponse(
                success=True,
                dataset=request.dataset,
                data=data.get("data", []),
                metadata=data.get("metadata", {}),
                pagination=data.get("pagination"),
                total_count=data.get("total_count", len(data.get("data", []))),
                cached=False,
                cache_key=cache_key,
                execution_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
            
            # Cache the response
            if request.include_metadata:
                await self.redis.setex(
                    cache_key,
                    self.cache_ttl,
                    json.dumps(response.dict())
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to fetch UN data: {str(e)}", exc_info=True)
            return UNDataResponse(
                success=False,
                dataset=request.dataset,
                error=str(e),
                execution_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    # ============ API Handlers ============
    
    async def _fetch_undata(self, request: UNDataRequest) -> Dict[str, Any]:
        """Fetch data from UN Data API (data.un.org)."""
        base_url = self.endpoints[UNDataAPI.UNDATA]
        
        # Build query parameters
        params = {
            "format": request.format,
            "pageSize": request.limit,
            "page": request.offset // request.limit + 1,
        }
        
        if request.indicator:
            params["indicator"] = request.indicator
        
        if request.country_code:
            params["country"] = request.country_code
        
        if request.year:
            params["year"] = request.year
        elif request.start_year and request.end_year:
            params["year"] = f"{request.start_year}-{request.end_year}"
        
        # Make request
        url = f"{base_url}/v1/data"
        if params:
            url += "?" + urlencode(params)
        
        async with self.http_client.get(url) as response:
            if response.status != 200:
                raise Exception(f"UN Data API error: {response.status}")
            
            data = await response.json()
            
            # Transform UN Data API response
            transformed_data = []
            for item in data.get("data", []):
                transformed_item = {
                    "year": item.get("Year"),
                    "country_code": item.get("Country_Code"),
                    "country_name": item.get("Country_Name"),
                    "indicator_code": item.get("Indicator_Code"),
                    "indicator_name": item.get("Indicator_Name"),
                    "value": item.get("Value"),
                    "unit": item.get("Unit"),
                    "footnote": item.get("Footnote"),
                    "source": item.get("Source"),
                }
                transformed_data.append(transformed_item)
            
            return {
                "data": transformed_data,
                "metadata": data.get("metadata", {}),
                "pagination": {
                    "page": data.get("page", 1),
                    "page_size": data.get("pageSize", request.limit),
                    "total_pages": data.get("totalPages", 1),
                },
                "total_count": data.get("total", len(transformed_data)),
            }
    
    async def _fetch_comtrade(self, request: UNDataRequest) -> Dict[str, Any]:
        """Fetch trade data from UN Comtrade API."""
        base_url = self.endpoints[UNDataAPI.COMTRADE]
        
        # Comtrade requires reporter and partner codes
        reporter_code = request.country_code or "all"
        partner_code = "world"  # Default to world
        
        # Build query
        params = {
            "fmt": "json",
            "max": request.limit,
            "type": "C",  # Commodities
            "freq": "A",  # Annual
            "px": "HS",   # HS classification
            "ps": str(request.year) if request.year else "recent",
            "r": reporter_code,
            "p": partner_code,
            "rg": "all",  # All trade regimes
            "cc": "TOTAL" if not request.indicator else request.indicator,
        }
        
        # Make request
        url = f"{base_url}/get"
        if params:
            url += "?" + urlencode(params)
        
        async with self.http_client.get(url) as response:
            if response.status != 200:
                raise Exception(f"UN Comtrade API error: {response.status}")
            
            data = await response.json()
            
            if data.get("validation", {}).get("status", {}).get("name") != "Ok":
                error_msg = data.get("validation", {}).get("status", {}).get("description", "Unknown error")
                raise Exception(f"UN Comtrade validation error: {error_msg}")
            
            # Transform Comtrade data
            transformed_data = []
            for item in data.get("dataset", []):
                transformed_item = {
                    "year": item.get("yr"),
                    "reporter_code": item.get("rtCode"),
                    "reporter_name": item.get("rtTitle"),
                    "partner_code": item.get("ptCode"),
                    "partner_name": item.get("ptTitle"),
                    "commodity_code": item.get("cmdCode"),
                    "commodity_name": item.get("cmdDescE"),
                    "trade_flow": item.get("rgDesc"),
                    "trade_value_usd": item.get("TradeValue"),
                    "quantity": item.get("qt"),
                    "quantity_unit": item.get("qtDesc"),
                    "netweight_kg": item.get("NetWeight"),
                }
                transformed_data.append(transformed_item)
            
            return {
                "data": transformed_data,
                "metadata": {
                    "api_version": data.get("version"),
                    "classification": data.get("classification", {}),
                },
                "total_count": len(transformed_data),
            }
    
    async def _fetch_sdg(self, request: UNDataRequest) -> Dict[str, Any]:
        """Fetch SDG indicators data."""
        base_url = self.endpoints[UNDataAPI.SDG]
        
        # Build query
        params = {
            "format": "json",
            "pageSize": request.limit,
            "page": request.offset // request.limit + 1,
        }
        
        if request.indicator:
            # SDG indicator code format: SDG_INDICATOR_CODE
            params["indicator"] = request.indicator
        
        if request.country_code:
            params["area"] = request.country_code
        
        if request.year:
            params["timePeriod"] = str(request.year)
        elif request.start_year and request.end_year:
            params["timePeriod"] = f"{request.start_year}-{request.end_year}"
        
        # Make request
        url = f"{base_url}/v1/sdg/Series/Data"
        if params:
            url += "?" + urlencode(params)
        
        async with self.http_client.get(url) as response:
            if response.status != 200:
                raise Exception(f"UN SDG API error: {response.status}")
            
            data = await response.json()
            
            # Transform SDG data
            transformed_data = []
            for item in data.get("data", []):
                transformed_item = {
                    "series_code": item.get("seriesCode"),
                    "series_description": item.get("seriesDescription"),
                    "geo_area_code": item.get("geoAreaCode"),
                    "geo_area_name": item.get("geoAreaName"),
                    "time_period": item.get("timePeriod"),
                    "value": item.get("value"),
                    "unit": item.get("unit"),
                    "footnote": item.get("footnote"),
                    "source": item.get("source"),
                    "goal": item.get("goal"),
                    "target": item.get("target"),
                    "indicator": item.get("indicator"),
                }
                transformed_data.append(transformed_item)
            
            return {
                "data": transformed_data,
                "metadata": {
                    "total": data.get("total"),
                    "page": data.get("page"),
                    "page_size": data.get("pageSize"),
                },
                "total_count": data.get("total", len(transformed_data)),
            }
    
    async def _fetch_population(self, request: UNDataRequest) -> Dict[str, Any]:
        """Fetch population data from UN Population Division."""
        base_url = self.endpoints[UNDataAPI.POPULATION]
        
        # Build query
        params = {
            "format": "json",
            "drilldowns": "Location,Time",
            "measures": "Population",
            "properties": "displayName",
        }
        
        if request.country_code and request.country_code != "world":
            params["Location"] = request.country_code
        
        if request.year:
            params["Time"] = str(request.year)
        
        # Population API uses different parameter format
        url = f"{base_url}/v1/data/WPP2022"
        if params:
            url += "?" + urlencode(params)
        
        async with self.http_client.get(url) as response:
            if response.status != 200:
                raise Exception(f"UN Population API error: {response.status}")
            
            data = await response.json()
            
            # Transform population data
            transformed_data = []
            for item in data.get("data", []):
                transformed_item = {
                    "location_code": item.get("Location", {}).get("code"),
                    "location_name": item.get("Location", {}).get("displayName"),
                    "time": item.get("Time"),
                    "population": item.get("Population"),
                    "variant": item.get("Variant", "Medium"),
                    "age_group": item.get("AgeGroup"),
                    "sex": item.get("Sex"),
                }
                transformed_data.append(transformed_item)
            
            return {
                "data": transformed_data,
                "metadata": {
                    "dataset": "WPP2022",
                    "variant": "Medium",
                    "source": "UN Population Division",
                },
                "total_count": len(transformed_data),
            }
    
    async def _fetch_undp(self, request: UNDataRequest) -> Dict[str, Any]:
        """Fetch UNDP Human Development data."""
        base_url = self.endpoints[UNDataAPI.UNDP]
        
        # UNDP API structure
        if request.indicator:
            endpoint = f"/v1/{request.indicator}"
        else:
            endpoint = "/v1/hdi"
        
        params = {}
        if request.country_code:
            params["country"] = request.country_code
        
        if request.year:
            params["year"] = request.year
        
        # Make request
        url = base_url + endpoint
        if params:
            url += "?" + urlencode(params)
        
        async with self.http_client.get(url) as response:
            if response.status != 200:
                raise Exception(f"UNDP API error: {response.status}")
            
            data = await response.json()
            
            return {
                "data": data if isinstance(data, list) else [data],
                "metadata": {
                    "source": "UNDP Human Development Report",
                    "api_version": "v1",
                },
                "total_count": 1 if not isinstance(data, list) else len(data),
            }
    
    async def _fetch_wfp(self, request: UNDataRequest) -> Dict[str, Any]:
        """Fetch World Food Programme data."""
        base_url = self.endpoints[UNDataAPI.WFP]
        
        # WFP requires API key
        api_key = self.api_keys.get(UNDataAPI.WFP)
        if not api_key:
            raise Exception("WFP API key not configured")
        
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        
        # Build query
        params = {
            "format": "json",
            "limit": request.limit,
            "offset": request.offset,
        }
        
        if request.country_code:
            params["country"] = request.country_code
        
        if request.year:
            params["year"] = request.year
        
        # WFP has different endpoints for different data types
        endpoint = "/v1/foodprices"  # Default to food prices
        if request.indicator:
            if "price" in request.indicator.lower():
                endpoint = "/v1/foodprices"
            elif "market" in request.indicator.lower():
                endpoint = "/v1/markets"
            elif "vam" in request.indicator.lower():
                endpoint = "/v1/vam"
        
        # Make request
        url = base_url + endpoint
        if params:
            url += "?" + urlencode(params)
        
        async with self.http_client.get(url, headers=headers) as response:
            if response.status != 200:
                raise Exception(f"WFP API error: {response.status}")
            
            data = await response.json()
            
            return {
                "data": data.get("data", []),
                "metadata": data.get("metadata", {}),
                "pagination": data.get("pagination"),
                "total_count": data.get("total_count", len(data.get("data", []))),
            }
    
    async def _fetch_unhcr(self, request: UNDataRequest) -> Dict[str, Any]:
        """Fetch UNHCR refugee data."""
        base_url = self.endpoints[UNDataAPI.UNHCR]
        
        # UNHCR API key (if available)
        api_key = self.api_keys.get(UNDataAPI.UNHCR)
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        # Build query
        params = {
            "format": "json",
            "limit": request.limit,
            "offset": request.offset,
        }
        
        if request.country_code:
            params["country"] = request.country_code
        
        if request.year:
            params["year"] = request.year
        
        # UNHCR endpoints
        endpoint = "/v1/population"  # Default to population statistics
        if request.indicator:
            if "asylum" in request.indicator.lower():
                endpoint = "/v1/asylum"
            elif "statistics" in request.indicator.lower():
                endpoint = "/v1/statistics"
        
        # Make request
        url = base_url + endpoint
        if params:
            url += "?" + urlencode(params)
        
        async with self.http_client.get(url, headers=headers) as response:
            if response.status != 200:
                raise Exception(f"UNHCR API error: {response.status}")
            
            data = await response.json()
            
            return {
                "data": data.get("data", []),
                "metadata": data.get("metadata", {}),
                "total_count": len(data.get("data", [])),
            }
    
    async def _fetch_who(self, request: UNDataRequest) -> Dict[str, Any]:
        """Fetch WHO health data."""
        base_url = self.endpoints[UNDataAPI.WHO]
        
        # WHO API parameters
        params = {
            "format": "json",
        }
        
        if request.indicator:
            params["indicator"] = request.indicator
        
        if request.country_code:
            params["country"] = request.country_code
        
        if request.year:
            params["year"] = request.year
        
        # Make request
        url = f"{base_url}/gho"
        if params:
            url += "?" + urlencode(params)
        
        async with self.http_client.get(url) as response:
            if response.status != 200:
                raise Exception(f"WHO API error: {response.status}")
            
            data = await response.json()
            
            # Transform WHO data (GHO format)
            transformed_data = []
            for fact in data.get("fact", []):
                transformed_item = {
                    "indicator": fact.get("dim", {}).get("GHO"),
                    "country": fact.get("dim", {}).get("COUNTRY"),
                    "year": fact.get("dim", {}).get("YEAR"),
                    "value": fact.get("value", {}).get("numeric"),
                    "low": fact.get("value", {}).get("low"),
                    "high": fact.get("value", {}).get("high"),
                }
                transformed_data.append(transformed_item)
            
            return {
                "data": transformed_data,
                "metadata": {
                    "source": "WHO Global Health Observatory",
                },
                "total_count": len(transformed_data),
            }
    
    async def _fetch_unesco(self, request: UNDataRequest) -> Dict[str, Any]:
        """Fetch UNESCO education data."""
        base_url = self.endpoints[UNDataAPI.UNESCO]
        
        # UNESCO UIS API
        params = {
            "format": "json",
            "limit": request.limit,
            "offset": request.offset,
        }
        
        if request.indicator:
            params["indicator"] = request.indicator
        
        if request.country_code:
            params["country"] = request.country_code
        
        if request.year:
            params["year"] = request.year
        
        # Make request
        url = f"{base_url}/v1.0"
        if params:
            url += "?" + urlencode(params)
        
        async with self.http_client.get(url) as response:
            if response.status != 200:
                raise Exception(f"UNESCO API error: {response.status}")
            
            data = await response.json()
            
            return {
                "data": data.get("data", []),
                "metadata": data.get("metadata", {}),
                "total_count": len(data.get("data", [])),
            }
    
    async def _fetch_fao(self, request: UNDataRequest) -> Dict[str, Any]:
        """Fetch FAO agriculture and food data."""
        base_url = self.endpoints[UNDataAPI.FAO]
        
        # FAOSTAT API
        params = {
            "format": "json",
        }
        
        if request.indicator:
            # FAOSTAT indicator codes
            params["indicator"] = request.indicator
        
        if request.country_code:
            params["area"] = request.country_code
        
        if request.year:
            params["year"] = request.year
        
        # Make request
        url = f"{base_url}/v3"
        if params:
            url += "?" + urlencode(params)
        
        async with self.http_client.get(url) as response:
            if response.status != 200:
                raise Exception(f"FAO API error: {response.status}")
            
            data = await response.json()
            
            # Transform FAO data
            transformed_data = []
            for item in data.get("data", []):
                transformed_item = {
                    "area_code": item.get("AreaCode"),
                    "area_name": item.get("AreaName"),
                    "item_code": item.get("ItemCode"),
                    "item_name": item.get("ItemName"),
                    "element_code": item.get("ElementCode"),
                    "element_name": item.get("ElementName"),
                    "year": item.get("Year"),
                    "unit": item.get("Unit"),
                    "value": item.get("Value"),
                    "flag": item.get("Flag"),
                }
                transformed_data.append(transformed_item)
            
            return {
                "data": transformed_data,
                "metadata": {
                    "dataset": data.get("dataset"),
                    "source": "FAO FAOSTAT",
                },
                "total_count": len(transformed_data),
            }
    
    # ============ Utility Methods ============
    
    async def get_country_info(self, country_code: str) -> Optional[UNCountry]:
        """Get UN country information by ISO3 code."""
        if not hasattr(self, 'countries'):
            await self._load_country_metadata()
        
        return self.countries.get(country_code.upper())
    
    async def search_countries(self, query: str, limit: int = 10) -> List[UNCountry]:
        """Search for countries by name or code."""
        if not hasattr(self, 'countries'):
            await self._load_country_metadata()
        
        query = query.lower()
        results = []
        
        for country in self.countries.values():
            if (query in country.name.lower() or 
                query in country.iso3_code.lower() or 
                query in country.iso2_code.lower() or
                (country.official_name and query in country.official_name.lower())):
                results.append(country)
            
            if len(results) >= limit:
                break
        
        return results
    
    async def get_indicator_info(self, indicator_code: str) -> Optional[UNIndicator]:
        """Get UN indicator information by code."""
        return self.indicator_mapping.get(indicator_code)
    
    async def search_indicators(self, query: str, category: Optional[UNIndicatorCategory] = None) -> List[UNIndicator]:
        """Search for UN indicators."""
        query = query.lower()
        results = []
        
        for indicator in self.indicator_mapping.values():
            if (query in indicator.name.lower() or 
                query in indicator.code.lower() or
                (indicator.description and query in indicator.description.lower())):
                
                if category and indicator.category != category:
                    continue
                
                results.append(indicator)
        
        # Sort by relevance (simple scoring)
        results.sort(key=lambda x: (
            query in x.name.lower(),
            query in x.code.lower(),
            len(x.name)
        ), reverse=True)
        
        return results[:50]  # Limit results
    
    async def get_sdg_indicators(self, goal: Optional[int] = None) -> List[UNIndicator]:
        """Get SDG indicators for a specific goal or all goals."""
        results = []
        
        for indicator in self.indicator_mapping.values():
            if indicator.sdg_goal:
                if goal is None or indicator.sdg_goal == goal:
                    results.append(indicator)
        
        return results
    
    async def get_population_trend(self, country_code: str, years: int = 10) -> List[Dict[str, Any]]:
        """Get population trend for a country."""
        current_year = datetime.now().year
        start_year = current_year - years
        
        request = UNDataRequest(
            dataset=UNDataAPI.POPULATION,
            country_code=country_code,
            start_year=start_year,
            end_year=current_year,
            limit=100,
        )
        
        response = await self.fetch_data(request)
        
        if not response.success:
            logger.error(f"Failed to fetch population trend: {response.error}")
            return []
        
        # Process and sort data
        trend_data = []
        for item in response.data:
            if item.get('time'):
                trend_data.append({
                    'year': int(item['time']),
                    'population': item.get('population'),
                    'variant': item.get('variant', 'Medium'),
                })
        
        # Sort by year
        trend_data.sort(key=lambda x: x['year'])
        
        return trend_data
    
    async def get_trade_partners(self, country_code: str, year: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get top trade partners for a country."""
        if not year:
            year = datetime.now().year - 1  # Use most recent complete year
        
        request = UNDataRequest(
            dataset=UNDataAPI.COMTRADE,
            country_code=country_code,
            year=year,
            indicator="TOTAL",  # Total trade
            limit=50,
        )
        
        response = await self.fetch_data(request)
        
        if not response.success:
            logger.error(f"Failed to fetch trade partners: {response.error}")
            return []
        
        # Aggregate trade by partner
        trade_by_partner = {}
        for item in response.data:
            partner_code = item.get('partner_code')
            trade_value = item.get('trade_value_usd', 0)
            trade_flow = item.get('trade_flow', '').lower()
            
            if partner_code and partner_code != country_code:
                if partner_code not in trade_by_partner:
                    trade_by_partner[partner_code] = {
                        'partner_code': partner_code,
                        'partner_name': item.get('partner_name'),
                        'exports': 0,
                        'imports': 0,
                        'total_trade': 0,
                    }
                
                if 'export' in trade_flow:
                    trade_by_partner[partner_code]['exports'] += trade_value
                elif 'import' in trade_flow:
                    trade_by_partner[partner_code]['imports'] += trade_value
                
                trade_by_partner[partner_code]['total_trade'] += trade_value
        
        # Convert to list and sort by total trade
        partners = list(trade_by_partner.values())
        partners.sort(key=lambda x: x['total_trade'], reverse=True)
        
        return partners[:20]  # Top 20 partners
    
    async def get_human_development_index(self, country_code: str) -> Optional[Dict[str, Any]]:
        """Get Human Development Index data for a country."""
        request = UNDataRequest(
            dataset=UNDataAPI.UNDP,
            country_code=country_code,
            indicator="hdi",
            limit=1,
        )
        
        response = await self.fetch_data(request)
        
        if not response.success or not response.data:
            logger.error(f"Failed to fetch HDI data: {response.error}")
            return None
        
        return response.data[0]
    
    async def get_country_comparison(self, country_codes: List[str], indicator: str) -> Dict[str, Any]:
        """Compare multiple countries on a specific indicator."""
        tasks = []
        for country_code in country_codes:
            request = UNDataRequest(
                dataset=UNDataAPI.UNDATA,
                country_code=country_code,
                indicator=indicator,
                limit=1,
            )
            tasks.append(self.fetch_data(request))
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        comparison_data = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Error fetching data for {country_codes[i]}: {str(response)}")
                continue
            
            if response.success and response.data:
                country_data = response.data[0]
                comparison_data.append({
                    'country_code': country_codes[i],
                    'country_name': country_data.get('country_name'),
                    'value': country_data.get('value'),
                    'year': country_data.get('year'),
                    'unit': country_data.get('unit'),
                })
        
        # Sort by value (descending)
        comparison_data.sort(key=lambda x: x.get('value') or 0, reverse=True)
        
        return {
            'indicator': indicator,
            'indicator_name': self.indicator_mapping.get(indicator, UNIndicator(code=indicator, name=indicator, category=UNIndicatorCategory.ECONOMIC)).name,
            'comparison': comparison_data,
            'timestamp': datetime.utcnow().isoformat(),
        }
    
    async def get_world_bank_comparison(self, indicator: str, year: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get world data for an indicator (all countries).
        Note: This uses World Bank data via UN Data API.
        """
        if not year:
            year = datetime.now().year - 1
        
        request = UNDataRequest(
            dataset=UNDataAPI.UNDATA,
            indicator=indicator,
            year=year,
            limit=200,  # Get as many countries as possible
        )
        
        response = await self.fetch_data(request)
        
        if not response.success:
            logger.error(f"Failed to fetch world data: {response.error}")
            return []
        
        # Filter out regional/group data, keep only countries
        country_data = []
        for item in response.data:
            country_code = item.get('country_code')
            if country_code and len(country_code) == 3:  # ISO3 code
                country_data.append({
                    'country_code': country_code,
                    'country_name': item.get('country_name'),
                    'value': item.get('value'),
                    'year': item.get('year'),
                })
        
        # Sort by value
        country_data.sort(key=lambda x: x.get('value') or 0, reverse=True)
        
        return country_data
    
    async def clear_cache(self, pattern: str = "un:data:*") -> int:
        """Clear UN data cache."""
        keys = await self.redis.keys(pattern)
        if keys:
            deleted = await self.redis.delete(*keys)
            logger.info(f"Cleared {deleted} UN cache keys")
            return deleted
        return 0


# ============ Factory Function ============

_un_client: Optional[UNClient] = None

async def get_un_client() -> UNClient:
    """
    Get or create a UN client singleton.
    
    Returns:
        UNClient instance
    """
    global _un_client
    
    if _un_client is None:
        _un_client = UNClient()
        await _un_client.initialize()
    
    return _un_client


# ============ Utility Functions ============

async def get_country_development_profile(country_code: str) -> Dict[str, Any]:
    """
    Get comprehensive development profile for a country.
    
    Args:
        country_code: ISO3 country code
        
    Returns:
        Country development profile
    """
    client = await get_un_client()
    
    # Get country info
    country_info = await client.get_country_info(country_code)
    if not country_info:
        return {"error": f"Country not found: {country_code}"}
    
    # Fetch key indicators in parallel
    indicators_to_fetch = [
        ("SP.POP.TOTL", "population"),
        ("NY.GDP.MKTP.CD", "gdp"),
        ("NY.GDP.PCAP.CD", "gdp_per_capita"),
        ("SP.DYN.LE00.IN", "life_expectancy"),
        ("SE.ADT.LITR.ZS", "literacy_rate"),
        ("SI.POV.DDAY", "poverty_rate"),
    ]
    
    tasks = []
    for indicator_code, _ in indicators_to_fetch:
        request = UNDataRequest(
            dataset=UNDataAPI.UNDATA,
            country_code=country_code,
            indicator=indicator_code,
            limit=1,
        )
        tasks.append(client.fetch_data(request))
    
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process responses
    indicator_data = {}
    for i, (indicator_code, key_name) in enumerate(indicators_to_fetch):
        response = responses[i]
        if isinstance(response, Exception):
            logger.error(f"Error fetching {indicator_code}: {str(response)}")
            continue
        
        if response.success and response.data:
            data = response.data[0]
            indicator_data[key_name] = {
                'value': data.get('value'),
                'year': data.get('year'),
                'unit': data.get('unit'),
            }
    
    # Get HDI data
    hdi_data = await client.get_human_development_index(country_code)
    
    # Get population trend
    population_trend = await client.get_population_trend(country_code, years=10)
    
    # Get trade partners
    trade_partners = await client.get_trade_partners(country_code)
    
    return {
        'country_info': country_info.dict(),
        'indicators': indicator_data,
        'hdi': hdi_data,
        'population_trend': population_trend,
        'trade_partners': trade_partners[:10],  # Top 10
        'last_updated': datetime.utcnow().isoformat(),
    }


async def get_global_development_report(year: Optional[int] = None) -> Dict[str, Any]:
    """
    Generate a global development report.
    
    Args:
        year: Report year (defaults to most recent)
        
    Returns:
        Global development report
    """
    if not year:
        year = datetime.now().year - 1
    
    client = await get_un_client()
    
    # Key global indicators
    indicators = {
        'world_population': 'SP.POP.TOTL',
        'world_gdp': 'NY.GDP.MKTP.CD',
        'global_poverty': 'SI.POV.DDAY',
        'life_expectancy': 'SP.DYN.LE00.IN',
        'literacy_rate': 'SE.ADT.LITR.ZS',
    }
    
    # Fetch world data for each indicator
    indicator_data = {}
    for key, indicator_code in indicators.items():
        request = UNDataRequest(
            dataset=UNDataAPI.UNDATA,
            indicator=indicator_code,
            year=year,
            country_code="WLD",  # World code
            limit=1,
        )
        
        response = await client.fetch_data(request)
        if response.success and response.data:
            data = response.data[0]
            indicator_data[key] = {
                'value': data.get('value'),
                'unit': data.get('unit'),
                'year': data.get('year'),
            }
    
    # Get regional data
    regions = [UNRegion.AFRICA, UNRegion.ASIA, UNRegion.EUROPE, 
               UNRegion.LATIN_AMERICA, UNRegion.NORTH_AMERICA]
    
    regional_gdp = {}
    for region in regions:
        # Note: This would need region-specific country codes
        # Simplified for example
        pass
    
    return {
        'year': year,
        'global_indicators': indicator_data,
        'regional_data': regional_gdp,
        'report_date': datetime.utcnow().isoformat(),
    }


async def get_sdg_progress_report(goal: Optional[int] = None) -> Dict[str, Any]:
    """
    Get SDG progress report.
    
    Args:
        goal: Specific SDG goal (1-17) or all if None
        
    Returns:
        SDG progress report
    """
    client = await get_un_client()
    
    # Get SDG indicators
    sdg_indicators = await client.get_sdg_indicators(goal)
    
    # For each indicator, get global progress
    progress_data = []
    
    for indicator in sdg_indicators[:10]:  # Limit to 10 indicators for performance
        request = UNDataRequest(
            dataset=UNDataAPI.SDG,
            indicator=indicator.code,
            limit=1,
            sort_by="TimePeriod",
            sort_order="desc",
        )
        
        response = await client.fetch_data(request)
        if response.success and response.data:
            latest_data = response.data[0]
            progress_data.append({
                'indicator': indicator.code,
                'indicator_name': indicator.name,
                'latest_value': latest_data.get('value'),
                'year': latest_data.get('time_period'),
                'unit': latest_data.get('unit'),
                'goal': indicator.sdg_goal,
                'target': indicator.sdg_target,
            })
    
    # Group by SDG goal
    goals = {}
    for item in progress_data:
        goal_num = item['goal']
        if goal_num not in goals:
            goals[goal_num] = []
        goals[goal_num].append(item)
    
    return {
        'sdg_goals': goals,
        'total_indicators': len(progress_data),
        'report_date': datetime.utcnow().isoformat(),
    }