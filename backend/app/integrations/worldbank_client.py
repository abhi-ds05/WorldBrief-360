# backend/app/integrations/worldbank_client.py
"""
World Bank Data API client.
Provides access to:
- World Development Indicators (WDI)
- World Bank Projects
- Climate Change Data
- Poverty and Inequality Data
- Education Statistics
- Health Statistics
- Economic Indicators
- Financial Sector Data
- Trade Statistics
- Infrastructure Data
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
from pydantic import BaseModel, Field, root_validator, validator, HttpUrl
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.config import settings
from app.core.logging_config import get_logger
from app.cache.redis_client import get_redis_client
from app.utils.http_client import get_http_client

logger = get_logger(__name__)


# ============ Data Models ============

class WorldBankDataset(str, Enum):
    """World Bank datasets."""
    WDI = "wdi"                    # World Development Indicators
    PROJECTS = "projects"          # World Bank Projects
    CLIMATE = "climate"           # Climate Change Data
    POVERTY = "poverty"           # Poverty and Inequality
    EDUCATION = "education"       # Education Statistics
    HEALTH = "health"             # Health Statistics
    FINANCE = "finance"           # Financial Sector
    TRADE = "trade"               # Trade Statistics
    INFRASTRUCTURE = "infra"      # Infrastructure
    ENTERPRISE = "enterprise"     # Enterprise Surveys
    GENDER = "gender"             # Gender Statistics
    JOBS = "jobs"                 # Jobs Data
    MACRO = "macro"               # Macroeconomic Data
    DEBT = "debt"                 # Debt Statistics
    ENERGY = "energy"             # Energy Statistics
    AGRICULTURE = "agriculture"   # Agriculture and Rural Development


class IncomeGroup(str, Enum):
    """World Bank income groups."""
    LOW_INCOME = "LIC"
    LOWER_MIDDLE_INCOME = "LMC"
    UPPER_MIDDLE_INCOME = "UMC"
    HIGH_INCOME = "HIC"
    ALL = "ALL"


class Region(str, Enum):
    """World Bank regions."""
    EAST_ASIA_PACIFIC = "EAS"
    EUROPE_CENTRAL_ASIA = "ECS"
    LATIN_AMERICA_CARIBBEAN = "LCN"
    MIDDLE_EAST_NORTH_AFRICA = "MEA"
    NORTH_AMERICA = "NAC"
    SOUTH_ASIA = "SAS"
    SUB_SAHARAN_AFRICA = "SSF"
    WORLD = "WLD"


class IndicatorCategory(str, Enum):
    """World Bank indicator categories."""
    POVERTY = "poverty"
    HEALTH = "health"
    EDUCATION = "education"
    ECONOMIC = "economic"
    FINANCE = "finance"
    TRADE = "trade"
    INFRASTRUCTURE = "infrastructure"
    CLIMATE = "climate"
    GOVERNANCE = "governance"
    SOCIAL = "social"
    DEMOGRAPHIC = "demographic"
    ENVIRONMENT = "environment"
    ENERGY = "energy"
    AGRICULTURE = "agriculture"
    INDUSTRY = "industry"
    SERVICES = "services"


class WorldBankRequest(BaseModel):
    """World Bank API request model."""
    dataset: WorldBankDataset = WorldBankDataset.WDI
    indicator: Optional[str] = None  # Indicator code (e.g., NY.GDP.MKTP.CD)
    country_code: Optional[str] = None  # ISO3 code or country ID
    country_name: Optional[str] = None
    region: Optional[Region] = None
    income_group: Optional[IncomeGroup] = None
    year: Optional[int] = None
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    category: Optional[IndicatorCategory] = None
    limit: int = Field(default=100, ge=1, le=10000)
    offset: int = 0
    format: str = "json"  # json, csv, xml
    per_page: Optional[int] = None
    page: Optional[int] = None
    sort_by: Optional[str] = None
    sort_order: str = "desc"  # asc, desc
    include_metadata: bool = True
    include_source: bool = True
    include_footnotes: bool = True
    cache_ttl: int = 86400  # 24 hours default
    
    @validator('year')
    def validate_year(cls, v):
        if v is not None:
            current_year = datetime.now().year
            if v < 1960 or v > current_year + 1:  # World Bank data starts around 1960
                raise ValueError(f'Year must be between 1960 and {current_year + 1}')
        return v
    
    @validator('start_year', 'end_year')
    def validate_year_range(cls, v, values):
        if v is not None:
            current_year = datetime.now().year
            if v < 1960 or v > current_year + 1:
                raise ValueError(f'Year must be between 1960 and {current_year + 1}')
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
    
    @validator('country_code')
    def validate_country_code(cls, v):
        if v is not None:
            # World Bank uses special codes like "WLD" for World
            valid_special_codes = ["WLD", "ALL", "INX", "ARB", "CEB", "EAP", "EAS", "ECA", "ECS", 
                                 "EMU", "EUU", "FCS", "HIC", "HPC", "IBD", "IBT", "IDA", "IDB", 
                                 "IDX", "LAC", "LCN", "LIC", "LMC", "LMY", "LTE", "MEA", "MIC", 
                                 "MNA", "NAC", "OED", "OSS", "PRE", "PSS", "PST", "SAS", "SSA", 
                                 "SSF", "SST", "TEA", "TEC", "TLA", "TMN", "TSA", "TSS", "UMC"]
            if len(v) == 3 and (v.isalpha() or v in valid_special_codes):
                return v
            raise ValueError('Country code must be 3-letter ISO code or World Bank special code')
        return v


class WorldBankResponse(BaseModel):
    """World Bank API response model."""
    request_id: str = Field(default_factory=lambda: f"wb_{datetime.utcnow().timestamp()}")
    success: bool
    dataset: WorldBankDataset
    data: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    pagination: Optional[Dict[str, Any]] = None
    total_count: Optional[int] = None
    cached: bool = False
    cache_key: Optional[str] = None
    execution_time_ms: Optional[float] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WorldBankIndicator(BaseModel):
    """World Bank indicator definition."""
    id: str  # Indicator code (e.g., NY.GDP.MKTP.CD)
    name: str
    description: Optional[str] = None
    source: str = "World Bank"
    source_note: Optional[str] = None
    source_organization: Optional[str] = None
    category: IndicatorCategory
    subcategory: Optional[str] = None
    unit: Optional[str] = None
    scale: Optional[str] = None  # e.g., "Millions", "Billions", "Percentage"
    decimal_places: int = 2
    topic: Optional[str] = None
    license_type: Optional[str] = None
    periodicity: Optional[str] = None  # Annual, Quarterly, Monthly
    base_period: Optional[str] = None
    aggregation_method: Optional[str] = None
    statistical_concept: Optional[str] = None
    development_relevance: Optional[str] = None
    limitations: Optional[str] = None
    notes_from_original_source: Optional[str] = None
    general_comments: Optional[str] = None
    last_updated: Optional[datetime] = None
    coverage: Optional[List[str]] = None  # Country/region coverage
    start_year: Optional[int] = None
    end_year: Optional[int] = None


class WorldBankCountry(BaseModel):
    """World Bank country information."""
    id: str  # ISO3 code
    iso2_code: Optional[str] = None
    name: str
    region: Region
    income_group: IncomeGroup
    capital_city: Optional[str] = None
    longitude: Optional[float] = None
    latitude: Optional[float] = None
    lending_type: Optional[str] = None  # IDA, IBRD, Blend, etc.
    population: Optional[int] = None
    surface_area: Optional[float] = None  # sq km
    gdp: Optional[float] = None  # current US$
    gdp_per_capita: Optional[float] = None
    hdi_index: Optional[float] = None  # Human Development Index
    poverty_headcount: Optional[float] = None  # % below poverty line
    last_updated: Optional[datetime] = None


class WorldBankProject(BaseModel):
    """World Bank project information."""
    project_id: str
    project_name: str
    country_code: str
    country_name: str
    region: Region
    status: str  # Active, Closed, Pipeline, etc.
    approval_date: Optional[datetime] = None
    closing_date: Optional[datetime] = None
    total_commitment: Optional[float] = None  # US$
    total_disbursement: Optional[float] = None  # US$
    sector: Optional[str] = None
    themes: List[str] = Field(default_factory=list)
    implementing_agency: Optional[str] = None
    environmental_category: Optional[str] = None
    project_development_objective: Optional[str] = None
    last_updated: Optional[datetime] = None


class WorldBankDataPoint(BaseModel):
    """World Bank data point for time series."""
    year: int
    value: Optional[float] = None
    decimal: Optional[int] = None
    footnote: Optional[str] = None
    date: Optional[str] = None  # For non-annual data


class WorldBankIndicatorValue(BaseModel):
    """World Bank indicator value for a specific country and year."""
    indicator_code: str
    indicator_name: str
    country_code: str
    country_name: str
    year: int
    value: Optional[float] = None
    unit: Optional[str] = None
    decimal: int = 2
    footnote: Optional[str] = None
    source: Optional[str] = None
    last_updated: Optional[datetime] = None


# ============ World Bank Client ============

class WorldBankClient:
    """
    World Bank Data API client with caching and rate limiting.
    """
    
    def __init__(self):
        self.http_client = get_http_client()
        self.redis = get_redis_client()
        self.default_cache_ttl = 86400  # 24 hours
        self.initialized = False
        
        # API endpoints
        self.endpoints = {
            WorldBankDataset.WDI: "https://api.worldbank.org/v2",
            WorldBankDataset.PROJECTS: "https://search.worldbank.org/api/v2/projects",
            WorldBankDataset.CLIMATE: "https://climateknowledgeportal.worldbank.org/api",
            WorldBankDataset.POVERTY: "https://api.worldbank.org/v2/poverty",
            # Other datasets use the main API with different endpoints
        }
        
        # API version
        self.api_version = "v2"
        
        # Rate limiting (requests per minute)
        self.rate_limit = 100
        
        # Common indicators mapping
        self.indicator_mapping = self._load_indicator_mapping()
        
        # Country metadata
        self.countries: Dict[str, WorldBankCountry] = {}
        
        # Dataset-specific configurations
        self.dataset_configs = {
            WorldBankDataset.WDI: {
                "base_url": "https://api.worldbank.org/v2",
                "endpoint": "/country/{country}/indicator/{indicator}",
                "format": "json",
                "per_page": 1000,
            },
            WorldBankDataset.PROJECTS: {
                "base_url": "https://search.worldbank.org/api/v2",
                "endpoint": "/projects",
                "format": "json",
                "per_page": 50,
            },
            WorldBankDataset.CLIMATE: {
                "base_url": "https://climateknowledgeportal.worldbank.org/api",
                "endpoint": "/data",
                "format": "json",
                "per_page": 100,
            },
        }
    
    def _load_indicator_mapping(self) -> Dict[str, WorldBankIndicator]:
        """Load World Bank indicator mappings."""
        # This would be loaded from a database or configuration file
        # Here's a sample of important World Bank indicators
        return {
            # Economic indicators
            "NY.GDP.MKTP.CD": WorldBankIndicator(
                id="NY.GDP.MKTP.CD",
                name="GDP (current US$)",
                description="Gross Domestic Product in current US dollars",
                category=IndicatorCategory.ECONOMIC,
                unit="US$",
                scale="Billions",
                decimal_places=2,
                periodicity="Annual",
                source_organization="World Bank, International Comparison Program database.",
                development_relevance="GDP at purchaser's prices is the sum of gross value added by all resident producers in the economy plus any product taxes and minus any subsidies not included in the value of the products.",
            ),
            "NY.GDP.MKTP.KD.ZG": WorldBankIndicator(
                id="NY.GDP.MKTP.KD.ZG",
                name="GDP growth (annual %)",
                description="Annual percentage growth rate of GDP at market prices based on constant local currency",
                category=IndicatorCategory.ECONOMIC,
                unit="%",
                decimal_places=2,
                periodicity="Annual",
                development_relevance="GDP is the sum of gross value added by all resident producers in the economy plus any product taxes and minus any subsidies not included in the value of the products.",
            ),
            "NY.GDP.PCAP.CD": WorldBankIndicator(
                id="NY.GDP.PCAP.CD",
                name="GDP per capita (current US$)",
                description="GDP per capita is gross domestic product divided by midyear population",
                category=IndicatorCategory.ECONOMIC,
                unit="US$",
                decimal_places=2,
                periodicity="Annual",
            ),
            
            # Population indicators
            "SP.POP.TOTL": WorldBankIndicator(
                id="SP.POP.TOTL",
                name="Population, total",
                description="Total population",
                category=IndicatorCategory.DEMOGRAPHIC,
                unit="persons",
                scale="Millions",
                decimal_places=0,
                periodicity="Annual",
            ),
            "SP.POP.GROW": WorldBankIndicator(
                id="SP.POP.GROW",
                name="Population growth (annual %)",
                description="Annual population growth rate",
                category=IndicatorCategory.DEMOGRAPHIC,
                unit="%",
                decimal_places=2,
                periodicity="Annual",
            ),
            
            # Poverty indicators
            "SI.POV.DDAY": WorldBankIndicator(
                id="SI.POV.DDAY",
                name="Poverty headcount ratio at $1.90 a day (2011 PPP) (% of population)",
                description="Percentage of population living on less than $1.90 a day at 2011 international prices",
                category=IndicatorCategory.POVERTY,
                unit="% of population",
                decimal_places=2,
                periodicity="Annual",
                source_organization="World Bank, Development Research Group.",
                development_relevance="Poverty headcount ratio at $1.90 a day is the percentage of the population living on less than $1.90 a day at 2011 international prices.",
            ),
            "SI.POV.GINI": WorldBankIndicator(
                id="SI.POV.GINI",
                name="Gini index",
                description="Measure of income inequality (0 represents perfect equality, 100 perfect inequality)",
                category=IndicatorCategory.POVERTY,
                unit="index",
                decimal_places=2,
                periodicity="Annual",
            ),
            
            # Health indicators
            "SP.DYN.LE00.IN": WorldBankIndicator(
                id="SP.DYN.LE00.IN",
                name="Life expectancy at birth, total (years)",
                description="Number of years a newborn infant would live if prevailing patterns of mortality at the time of its birth were to stay the same throughout its life",
                category=IndicatorCategory.HEALTH,
                unit="years",
                decimal_places=2,
                periodicity="Annual",
            ),
            "SH.DYN.MORT": WorldBankIndicator(
                id="SH.DYN.MORT",
                name="Mortality rate, under-5 (per 1,000 live births)",
                description="Probability per 1,000 that a newborn baby will die before reaching age five, if subject to age-specific mortality rates of the specified year",
                category=IndicatorCategory.HEALTH,
                unit="per 1,000 live births",
                decimal_places=2,
                periodicity="Annual",
            ),
            
            # Education indicators
            "SE.PRM.ENRR": WorldBankIndicator(
                id="SE.PRM.ENRR",
                name="School enrollment, primary (% gross)",
                description="Gross enrollment ratio is the ratio of total enrollment, regardless of age, to the population of the age group that officially corresponds to the level of education shown",
                category=IndicatorCategory.EDUCATION,
                unit="%",
                decimal_places=2,
                periodicity="Annual",
            ),
            "SE.ADT.LITR.ZS": WorldBankIndicator(
                id="SE.ADT.LITR.ZS",
                name="Literacy rate, adult total (% of people ages 15 and above)",
                description="Percentage of population ages 15 and above who can both read and write with understanding a short simple statement about their everyday life",
                category=IndicatorCategory.EDUCATION,
                unit="%",
                decimal_places=2,
                periodicity="Annual",
            ),
            
            # Climate indicators
            "EN.ATM.CO2E.PC": WorldBankIndicator(
                id="EN.ATM.CO2E.PC",
                name="CO2 emissions (metric tons per capita)",
                description="Carbon dioxide emissions are those stemming from the burning of fossil fuels and the manufacture of cement",
                category=IndicatorCategory.CLIMATE,
                unit="metric tons per capita",
                decimal_places=2,
                periodicity="Annual",
            ),
            "AG.LND.FRST.ZS": WorldBankIndicator(
                id="AG.LND.FRST.ZS",
                name="Forest area (% of land area)",
                description="Forest area is land under natural or planted stands of trees of at least 5 meters in situ, whether productive or not, and excludes tree stands in agricultural production systems",
                category=IndicatorCategory.CLIMATE,
                unit="% of land area",
                decimal_places=2,
                periodicity="Annual",
            ),
            
            # Infrastructure indicators
            "EG.ELC.ACCS.ZS": WorldBankIndicator(
                id="EG.ELC.ACCS.ZS",
                name="Access to electricity (% of population)",
                description="Percentage of population with access to electricity",
                category=IndicatorCategory.INFRASTRUCTURE,
                unit="% of population",
                decimal_places=2,
                periodicity="Annual",
            ),
            "IS.RRS.TOTL.KM": WorldBankIndicator(
                id="IS.RRS.TOTL.KM",
                name="Railways, total route-km",
                description="Total route length of railway lines in operation",
                category=IndicatorCategory.INFRASTRUCTURE,
                unit="km",
                decimal_places=0,
                periodicity="Annual",
            ),
        }
    
    async def initialize(self) -> None:
        """Initialize the World Bank client."""
        if self.initialized:
            return
        
        logger.info("Initializing World Bank Data API client...")
        
        # Load country metadata
        await self._load_country_metadata()
        
        # Load indicator metadata
        await self._load_indicator_metadata()
        
        self.initialized = True
        logger.info(f"World Bank client initialized with {len(self.countries)} countries and {len(self.indicator_mapping)} indicators")
    
    async def _load_country_metadata(self) -> None:
        """Load World Bank country metadata."""
        cache_key = "wb:countries:metadata"
        
        # Try cache first
        cached = await self.redis.get(cache_key)
        if cached:
            self.countries = {c.id: c for c in [WorldBankCountry(**json.loads(c)) for c in json.loads(cached)]}
            logger.info(f"Loaded {len(self.countries)} countries from cache")
            return
        
        # Load from World Bank API
        base_url = "https://api.worldbank.org/v2"
        params = {
            "format": "json",
            "per_page": "300",  # Get all countries
        }
        
        url = f"{base_url}/country"
        if params:
            url += "?" + urlencode(params)
        
        try:
            async with self.http_client.get(url) as response:
                if response.status != 200:
                    logger.warning(f"Failed to load country metadata: HTTP {response.status}")
                    self.countries = {}
                    return
                
                data = await response.json()
                
                # World Bank API returns array of arrays
                countries_data = data[1] if len(data) > 1 else []
                
                # Parse countries
                for country_data in countries_data:
                    try:
                        country = WorldBankCountry(
                            id=country_data.get("id"),
                            iso2_code=country_data.get("iso2Code"),
                            name=country_data.get("name"),
                            region=Region(country_data.get("region", {}).get("id")),
                            income_group=IncomeGroup(country_data.get("incomeLevel", {}).get("id")),
                            capital_city=country_data.get("capitalCity"),
                            longitude=float(country_data.get("longitude")) if country_data.get("longitude") else None,
                            latitude=float(country_data.get("latitude")) if country_data.get("latitude") else None,
                            lending_type=country_data.get("lendingType", {}).get("id"),
                            last_updated=datetime.utcnow(),
                        )
                        self.countries[country.id] = country
                    except Exception as e:
                        logger.warning(f"Failed to parse country {country_data.get('id')}: {str(e)}")
                
                logger.info(f"Loaded {len(self.countries)} countries from API")
                
                # Cache for 30 days
                country_list = [c.dict() for c in self.countries.values()]
                await self.redis.setex(cache_key, 30 * 24 * 3600, json.dumps(country_list))
                
        except Exception as e:
            logger.error(f"Failed to load country metadata: {str(e)}")
            self.countries = {}
    
    async def _load_indicator_metadata(self) -> None:
        """Load World Bank indicator metadata."""
        # This would load additional metadata from World Bank API
        # For now, we use the pre-defined mapping
        pass
    
    async def _get_cache_key(self, request: WorldBankRequest) -> str:
        """Generate cache key for request."""
        request_dict = request.dict(exclude={'cache_ttl', 'include_metadata', 'include_source', 'include_footnotes'})
        request_json = json.dumps(request_dict, sort_keys=True)
        request_hash = hashlib.md5(request_json.encode()).hexdigest()
        return f"wb:data:{request.dataset.value}:{request_hash}"
    
    async def fetch_data(self, request: WorldBankRequest) -> WorldBankResponse:
        """
        Fetch data from World Bank APIs.
        
        Args:
            request: World Bank data request
            
        Returns:
            WorldBankResponse with data
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
                return WorldBankResponse(**response_data)
        
        try:
            # Route to appropriate API handler
            handler_map = {
                WorldBankDataset.WDI: self._fetch_wdi,
                WorldBankDataset.PROJECTS: self._fetch_projects,
                WorldBankDataset.CLIMATE: self._fetch_climate,
                WorldBankDataset.POVERTY: self._fetch_poverty,
                WorldBankDataset.EDUCATION: self._fetch_education,
                WorldBankDataset.HEALTH: self._fetch_health,
                WorldBankDataset.FINANCE: self._fetch_finance,
                WorldBankDataset.TRADE: self._fetch_trade,
                WorldBankDataset.INFRASTRUCTURE: self._fetch_infrastructure,
                WorldBankDataset.ENTERPRISE: self._fetch_enterprise,
                WorldBankDataset.GENDER: self._fetch_gender,
                WorldBankDataset.JOBS: self._fetch_jobs,
                WorldBankDataset.MACRO: self._fetch_macro,
                WorldBankDataset.DEBT: self._fetch_debt,
                WorldBankDataset.ENERGY: self._fetch_energy,
                WorldBankDataset.AGRICULTURE: self._fetch_agriculture,
            }
            
            handler = handler_map.get(request.dataset)
            if not handler:
                raise ValueError(f"Unsupported dataset: {request.dataset}")
            
            data = await handler(request)
            
            # Create response
            response = WorldBankResponse(
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
                cache_ttl = request.cache_ttl or self.default_cache_ttl
                await self.redis.setex(
                    cache_key,
                    cache_ttl,
                    json.dumps(response.dict())
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to fetch World Bank data: {str(e)}", exc_info=True)
            return WorldBankResponse(
                success=False,
                dataset=request.dataset,
                error=str(e),
                execution_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    # ============ Dataset Handlers ============
    
    async def _fetch_wdi(self, request: WorldBankRequest) -> Dict[str, Any]:
        """Fetch World Development Indicators data."""
        base_url = self.endpoints[WorldBankDataset.WDI]
        
        # Determine country code
        country_code = request.country_code or "all"
        if request.region:
            country_code = request.region.value
        elif request.income_group and request.income_group != IncomeGroup.ALL:
            country_code = request.income_group.value
        
        # Build endpoint URL
        if request.indicator:
            endpoint = f"/country/{country_code}/indicator/{request.indicator}"
        else:
            # List indicators
            endpoint = "/indicator"
        
        # Build query parameters
        params = {
            "format": request.format,
            "per_page": request.limit,
            "page": request.page or (request.offset // request.limit + 1),
        }
        
        if request.year:
            params["date"] = str(request.year)
        elif request.start_year and request.end_year:
            params["date"] = f"{request.start_year}:{request.end_year}"
        
        if request.include_source:
            params["source"] = "2"  # World Development Indicators source
        
        # Make request
        url = base_url + endpoint
        if params:
            url += "?" + urlencode(params)
        
        async with self.http_client.get(url) as response:
            if response.status != 200:
                raise Exception(f"World Bank API error: {response.status}")
            
            data = await response.json()
            
            # World Bank API returns array of arrays
            if not isinstance(data, list) or len(data) < 2:
                raise Exception("Invalid World Bank API response")
            
            metadata = data[0]
            wdi_data = data[1]
            
            # Transform data
            transformed_data = []
            for item in wdi_data:
                transformed_item = {
                    "indicator_id": item.get("indicator", {}).get("id"),
                    "indicator_name": item.get("indicator", {}).get("value"),
                    "country_id": item.get("country", {}).get("id"),
                    "country_name": item.get("country", {}).get("value"),
                    "year": int(item.get("date")) if item.get("date") else None,
                    "value": float(item.get("value")) if item.get("value") is not None else None,
                    "decimal": int(item.get("decimal")) if item.get("decimal") else None,
                    "unit": None,  # Would need to map from indicator
                    "footnote": item.get("footnote"),
                    "last_updated": item.get("lastupdated"),
                }
                
                # Add source info if available
                if request.include_source:
                    transformed_item["source"] = item.get("source", {}).get("value")
                
                transformed_data.append(transformed_item)
            
            return {
                "data": transformed_data,
                "metadata": {
                    "page": metadata.get("page"),
                    "pages": metadata.get("pages"),
                    "per_page": metadata.get("per_page"),
                    "total": metadata.get("total"),
                    "last_updated": metadata.get("lastupdated"),
                },
                "pagination": {
                    "page": metadata.get("page", 1),
                    "per_page": metadata.get("per_page", request.limit),
                    "total_pages": metadata.get("pages", 1),
                    "total_items": metadata.get("total"),
                },
                "total_count": metadata.get("total", len(transformed_data)),
            }
    
    async def _fetch_projects(self, request: WorldBankRequest) -> Dict[str, Any]:
        """Fetch World Bank projects data."""
        base_url = self.endpoints[WorldBankDataset.PROJECTS]
        
        # Build query parameters
        params = {
            "format": request.format,
            "rows": request.limit,
            "start": request.offset,
            "fl": "id,project_name,country_code,country_name,region_name,status,approval_date,closing_date,total_commitment,total_disbursement,sector_name,theme_name,implementing_agency,environmental_category,project_development_objective",
        }
        
        # Add filters
        filters = []
        
        if request.country_code:
            filters.append(f'country_code:"{request.country_code}"')
        
        if request.region:
            filters.append(f'region_name:"{request.region.value}"')
        
        if request.year:
            filters.append(f'approval_year:{request.year}')
        
        if filters:
            params["q"] = " AND ".join(filters)
        
        # Make request
        url = base_url
        if params:
            url += "?" + urlencode(params)
        
        async with self.http_client.get(url) as response:
            if response.status != 200:
                raise Exception(f"World Bank Projects API error: {response.status}")
            
            data = await response.json()
            
            # Transform projects data
            transformed_data = []
            for item in data.get("projects", []):
                transformed_item = {
                    "project_id": item.get("id"),
                    "project_name": item.get("project_name"),
                    "country_code": item.get("country_code"),
                    "country_name": item.get("country_name"),
                    "region": item.get("region_name"),
                    "status": item.get("status"),
                    "approval_date": item.get("approval_date"),
                    "closing_date": item.get("closing_date"),
                    "total_commitment": float(item.get("total_commitment")) if item.get("total_commitment") else None,
                    "total_disbursement": float(item.get("total_disbursement")) if item.get("total_disbursement") else None,
                    "sector": item.get("sector_name"),
                    "themes": item.get("theme_name", "").split(",") if item.get("theme_name") else [],
                    "implementing_agency": item.get("implementing_agency"),
                    "environmental_category": item.get("environmental_category"),
                    "project_development_objective": item.get("project_development_objective"),
                }
                transformed_data.append(transformed_item)
            
            return {
                "data": transformed_data,
                "metadata": {
                    "numFound": data.get("numFound"),
                    "start": data.get("start"),
                    "rows": data.get("rows"),
                },
                "pagination": {
                    "start": data.get("start", request.offset),
                    "rows": data.get("rows", request.limit),
                    "total_items": data.get("numFound"),
                },
                "total_count": data.get("numFound", len(transformed_data)),
            }
    
    async def _fetch_climate(self, request: WorldBankRequest) -> Dict[str, Any]:
        """Fetch climate change data."""
        base_url = self.endpoints[WorldBankDataset.CLIMATE]
        
        # Build query parameters
        params = {
            "format": request.format,
            "limit": request.limit,
            "offset": request.offset,
        }
        
        # Add dataset-specific parameters
        if request.indicator:
            params["indicator"] = request.indicator
        
        if request.country_code:
            params["country"] = request.country_code
        
        if request.year:
            params["year"] = request.year
        elif request.start_year and request.end_year:
            params["start_year"] = request.start_year
            params["end_year"] = request.end_year
        
        # Make request
        url = base_url + "/data"
        if params:
            url += "?" + urlencode(params)
        
        async with self.http_client.get(url) as response:
            if response.status != 200:
                raise Exception(f"World Bank Climate API error: {response.status}")
            
            data = await response.json()
            
            return {
                "data": data.get("data", []),
                "metadata": data.get("metadata", {}),
                "total_count": len(data.get("data", [])),
            }
    
    async def _fetch_poverty(self, request: WorldBankRequest) -> Dict[str, Any]:
        """Fetch poverty and inequality data."""
        # Use WDI endpoint with poverty indicators
        poverty_indicators = [
            "SI.POV.DDAY",  # $1.90/day poverty
            "SI.POV.UMIC",  # Upper middle income poverty
            "SI.POV.LMIC",  # Lower middle income poverty
            "SI.POV.GINI",  # Gini index
            "SI.POV.NAHC",  # Poverty headcount ratio at national poverty lines
        ]
        
        if request.indicator and request.indicator not in poverty_indicators:
            poverty_indicators = [request.indicator]
        
        # Fetch data for each indicator
        all_data = []
        for indicator in poverty_indicators:
            wdi_request = WorldBankRequest(
                dataset=WorldBankDataset.WDI,
                indicator=indicator,
                country_code=request.country_code,
                year=request.year,
                start_year=request.start_year,
                end_year=request.end_year,
                limit=request.limit,
                format=request.format,
            )
            
            wdi_response = await self._fetch_wdi(wdi_request)
            all_data.extend(wdi_response.get("data", []))
        
        return {
            "data": all_data,
            "metadata": {
                "indicators": poverty_indicators,
                "dataset": "Poverty and Inequality",
            },
            "total_count": len(all_data),
        }
    
    async def _fetch_education(self, request: WorldBankRequest) -> Dict[str, Any]:
        """Fetch education statistics."""
        education_indicators = [
            "SE.PRM.ENRR",  # Primary enrollment
            "SE.SEC.ENRR",  # Secondary enrollment
            "SE.TER.ENRR",  # Tertiary enrollment
            "SE.ADT.LITR.ZS",  # Adult literacy
            "SE.PRM.TCHR",  # Pupil-teacher ratio
            "SE.XPD.TOTL.GD.ZS",  # Education expenditure
        ]
        
        return await self._fetch_indicators_by_category(request, education_indicators, "Education")
    
    async def _fetch_health(self, request: WorldBankRequest) -> Dict[str, Any]:
        """Fetch health statistics."""
        health_indicators = [
            "SP.DYN.LE00.IN",  # Life expectancy
            "SH.DYN.MORT",  # Under-5 mortality
            "SH.STA.MMRT",  # Maternal mortality
            "SH.HIV.INCD",  # HIV incidence
            "SH.XPD.CHEX.GD.ZS",  # Health expenditure
            "SH.MED.BEDS.ZS",  # Hospital beds
        ]
        
        return await self._fetch_indicators_by_category(request, health_indicators, "Health")
    
    async def _fetch_finance(self, request: WorldBankRequest) -> Dict[str, Any]:
        """Fetch financial sector data."""
        finance_indicators = [
            "FD.RES.LIQU.AS.ZS",  # Liquid assets
            "FB.BNK.CAPA.ZS",  # Bank capital
            "FS.AST.DOMS.GD.ZS",  # Domestic credit
            "GFDD.DI.01",  # Bank deposits
            "GFDD.DI.14",  # Bank nonperforming loans
        ]
        
        return await self._fetch_indicators_by_category(request, finance_indicators, "Finance")
    
    async def _fetch_trade(self, request: WorldBankRequest) -> Dict[str, Any]:
        """Fetch trade statistics."""
        trade_indicators = [
            "NE.EXP.GNFS.ZS",  # Exports of goods and services
            "NE.IMP.GNFS.ZS",  # Imports of goods and services
            "NE.TRD.GNFS.ZS",  # Trade
            "TX.VAL.MRCH.XD.WD",  # Merchandise exports
            "TM.VAL.MRCH.XD.WD",  # Merchandise imports
        ]
        
        return await self._fetch_indicators_by_category(request, trade_indicators, "Trade")
    
    async def _fetch_infrastructure(self, request: WorldBankRequest) -> Dict[str, Any]:
        """Fetch infrastructure data."""
        infrastructure_indicators = [
            "EG.ELC.ACCS.ZS",  # Electricity access
            "IS.RRS.TOTL.KM",  # Railways
            "IS.ROD.TOTL.KM",  # Roads
            "IT.NET.USER.ZS",  # Internet users
            "IC.ELC.TIME",  # Time to get electricity
        ]
        
        return await self._fetch_indicators_by_category(request, infrastructure_indicators, "Infrastructure")
    
    async def _fetch_enterprise(self, request: WorldBankRequest) -> Dict[str, Any]:
        """Fetch enterprise surveys data."""
        # Enterprise surveys have different API structure
        base_url = "https://api.worldbank.org/v2"
        
        params = {
            "format": request.format,
            "per_page": request.limit,
            "page": request.page or (request.offset // request.limit + 1),
        }
        
        if request.country_code:
            params["country"] = request.country_code
        
        url = base_url + "/enterprisesurveys"
        if params:
            url += "?" + urlencode(params)
        
        async with self.http_client.get(url) as response:
            if response.status != 200:
                raise Exception(f"World Bank Enterprise Surveys API error: {response.status}")
            
            data = await response.json()
            
            # Process enterprise survey data
            transformed_data = []
            for item in data.get("surveys", []):
                transformed_data.append({
                    "survey_id": item.get("id"),
                    "country_code": item.get("country_code"),
                    "year": item.get("year"),
                    "status": item.get("status"),
                    "firms_sampled": item.get("firms_sampled"),
                    "response_rate": item.get("response_rate"),
                    "topics_covered": item.get("topics_covered", []),
                })
            
            return {
                "data": transformed_data,
                "metadata": data.get("metadata", {}),
                "total_count": len(transformed_data),
            }
    
    async def _fetch_gender(self, request: WorldBankRequest) -> Dict[str, Any]:
        """Fetch gender statistics."""
        gender_indicators = [
            "SG.GEN.PARL.ZS",  # Women in parliament
            "SL.TLF.TOTL.FE.ZS",  # Female labor force
            "SE.ENR.PRSC.FM.ZS",  # Gender parity in primary education
            "SH.STA.OWGH.FE.ZS",  # Women overweight
            "SP.DYN.LE00.FE.IN",  # Female life expectancy
        ]
        
        return await self._fetch_indicators_by_category(request, gender_indicators, "Gender")
    
    async def _fetch_jobs(self, request: WorldBankRequest) -> Dict[str, Any]:
        """Fetch jobs and labor data."""
        jobs_indicators = [
            "SL.UEM.TOTL.ZS",  # Unemployment rate
            "SL.TLF.TOTL.IN",  # Labor force
            "SL.EMP.TOTL.SP.ZS",  # Employment to population
            "SL.AGR.EMPL.ZS",  # Employment in agriculture
            "SL.IND.EMPL.ZS",  # Employment in industry
        ]
        
        return await self._fetch_indicators_by_category(request, jobs_indicators, "Jobs")
    
    async def _fetch_macro(self, request: WorldBankRequest) -> Dict[str, Any]:
        """Fetch macroeconomic data."""
        macro_indicators = [
            "NY.GDP.MKTP.CD",  # GDP
            "NY.GDP.MKTP.KD.ZG",  # GDP growth
            "NY.GDP.PCAP.CD",  # GDP per capita
            "FP.CPI.TOTL.ZG",  # Inflation
            "GC.DOD.TOTL.GD.ZS",  # Government debt
        ]
        
        return await self._fetch_indicators_by_category(request, macro_indicators, "Macroeconomic")
    
    async def _fetch_debt(self, request: WorldBankRequest) -> Dict[str, Any]:
        """Fetch debt statistics."""
        debt_indicators = [
            "DT.DOD.DECT.GN.ZS",  # External debt stocks
            "DT.TDS.DECT.GN.ZS",  # Total debt service
            "DT.INT.DECT.GN.ZS",  # Interest payments
            "GC.DOD.TOTL.GD.ZS",  # Government debt
            "DT.DOD.PVLX.EX.ZS",  # Present value of external debt
        ]
        
        return await self._fetch_indicators_by_category(request, debt_indicators, "Debt")
    
    async def _fetch_energy(self, request: WorldBankRequest) -> Dict[str, Any]:
        """Fetch energy statistics."""
        energy_indicators = [
            "EG.USE.PCAP.KG.OE",  # Energy use per capita
            "EG.IMP.CONS.ZS",  # Energy imports
            "EG.FEC.RNEW.ZS",  # Renewable energy consumption
            "EG.ELC.FOSL.ZS",  # Fossil fuel electricity
            "EN.ATM.CO2E.PC",  # CO2 emissions
        ]
        
        return await self._fetch_indicators_by_category(request, energy_indicators, "Energy")
    
    async def _fetch_agriculture(self, request: WorldBankRequest) -> Dict[str, Any]:
        """Fetch agriculture and rural development data."""
        agriculture_indicators = [
            "AG.LND.AGRI.ZS",  # Agricultural land
            "AG.PRD.LVSK.XD",  # Livestock production
            "AG.PRD.CROP.XD",  # Crop production
            "AG.CON.FERT.ZS",  # Fertilizer consumption
            "ER.H2O.FWAG.ZS",  # Freshwater withdrawal for agriculture
        ]
        
        return await self._fetch_indicators_by_category(request, agriculture_indicators, "Agriculture")
    
    async def _fetch_indicators_by_category(self, request: WorldBankRequest, indicators: List[str], category_name: str) -> Dict[str, Any]:
        """Fetch data for a list of indicators."""
        all_data = []
        
        for indicator in indicators:
            if request.indicator and request.indicator != indicator:
                continue
            
            wdi_request = WorldBankRequest(
                dataset=WorldBankDataset.WDI,
                indicator=indicator,
                country_code=request.country_code,
                year=request.year,
                start_year=request.start_year,
                end_year=request.end_year,
                limit=request.limit,
                format=request.format,
            )
            
            try:
                wdi_response = await self._fetch_wdi(wdi_request)
                all_data.extend(wdi_response.get("data", []))
                
                # Break if we have enough data
                if len(all_data) >= request.limit:
                    all_data = all_data[:request.limit]
                    break
            except Exception as e:
                logger.warning(f"Failed to fetch indicator {indicator}: {str(e)}")
                continue
        
        return {
            "data": all_data,
            "metadata": {
                "category": category_name,
                "indicators": indicators,
            },
            "total_count": len(all_data),
        }
    
    # ============ Utility Methods ============
    
    async def get_country_info(self, country_code: str) -> Optional[WorldBankCountry]:
        """Get World Bank country information by code."""
        if not self.countries:
            await self._load_country_metadata()
        
        return self.countries.get(country_code.upper())
    
    async def search_countries(self, query: str, limit: int = 10) -> List[WorldBankCountry]:
        """Search for countries by name or code."""
        if not self.countries:
            await self._load_country_metadata()
        
        query = query.lower()
        results = []
        
        for country in self.countries.values():
            if (query in country.name.lower() or 
                query in country.id.lower() or 
                (country.iso2_code and query in country.iso2_code.lower())):
                results.append(country)
            
            if len(results) >= limit:
                break
        
        return results
    
    async def get_indicator_info(self, indicator_code: str) -> Optional[WorldBankIndicator]:
        """Get World Bank indicator information by code."""
        return self.indicator_mapping.get(indicator_code)
    
    async def search_indicators(self, query: str, category: Optional[IndicatorCategory] = None) -> List[WorldBankIndicator]:
        """Search for World Bank indicators."""
        query = query.lower()
        results = []
        
        for indicator in self.indicator_mapping.values():
            if (query in indicator.name.lower() or 
                query in indicator.id.lower() or
                (indicator.description and query in indicator.description.lower())):
                
                if category and indicator.category != category:
                    continue
                
                results.append(indicator)
        
        # Sort by relevance
        results.sort(key=lambda x: (
            query in x.name.lower(),
            query in x.id.lower(),
            len(x.name)
        ), reverse=True)
        
        return results[:50]  # Limit results
    
    async def get_indicator_time_series(self, indicator_code: str, country_code: str, 
                                      start_year: Optional[int] = None, 
                                      end_year: Optional[int] = None) -> List[WorldBankDataPoint]:
        """Get time series data for an indicator."""
        current_year = datetime.now().year
        
        if not start_year:
            start_year = 1960  # World Bank data typically starts around 1960
        if not end_year:
            end_year = current_year - 1  # Most recent complete year
        
        request = WorldBankRequest(
            dataset=WorldBankDataset.WDI,
            indicator=indicator_code,
            country_code=country_code,
            start_year=start_year,
            end_year=end_year,
            limit=1000,  # Get as many years as possible
        )
        
        response = await self.fetch_data(request)
        
        if not response.success:
            logger.error(f"Failed to fetch time series: {response.error}")
            return []
        
        # Convert to data points
        data_points = []
        for item in response.data:
            if item.get("year") and item.get("value") is not None:
                data_points.append(WorldBankDataPoint(
                    year=item["year"],
                    value=item["value"],
                    decimal=item.get("decimal"),
                    footnote=item.get("footnote"),
                ))
        
        # Sort by year
        data_points.sort(key=lambda x: x.year)
        
        return data_points
    
    async def get_country_profile(self, country_code: str) -> Dict[str, Any]:
        """
        Get comprehensive country profile with key indicators.
        
        Args:
            country_code: ISO3 country code
            
        Returns:
            Country profile data
        """
        # Get country info
        country_info = await self.get_country_info(country_code)
        if not country_info:
            return {"error": f"Country not found: {country_code}"}
        
        # Key indicators to fetch
        key_indicators = {
            "gdp": "NY.GDP.MKTP.CD",
            "gdp_per_capita": "NY.GDP.PCAP.CD",
            "gdp_growth": "NY.GDP.MKTP.KD.ZG",
            "population": "SP.POP.TOTL",
            "population_growth": "SP.POP.GROW",
            "life_expectancy": "SP.DYN.LE00.IN",
            "poverty": "SI.POV.DDAY",
            "gini": "SI.POV.GINI",
            "inflation": "FP.CPI.TOTL.ZG",
            "unemployment": "SL.UEM.TOTL.ZS",
            "exports": "NE.EXP.GNFS.ZS",
            "imports": "NE.IMP.GNFS.ZS",
            "debt": "GC.DOD.TOTL.GD.ZS",
            "co2_emissions": "EN.ATM.CO2E.PC",
        }
        
        # Fetch latest values for each indicator
        current_year = datetime.now().year - 1  # Most recent complete year
        
        indicator_data = {}
        for key, indicator_code in key_indicators.items():
            request = WorldBankRequest(
                dataset=WorldBankDataset.WDI,
                indicator=indicator_code,
                country_code=country_code,
                year=current_year,
                limit=1,
            )
            
            response = await self.fetch_data(request)
            if response.success and response.data:
                data = response.data[0]
                indicator_data[key] = {
                    "value": data.get("value"),
                    "year": data.get("year"),
                    "indicator_name": data.get("indicator_name"),
                    "unit": self.indicator_mapping.get(indicator_code, WorldBankIndicator(id=indicator_code, name="", category=IndicatorCategory.ECONOMIC)).unit,
                }
        
        # Get time series for GDP and population (last 10 years)
        gdp_series = await self.get_indicator_time_series("NY.GDP.MKTP.CD", country_code, current_year - 10, current_year)
        population_series = await self.get_indicator_time_series("SP.POP.TOTL", country_code, current_year - 10, current_year)
        
        # Get projects
        projects_request = WorldBankRequest(
            dataset=WorldBankDataset.PROJECTS,
            country_code=country_code,
            limit=10,
        )
        
        projects_response = await self.fetch_data(projects_request)
        projects = projects_response.data if projects_response.success else []
        
        return {
            "country_info": country_info.dict(),
            "indicators": indicator_data,
            "time_series": {
                "gdp": [{"year": p.year, "value": p.value} for p in gdp_series],
                "population": [{"year": p.year, "value": p.value} for p in population_series],
            },
            "projects": projects[:5],  # Top 5 projects
            "last_updated": datetime.utcnow().isoformat(),
        }
    
    async def compare_countries(self, country_codes: List[str], indicator_code: str, year: Optional[int] = None) -> Dict[str, Any]:
        """Compare multiple countries on a specific indicator."""
        if not year:
            year = datetime.now().year - 1  # Most recent complete year
        
        tasks = []
        for country_code in country_codes:
            request = WorldBankRequest(
                dataset=WorldBankDataset.WDI,
                indicator=indicator_code,
                country_code=country_code,
                year=year,
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
                })
        
        # Sort by value (descending)
        comparison_data.sort(key=lambda x: x.get('value') or 0, reverse=True)
        
        # Get indicator info
        indicator_info = self.indicator_mapping.get(indicator_code)
        
        return {
            'indicator_code': indicator_code,
            'indicator_name': indicator_info.name if indicator_info else indicator_code,
            'year': year,
            'comparison': comparison_data,
            'timestamp': datetime.utcnow().isoformat(),
        }
    
    async def get_regional_average(self, region: Region, indicator_code: str, year: Optional[int] = None) -> Optional[float]:
        """Get regional average for an indicator."""
        if not year:
            year = datetime.now().year - 1
        
        request = WorldBankRequest(
            dataset=WorldBankDataset.WDI,
            indicator=indicator_code,
            region=region,
            year=year,
            limit=100,  # Get all countries in region
        )
        
        response = await self.fetch_data(request)
        
        if not response.success or not response.data:
            return None
        
        # Calculate average (excluding null values)
        values = [item.get('value') for item in response.data if item.get('value') is not None]
        
        if not values:
            return None
        
        return sum(values) / len(values)
    
    async def get_income_group_average(self, income_group: IncomeGroup, indicator_code: str, year: Optional[int] = None) -> Optional[float]:
        """Get income group average for an indicator."""
        if not year:
            year = datetime.now().year - 1
        
        request = WorldBankRequest(
            dataset=WorldBankDataset.WDI,
            indicator=indicator_code,
            income_group=income_group,
            year=year,
            limit=100,
        )
        
        response = await self.fetch_data(request)
        
        if not response.success or not response.data:
            return None
        
        # Calculate average
        values = [item.get('value') for item in response.data if item.get('value') is not None]
        
        if not values:
            return None
        
        return sum(values) / len(values)
    
    async def get_world_average(self, indicator_code: str, year: Optional[int] = None) -> Optional[float]:
        """Get world average for an indicator."""
        return await self.get_regional_average(Region.WORLD, indicator_code, year)
    
    async def get_historical_trend(self, indicator_code: str, country_code: str, years: int = 20) -> Dict[str, Any]:
        """Get historical trend for an indicator."""
        current_year = datetime.now().year - 1  # Most recent complete year
        start_year = current_year - years
        
        data_points = await self.get_indicator_time_series(indicator_code, country_code, start_year, current_year)
        
        if not data_points:
            return {"error": "No data available"}
        
        # Calculate trend statistics
        values = [p.value for p in data_points if p.value is not None]
        
        if len(values) < 2:
            return {
                "data_points": [p.dict() for p in data_points],
                "trend": "insufficient_data",
            }
        
        # Simple linear trend calculation
        x = list(range(len(values)))
        y = values
        
        # Calculate slope (simple linear regression)
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi ** 2 for xi in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        
        # Determine trend
        if slope > 0.01:
            trend = "increasing"
        elif slope < -0.01:
            trend = "decreasing"
        else:
            trend = "stable"
        
        # Calculate percentage change
        if values[0] and values[-1] and values[0] != 0:
            percent_change = ((values[-1] - values[0]) / values[0]) * 100
        else:
            percent_change = None
        
        return {
            "data_points": [p.dict() for p in data_points],
            "trend": trend,
            "slope": slope,
            "percent_change": percent_change,
            "start_value": values[0],
            "end_value": values[-1],
            "years": years,
        }
    
    async def find_similar_countries(self, country_code: str, indicators: List[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Find countries similar to the given country based on indicators."""
        if not indicators:
            # Default indicators for similarity
            indicators = ["NY.GDP.PCAP.CD", "SP.POP.TOTL", "SP.DYN.LE00.IN", "SE.ADT.LITR.ZS"]
        
        # Get target country data
        target_data = {}
        for indicator in indicators:
            request = WorldBankRequest(
                dataset=WorldBankDataset.WDI,
                indicator=indicator,
                country_code=country_code,
                limit=1,
            )
            
            response = await self.fetch_data(request)
            if response.success and response.data:
                target_data[indicator] = response.data[0].get('value')
        
        if not target_data:
            return []
        
        # Get all countries data for these indicators
        all_countries_data = {}
        
        for indicator in indicators:
            request = WorldBankRequest(
                dataset=WorldBankDataset.WDI,
                indicator=indicator,
                limit=200,  # Get data for all countries
            )
            
            response = await self.fetch_data(request)
            if response.success:
                for item in response.data:
                    country = item.get('country_id')
                    value = item.get('value')
                    
                    if country not in all_countries_data:
                        all_countries_data[country] = {}
                    
                    all_countries_data[country][indicator] = value
        
        # Calculate similarity scores
        similarity_scores = []
        
        for country, data in all_countries_data.items():
            if country == country_code:
                continue
            
            # Calculate Euclidean distance for available indicators
            distance = 0
            available_indicators = 0
            
            for indicator in indicators:
                target_val = target_data.get(indicator)
                country_val = data.get(indicator)
                
                if target_val is not None and country_val is not None:
                    # Normalize values to 0-1 range (crude normalization)
                    # In production, you'd want better normalization
                    max_val = max(abs(target_val), abs(country_val), 1)
                    norm_target = target_val / max_val
                    norm_country = country_val / max_val
                    
                    distance += (norm_target - norm_country) ** 2
                    available_indicators += 1
            
            if available_indicators > 0:
                # Normalize distance by number of indicators
                distance = (distance / available_indicators) ** 0.5
                similarity = 1 / (1 + distance)  # Convert to similarity score
                
                similarity_scores.append({
                    'country_code': country,
                    'similarity_score': similarity,
                    'available_indicators': available_indicators,
                    'data': data,
                })
        
        # Sort by similarity score
        similarity_scores.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return similarity_scores[:limit]
    
    async def clear_cache(self, pattern: str = "wb:*") -> int:
        """Clear World Bank cache."""
        keys = await self.redis.keys(pattern)
        if keys:
            deleted = await self.redis.delete(*keys)
            logger.info(f"Cleared {deleted} World Bank cache keys")
            return deleted
        return 0


# ============ Factory Function ============

_wb_client: Optional[WorldBankClient] = None

async def get_worldbank_client() -> WorldBankClient:
    """
    Get or create a World Bank client singleton.
    
    Returns:
        WorldBankClient instance
    """
    global _wb_client
    
    if _wb_client is None:
        _wb_client = WorldBankClient()
        await _wb_client.initialize()
    
    return _wb_client


# ============ Utility Functions ============

async def get_country_economic_profile(country_code: str) -> Dict[str, Any]:
    """
    Get economic profile for a country.
    
    Args:
        country_code: ISO3 country code
        
    Returns:
        Economic profile data
    """
    client = await get_worldbank_client()
    
    # Get country info
    country_info = await client.get_country_info(country_code)
    if not country_info:
        return {"error": f"Country not found: {country_code}"}
    
    # Key economic indicators
    economic_indicators = [
        ("NY.GDP.MKTP.CD", "gdp"),
        ("NY.GDP.MKTP.KD.ZG", "gdp_growth"),
        ("NY.GDP.PCAP.CD", "gdp_per_capita"),
        ("FP.CPI.TOTL.ZG", "inflation"),
        ("SL.UEM.TOTL.ZS", "unemployment"),
        ("NE.EXP.GNFS.ZS", "exports"),
        ("NE.IMP.GNFS.ZS", "imports"),
        ("GC.DOD.TOTL.GD.ZS", "government_debt"),
        ("DT.DOD.DECT.GN.ZS", "external_debt"),
        ("BN.KLT.DINV.CD", "fdi_inflows"),
    ]
    
    # Fetch latest values
    current_year = datetime.now().year - 1
    indicator_data = {}
    
    for indicator_code, key in economic_indicators:
        request = WorldBankRequest(
            dataset=WorldBankDataset.WDI,
            indicator=indicator_code,
            country_code=country_code,
            year=current_year,
            limit=1,
        )
        
        response = await client.fetch_data(request)
        if response.success and response.data:
            data = response.data[0]
            indicator_data[key] = {
                "value": data.get("value"),
                "year": data.get("year"),
                "indicator_name": data.get("indicator_name"),
            }
    
    # Get GDP trend
    gdp_trend = await client.get_historical_trend("NY.GDP.MKTP.CD", country_code, years=10)
    
    # Compare with regional average
    regional_avg = await client.get_regional_average(country_info.region, "NY.GDP.PCAP.CD", current_year)
    
    # Compare with income group average
    income_avg = await client.get_income_group_average(country_info.income_group, "NY.GDP.PCAP.CD", current_year)
    
    return {
        "country_info": country_info.dict(),
        "economic_indicators": indicator_data,
        "gdp_trend": gdp_trend,
        "comparisons": {
            "regional_gdp_per_capita_avg": regional_avg,
            "income_group_gdp_per_capita_avg": income_avg,
        },
        "analysis_year": current_year,
        "last_updated": datetime.utcnow().isoformat(),
    }


async def get_poverty_analysis(country_code: str) -> Dict[str, Any]:
    """
    Get poverty analysis for a country.
    
    Args:
        country_code: ISO3 country code
        
    Returns:
        Poverty analysis data
    """
    client = await get_worldbank_client()
    
    # Poverty indicators
    poverty_indicators = {
        "extreme_poverty": "SI.POV.DDAY",  # $1.90/day
        "national_poverty": "SI.POV.NAHC",  # National poverty line
        "gini_index": "SI.POV.GINI",  # Inequality
        "income_share_lowest_20": "SI.DST.FRST.20",  # Income share lowest 20%
        "income_share_highest_20": "SI.DST.05TH.20",  # Income share highest 20%
    }
    
    # Fetch data
    current_year = datetime.now().year - 1
    poverty_data = {}
    
    for key, indicator_code in poverty_indicators.items():
        request = WorldBankRequest(
            dataset=WorldBankDataset.WDI,
            indicator=indicator_code,
            country_code=country_code,
            year=current_year,
            limit=1,
        )
        
        response = await client.fetch_data(request)
        if response.success and response.data:
            data = response.data[0]
            poverty_data[key] = {
                "value": data.get("value"),
                "year": data.get("year"),
            }
    
    # Get regional comparison
    region = (await client.get_country_info(country_code)).region
    regional_poverty = await client.get_regional_average(region, "SI.POV.DDAY", current_year)
    
    # Get trend
    poverty_trend = await client.get_historical_trend("SI.POV.DDAY", country_code, years=15)
    
    return {
        "poverty_data": poverty_data,
        "regional_comparison": {
            "regional_extreme_poverty_avg": regional_poverty,
            "region": region.value,
        },
        "poverty_trend": poverty_trend,
        "analysis_year": current_year,
        "last_updated": datetime.utcnow().isoformat(),
    }


async def get_climate_risk_profile(country_code: str) -> Dict[str, Any]:
    """
    Get climate risk profile for a country.
    
    Args:
        country_code: ISO3 country code
        
    Returns:
        Climate risk profile
    """
    client = await get_worldbank_client()
    
    # Climate indicators
    climate_indicators = {
        "co2_emissions_per_capita": "EN.ATM.CO2E.PC",
        "co2_emissions_total": "EN.ATM.CO2E.KT",
        "renewable_energy_consumption": "EG.FEC.RNEW.ZS",
        "forest_area": "AG.LND.FRST.ZS",
        "freshwater_withdrawal": "ER.H2O.FWTL.ZS",
        "disaster_risk_reduction": "VC.DRR.PUBL.GD.ZS",
    }
    
    # Fetch data
    current_year = datetime.now().year - 1
    climate_data = {}
    
    for key, indicator_code in climate_indicators.items():
        request = WorldBankRequest(
            dataset=WorldBankDataset.WDI,
            indicator=indicator_code,
            country_code=country_code,
            year=current_year,
            limit=1,
        )
        
        response = await client.fetch_data(request)
        if response.success and response.data:
            data = response.data[0]
            climate_data[key] = {
                "value": data.get("value"),
                "year": data.get("year"),
                "indicator_name": data.get("indicator_name"),
            }
    
    # Get climate projects
    projects_request = WorldBankRequest(
        dataset=WorldBankDataset.PROJECTS,
        country_code=country_code,
        limit=20,
    )
    
    projects_response = await client.fetch_data(projects_request)
    climate_projects = []
    
    if projects_response.success:
        for project in projects_response.data:
            if any(theme.lower() in ["climate", "environment", "disaster"] for theme in project.get("themes", [])):
                climate_projects.append(project)
    
    # Get regional comparison
    region = (await client.get_country_info(country_code)).region
    regional_co2 = await client.get_regional_average(region, "EN.ATM.CO2E.PC", current_year)
    
    return {
        "climate_indicators": climate_data,
        "climate_projects": climate_projects[:5],
        "regional_comparison": {
            "regional_co2_per_capita_avg": regional_co2,
            "region": region.value,
        },
        "analysis_year": current_year,
        "last_updated": datetime.utcnow().isoformat(),
    }


async def generate_country_briefing(country_code: str) -> Dict[str, Any]:
    """
    Generate a comprehensive country briefing.
    
    Args:
        country_code: ISO3 country code
        
    Returns:
        Country briefing data
    """
    client = await get_worldbank_client()
    
    # Get all data in parallel
    economic_profile = get_country_economic_profile(country_code)
    poverty_analysis = get_poverty_analysis(country_code)
    climate_profile = get_climate_risk_profile(country_code)
    
    results = await asyncio.gather(
        economic_profile,
        poverty_analysis,
        climate_profile,
        return_exceptions=True
    )
    
    # Combine results
    briefing = {
        "country_code": country_code,
        "generated_at": datetime.utcnow().isoformat(),
    }
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Error generating briefing component: {str(result)}")
            continue
        
        if i == 0 and "error" not in result:
            briefing["economic_profile"] = result
        elif i == 1 and "error" not in result:
            briefing["poverty_analysis"] = result
        elif i == 2 and "error" not in result:
            briefing["climate_profile"] = result
    
    # Generate key insights
    insights = []
    
    if "economic_profile" in briefing:
        econ = briefing["economic_profile"]
        if "economic_indicators" in econ:
            indicators = econ["economic_indicators"]
            
            if "gdp_growth" in indicators and indicators["gdp_growth"]["value"]:
                growth = indicators["gdp_growth"]["value"]
                if growth > 5:
                    insights.append(f"Strong economic growth: {growth}% GDP growth")
                elif growth < 0:
                    insights.append(f"Economic contraction: {growth}% GDP growth")
            
            if "inflation" in indicators and indicators["inflation"]["value"]:
                inflation = indicators["inflation"]["value"]
                if inflation > 10:
                    insights.append(f"High inflation: {inflation}%")
    
    if "poverty_analysis" in briefing:
        poverty = briefing["poverty_analysis"]
        if "poverty_data" in poverty and "extreme_poverty" in poverty["poverty_data"]:
            extreme_pov = poverty["poverty_data"]["extreme_poverty"]["value"]
            if extreme_pov and extreme_pov > 20:
                insights.append(f"High extreme poverty: {extreme_pov}% of population")
    
    briefing["key_insights"] = insights[:5]  # Top 5 insights
    
    return briefing