# backend/app/integrations/imf_client.py
"""
International Monetary Fund (IMF) API integration for WorldBrief 360.
Provides access to IMF economic and financial data including WEO, IFS, and other datasets.
"""

import asyncio
import json
import csv
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from io import StringIO
from decimal import Decimal

import aiohttp
import pandas as pd
from pydantic import BaseModel, Field, validator, HttpUrl
from tenacity import retry, stop_after_attempt, wait_exponential
from cachetools import TTLCache
import xml.etree.ElementTree as ET

from app.core.config import settings
from app.core.logging_config import logger
from app.services.utils.http_client import AsyncHTTPClient
from app.schemas.request.economics import EconomicDataQuery


class IMFDataset(Enum):
    """IMF datasets available through API."""
    WEO = "WEO"  # World Economic Outlook
    IFS = "IFS"  # International Financial Statistics
    GFS = "GFS"  # Government Finance Statistics
    BOP = "BOP"  # Balance of Payments
    DOTS = "DOTS"  # Direction of Trade Statistics
    CPI = "CPI"  # Consumer Price Index
    PSR = "PSR"  # Primary Commodity Prices
    FM = "FM"  # Financial Soundness Indicators
    APDREO = "APDREO"  # Asia and Pacific Regional Economic Outlook
    AFRREO = "AFRREO"  # African Regional Economic Outlook
    WHDREO = "WHDREO"  # Western Hemisphere Regional Economic Outlook


class IMFCountryCode(Enum):
    """Common IMF country codes."""
    WORLD = "W00"
    UNITED_STATES = "111"
    CHINA = "924"
    JAPAN = "158"
    GERMANY = "134"
    UNITED_KINGDOM = "112"
    INDIA = "534"
    FRANCE = "132"
    ITALY = "136"
    BRAZIL = "223"
    CANADA = "156"
    RUSSIA = "922"
    AUSTRALIA = "193"
    SOUTH_KOREA = "542"
    MEXICO = "273"
    INDONESIA = "536"
    SAUDI_ARABIA = "456"
    TURKEY = "186"
    SWITZERLAND = "146"
    NETHERLANDS = "138"
    SPAIN = "184"


class IMFIndicator(Enum):
    """Common IMF economic indicators."""
    # GDP and National Accounts
    GDP_CURRENT_USD = "NGDPD"
    GDP_PPP_CURRENT_INTL = "PPPGDP"
    GDP_REAL_GROWTH = "NGDP_RPCH"
    GDP_PER_CAPITA_USD = "NGDPDPC"
    GDP_PER_CAPITA_PPP = "PPPGDPPC"
    
    # Fiscal Indicators
    GOV_DEBT_PERCENT_GDP = "GGXWDG_NGDP"
    GOV_REVENUE_PERCENT_GDP = "GGR_NGDP"
    GOV_EXPENDITURE_PERCENT_GDP = "GGX_NGDP"
    GOV_BALANCE_PERCENT_GDP = "GGXCNL_NGDP"
    
    # Monetary Indicators
    INFLATION_CPI = "PCPI_PCH"
    INTEREST_RATE = "FPOLM_PA"
    MONEY_SUPPLY_M2 = "FMB_PA"
    EXCHANGE_RATE = "ENDA_XDC_USD_RATE"
    
    # Trade and Balance of Payments
    CURRENT_ACCOUNT_BALANCE_PERCENT_GDP = "BCA_NGDPD"
    TRADE_BALANCE_PERCENT_GDP = "TBG_BP6_USD"
    EXPORTS_GOODS_SERVICES = "TXG_BP6_USD"
    IMPORTS_GOODS_SERVICES = "TMG_BP6_USD"
    
    # Financial Indicators
    UNEMPLOYMENT_RATE = "LUR_PT"
    POPULATION = "LP_PE"
    RESERVES = "RAXG_USD"


@dataclass
class IMFCountry:
    """IMF country/area information."""
    code: str
    name: str
    region: str
    income_group: Optional[str] = None
    currency_code: Optional[str] = None
    iso2_code: Optional[str] = None
    iso3_code: Optional[str] = None


@dataclass
class IMFIndicatorInfo:
    """IMF indicator information."""
    code: str
    name: str
    description: str
    unit: str
    scale: str
    dataset: IMFDataset
    frequency: str  # A=Annual, Q=Quarterly, M=Monthly
    notes: Optional[str] = None


class IMFDataPoint(BaseModel):
    """Single data point from IMF."""
    country_code: str
    country_name: str
    indicator_code: str
    indicator_name: str
    year: int
    period: Optional[str] = None  # For quarterly/monthly data
    value: Optional[Decimal] = None
    unit: str
    scale: str
    dataset: IMFDataset
    last_updated: Optional[datetime] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True
    
    @validator('value', pre=True)
    def parse_value(cls, v):
        if v is None or v == "":
            return None
        try:
            return Decimal(str(v))
        except:
            return None


class IMFDataSeries(BaseModel):
    """Time series data from IMF."""
    country_code: str
    country_name: str
    indicator_code: str
    indicator_name: str
    dataset: IMFDataset
    frequency: str
    unit: str
    scale: str
    data_points: List[IMFDataPoint] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        data = []
        for point in self.data_points:
            data.append({
                'year': point.year,
                'period': point.period,
                'value': float(point.value) if point.value else None,
                'country_code': point.country_code,
                'country_name': point.country_name,
                'indicator_code': point.indicator_code,
                'indicator_name': point.indicator_name,
                'unit': point.unit,
                'scale': point.scale
            })
        return pd.DataFrame(data)
    
    def get_latest_value(self) -> Optional[IMFDataPoint]:
        """Get the most recent data point."""
        if not self.data_points:
            return None
        return max(self.data_points, key=lambda x: (x.year, x.period or ""))
    
    def filter_by_years(self, start_year: int, end_year: int) -> 'IMFDataSeries':
        """Filter data points by year range."""
        filtered_points = [
            point for point in self.data_points
            if start_year <= point.year <= end_year
        ]
        return IMFDataSeries(
            country_code=self.country_code,
            country_name=self.country_name,
            indicator_code=self.indicator_code,
            indicator_name=self.indicator_name,
            dataset=self.dataset,
            frequency=self.frequency,
            unit=self.unit,
            scale=self.scale,
            data_points=filtered_points,
            metadata=self.metadata
        )


class IMFClient:
    """
    Client for IMF Data API (JSON REST API).
    Provides access to IMF's extensive economic and financial databases.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "http://dataservices.imf.org/REST/SDMX_JSON.svc",
        timeout: int = 60,
        max_retries: int = 3,
        cache_ttl: int = 86400  # 24 hours cache
    ):
        """
        Initialize IMF client.
        
        Args:
            api_key: IMF API key (optional for public datasets)
            base_url: IMF API base URL
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            cache_ttl: Cache TTL in seconds
        """
        self.api_key = api_key or settings.IMF_API_KEY
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        
        self.http_client = AsyncHTTPClient(
            timeout=timeout,
            retries=max_retries
        )
        
        # Caches
        self.dataset_cache = TTLCache(maxsize=50, ttl=cache_ttl)
        self.country_cache = TTLCache(maxsize=200, ttl=cache_ttl * 7)  # 7 days
        self.indicator_cache = TTLCache(maxsize=500, ttl=cache_ttl)
        self.data_cache = TTLCache(maxsize=1000, ttl=cache_ttl // 2)  # 12 hours
        
        # Country and indicator mappings
        self._country_mapping: Dict[str, IMFCountry] = {}
        self._indicator_mapping: Dict[str, IMFIndicatorInfo] = {}
        
        # Initialize metadata
        asyncio.create_task(self._initialize_metadata())
        
        logger.info("IMF client initialized")
    
    async def _initialize_metadata(self):
        """Initialize country and indicator metadata."""
        try:
            await self._load_country_mapping()
            await self._load_indicator_mapping()
            logger.info("IMF metadata initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize IMF metadata: {str(e)}")
    
    async def _load_country_mapping(self):
        """Load country code mapping."""
        try:
            # Try to get country list from IFS dataset
            url = f"{self.base_url}/DataStructure/IFS"
            
            async with self.http_client.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    countries = self._parse_countries_from_structure(data)
                    for country in countries:
                        self._country_mapping[country.code] = country
                    
                    logger.info(f"Loaded {len(countries)} countries from IMF")
                else:
                    logger.warning("Could not load country metadata from IMF API")
                    # Load fallback mapping
                    self._load_fallback_country_mapping()
                    
        except Exception as e:
            logger.error(f"Error loading country mapping: {str(e)}")
            self._load_fallback_country_mapping()
    
    def _load_fallback_country_mapping(self):
        """Load fallback country mapping."""
        # Common countries with their IMF codes
        fallback_countries = [
            IMFCountry(code="111", name="United States", region="Advanced Economies"),
            IMFCountry(code="924", name="China", region="Emerging and Developing Economies"),
            IMFCountry(code="158", name="Japan", region="Advanced Economies"),
            IMFCountry(code="134", name="Germany", region="Advanced Economies"),
            IMFCountry(code="112", name="United Kingdom", region="Advanced Economies"),
            IMFCountry(code="534", name="India", region="Emerging and Developing Economies"),
            IMFCountry(code="132", name="France", region="Advanced Economies"),
            IMFCountry(code="136", name="Italy", region="Advanced Economies"),
            IMFCountry(code="223", name="Brazil", region="Emerging and Developing Economies"),
            IMFCountry(code="156", name="Canada", region="Advanced Economies"),
            IMFCountry(code="922", name="Russian Federation", region="Emerging and Developing Economies"),
            IMFCountry(code="193", name="Australia", region="Advanced Economies"),
            IMFCountry(code="542", name="Korea", region="Advanced Economies"),
            IMFCountry(code="273", name="Mexico", region="Emerging and Developing Economies"),
            IMFCountry(code="536", name="Indonesia", region="Emerging and Developing Economies"),
            IMFCountry(code="456", name="Saudi Arabia", region="Emerging and Developing Economies"),
            IMFCountry(code="186", name="Turkey", region="Emerging and Developing Economies"),
            IMFCountry(code="146", name="Switzerland", region="Advanced Economies"),
            IMFCountry(code="138", name="Netherlands", region="Advanced Economies"),
            IMFCountry(code="184", name="Spain", region="Advanced Economies"),
        ]
        
        for country in fallback_countries:
            self._country_mapping[country.code] = country
    
    def _parse_countries_from_structure(self, data: Dict[str, Any]) -> List[IMFCountry]:
        """Parse country information from data structure."""
        countries = []
        
        try:
            # IMF API structure varies by dataset
            # This is a simplified parser
            if 'Structure' in data and 'CodeLists' in data['Structure']:
                code_lists = data['Structure']['CodeLists']['CodeList']
                for code_list in code_lists:
                    if isinstance(code_list, dict) and code_list.get('@id') == 'CL_AREA':
                        codes = code_list.get('Code', [])
                        if not isinstance(codes, list):
                            codes = [codes]
                        
                        for code in codes:
                            if isinstance(code, dict):
                                country_code = code.get('@value', '')
                                description = code.get('Description', {})
                                if isinstance(description, dict):
                                    country_name = description.get('@value', '')
                                    
                                    # Extract region from annotations if available
                                    region = "Unknown"
                                    annotations = code.get('Annotations', {})
                                    if isinstance(annotations, dict):
                                        annotation = annotations.get('Annotation', {})
                                        if isinstance(annotation, dict):
                                            region = annotation.get('AnnotationTitle', {}).get('@value', 'Unknown')
                                    
                                    countries.append(IMFCountry(
                                        code=country_code,
                                        name=country_name,
                                        region=region
                                    ))
        except Exception as e:
            logger.error(f"Error parsing country structure: {str(e)}")
        
        return countries
    
    async def _load_indicator_mapping(self):
        """Load indicator code mapping."""
        try:
            # Load indicators from multiple datasets
            datasets = [IMFDataset.WEO, IMFDataset.IFS]
            
            for dataset in datasets:
                url = f"{self.base_url}/DataStructure/{dataset.value}"
                
                async with self.http_client.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        indicators = self._parse_indicators_from_structure(data, dataset)
                        for indicator in indicators:
                            self._indicator_mapping[indicator.code] = indicator
                        
                        logger.info(f"Loaded {len(indicators)} indicators from {dataset.value}")
                        
        except Exception as e:
            logger.error(f"Error loading indicator mapping: {str(e)}")
            self._load_fallback_indicator_mapping()
    
    def _load_fallback_indicator_mapping(self):
        """Load fallback indicator mapping."""
        # Common indicators with their IMF codes
        fallback_indicators = [
            IMFIndicatorInfo(
                code="NGDPD",
                name="Gross Domestic Product, Current Prices, U.S. Dollars",
                description="GDP at current prices in U.S. dollars",
                unit="U.S. dollars",
                scale="Units",
                dataset=IMFDataset.WEO,
                frequency="A"
            ),
            IMFIndicatorInfo(
                code="NGDP_RPCH",
                name="Gross Domestic Product, Real Growth, Percent Change",
                description="Real GDP growth rate",
                unit="Percent change",
                scale="Units",
                dataset=IMFDataset.WEO,
                frequency="A"
            ),
            IMFIndicatorInfo(
                code="NGDPDPC",
                name="Gross Domestic Product Per Capita, Current Prices, U.S. Dollars",
                description="GDP per capita at current prices in U.S. dollars",
                unit="U.S. dollars",
                scale="Units",
                dataset=IMFDataset.WEO,
                frequency="A"
            ),
            IMFIndicatorInfo(
                code="PCPI_PCH",
                name="Inflation, Average Consumer Prices, Percent Change",
                description="Inflation rate based on consumer price index",
                unit="Percent change",
                scale="Units",
                dataset=IMFDataset.WEO,
                frequency="A"
            ),
            IMFIndicatorInfo(
                code="LUR_PT",
                name="Unemployment Rate, Percent of Total Labor Force",
                description="Unemployment rate as percentage of labor force",
                unit="Percent",
                scale="Units",
                dataset=IMFDataset.WEO,
                frequency="A"
            ),
            IMFIndicatorInfo(
                code="GGXWDG_NGDP",
                name="General Government Gross Debt, Percent of GDP",
                description="General government gross debt as percentage of GDP",
                unit="Percent",
                scale="Units",
                dataset=IMFDataset.WEO,
                frequency="A"
            ),
            IMFIndicatorInfo(
                code="BCA_NGDPD",
                name="Current Account Balance, Percent of GDP",
                description="Current account balance as percentage of GDP",
                unit="Percent",
                scale="Units",
                dataset=IMFDataset.WEO,
                frequency="A"
            ),
        ]
        
        for indicator in fallback_indicators:
            self._indicator_mapping[indicator.code] = indicator
    
    def _parse_indicators_from_structure(self, data: Dict[str, Any], dataset: IMFDataset) -> List[IMFIndicatorInfo]:
        """Parse indicator information from data structure."""
        indicators = []
        
        try:
            if 'Structure' in data and 'CodeLists' in data['Structure']:
                code_lists = data['Structure']['CodeLists']['CodeList']
                for code_list in code_lists:
                    if isinstance(code_list, dict) and code_list.get('@id') == 'CL_INDICATOR':
                        codes = code_list.get('Code', [])
                        if not isinstance(codes, list):
                            codes = [codes]
                        
                        for code in codes:
                            if isinstance(code, dict):
                                indicator_code = code.get('@value', '')
                                description = code.get('Description', {})
                                if isinstance(description, dict):
                                    indicator_name = description.get('@value', '')
                                    
                                    # Try to extract unit and scale from annotations
                                    unit = "Not specified"
                                    scale = "Not specified"
                                    frequency = "A"  # Default to annual
                                    
                                    annotations = code.get('Annotations', {})
                                    if isinstance(annotations, dict):
                                        annotation_list = annotations.get('Annotation', [])
                                        if not isinstance(annotation_list, list):
                                            annotation_list = [annotation_list]
                                        
                                        for annotation in annotation_list:
                                            if isinstance(annotation, dict):
                                                title = annotation.get('AnnotationTitle', {}).get('@value', '')
                                                text = annotation.get('AnnotationText', {}).get('@value', '')
                                                
                                                if 'unit' in title.lower():
                                                    unit = text
                                                elif 'scale' in title.lower():
                                                    scale = text
                                                elif 'frequency' in title.lower():
                                                    frequency = text[0] if text else "A"
                                    
                                    indicators.append(IMFIndicatorInfo(
                                        code=indicator_code,
                                        name=indicator_name,
                                        description=indicator_name,  # Use name as description
                                        unit=unit,
                                        scale=scale,
                                        dataset=dataset,
                                        frequency=frequency
                                    ))
        except Exception as e:
            logger.error(f"Error parsing indicator structure: {str(e)}")
        
        return indicators
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def get_datasets(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Get list of available IMF datasets.
        
        Args:
            use_cache: Whether to use cache
            
        Returns:
            List of dataset information
        """
        cache_key = "datasets"
        
        if use_cache and cache_key in self.dataset_cache:
            return self.dataset_cache[cache_key]
        
        try:
            url = f"{self.base_url}/Dataflow"
            
            async with self.http_client.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    datasets = self._parse_datasets(data)
                    self.dataset_cache[cache_key] = datasets
                    return datasets
                else:
                    logger.error(f"Failed to get datasets: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting datasets: {str(e)}")
            return []
    
    def _parse_datasets(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse datasets from API response."""
        datasets = []
        
        try:
            if 'Structure' in data and 'Dataflows' in data['Structure']:
                dataflows = data['Structure']['Dataflows']['Dataflow']
                if not isinstance(dataflows, list):
                    dataflows = [dataflows]
                
                for dataflow in dataflows:
                    if isinstance(dataflow, dict):
                        dataset_id = dataflow.get('@id', '')
                        name = dataflow.get('Name', {}).get('@value', '')
                        description = dataflow.get('Description', {}).get('@value', '')
                        
                        # Map to our enum if possible
                        dataset_type = None
                        for ds in IMFDataset:
                            if ds.value in dataset_id:
                                dataset_type = ds
                                break
                        
                        datasets.append({
                            'id': dataset_id,
                            'name': name,
                            'description': description,
                            'type': dataset_type.value if dataset_type else 'Unknown',
                            'enum_type': dataset_type
                        })
        except Exception as e:
            logger.error(f"Error parsing datasets: {str(e)}")
        
        return datasets
    
    async def get_countries(
        self,
        region: Optional[str] = None,
        use_cache: bool = True
    ) -> List[IMFCountry]:
        """
        Get list of countries/areas.
        
        Args:
            region: Filter by region
            use_cache: Whether to use cache
            
        Returns:
            List of country information
        """
        # Ensure country mapping is loaded
        if not self._country_mapping:
            await self._load_country_mapping()
        
        countries = list(self._country_mapping.values())
        
        if region:
            countries = [c for c in countries if region.lower() in c.region.lower()]
        
        return countries
    
    async def get_country(self, country_code: str) -> Optional[IMFCountry]:
        """
        Get country information by code.
        
        Args:
            country_code: IMF country code
            
        Returns:
            Country information or None if not found
        """
        # Ensure country mapping is loaded
        if not self._country_mapping:
            await self._load_country_mapping()
        
        return self._country_mapping.get(country_code)
    
    async def get_indicators(
        self,
        dataset: Optional[IMFDataset] = None,
        use_cache: bool = True
    ) -> List[IMFIndicatorInfo]:
        """
        Get list of economic indicators.
        
        Args:
            dataset: Filter by dataset
            use_cache: Whether to use cache
            
        Returns:
            List of indicator information
        """
        # Ensure indicator mapping is loaded
        if not self._indicator_mapping:
            await self._load_indicator_mapping()
        
        indicators = list(self._indicator_mapping.values())
        
        if dataset:
            indicators = [i for i in indicators if i.dataset == dataset]
        
        return indicators
    
    async def get_indicator(self, indicator_code: str) -> Optional[IMFIndicatorInfo]:
        """
        Get indicator information by code.
        
        Args:
            indicator_code: IMF indicator code
            
        Returns:
            Indicator information or None if not found
        """
        # Ensure indicator mapping is loaded
        if not self._indicator_mapping:
            await self._load_indicator_mapping()
        
        return self._indicator_mapping.get(indicator_code)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def get_data(
        self,
        query: EconomicDataQuery,
        use_cache: bool = True
    ) -> List[IMFDataSeries]:
        """
        Get economic data based on query.
        
        Args:
            query: Data query parameters
            use_cache: Whether to use cache
            
        Returns:
            List of data series
        """
        cache_key = self._generate_cache_key(query)
        
        if use_cache and cache_key in self.data_cache:
            logger.debug(f"Cache hit for IMF data query: {cache_key}")
            return self.data_cache[cache_key]
        
        try:
            # Build API URL based on query
            url = self._build_data_url(query)
            
            async with self.http_client.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    series_list = self._parse_data_response(data, query)
                    
                    # Cache the results
                    if use_cache:
                        self.data_cache[cache_key] = series_list
                    
                    return series_list
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get IMF data: {response.status} - {error_text}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting IMF data: {str(e)}")
            return []
    
    def _build_data_url(self, query: EconomicDataQuery) -> str:
        """Build IMF API URL from query."""
        # IMF API format: /CompactData/{dataset}/{frequency}.{country_code}.{indicator_code}?startPeriod=YYYY&endPeriod=YYYY
        
        dataset = query.dataset or IMFDataset.WEO
        frequency = query.frequency or "A"  # Annual
        
        # Build country codes
        if query.country_codes:
            country_codes = "+".join(query.country_codes)
        elif query.region:
            # Get countries in region
            countries = [c for c in self._country_mapping.values() 
                        if query.region.lower() in c.region.lower()]
            country_codes = "+".join([c.code for c in countries[:10]])  # Limit to 10
        else:
            country_codes = "all"  # All countries
        
        # Build indicator codes
        if query.indicator_codes:
            indicator_codes = "+".join(query.indicator_codes)
        else:
            # Use default indicators for the dataset
            default_indicators = {
                IMFDataset.WEO: ["NGDPD", "NGDP_RPCH", "PCPI_PCH"],
                IMFDataset.IFS: ["FMB_PA", "FPOLM_PA", "ENDA_XDC_USD_RATE"],
                IMFDataset.GFS: ["GGXWDG_NGDP", "GGR_NGDP", "GGX_NGDP"],
            }
            indicator_codes = "+".join(default_indicators.get(dataset, ["NGDPD"]))
        
        # Build periods
        start_period = query.start_year or 2000
        end_period = query.end_year or datetime.now().year
        
        url = (
            f"{self.base_url}/CompactData/{dataset.value}/"
            f"{frequency}.{country_codes}.{indicator_codes}"
            f"?startPeriod={start_period}&endPeriod={end_period}"
        )
        
        logger.debug(f"Built IMF API URL: {url}")
        return url
    
    def _parse_data_response(self, data: Dict[str, Any], query: EconomicDataQuery) -> List[IMFDataSeries]:
        """Parse data from IMF API response."""
        series_list = []
        
        try:
            if 'CompactData' in data and 'DataSet' in data['CompactData']:
                dataset = data['CompactData']['DataSet']
                
                if 'Series' in dataset:
                    series_items = dataset['Series']
                    if not isinstance(series_items, list):
                        series_items = [series_items]
                    
                    for series_item in series_items:
                        if isinstance(series_item, dict):
                            series = self._parse_series(series_item, query)
                            if series:
                                series_list.append(series)
        
        except Exception as e:
            logger.error(f"Error parsing IMF data response: {str(e)}")
        
        return series_list
    
    def _parse_series(self, series_data: Dict[str, Any], query: EconomicDataQuery) -> Optional[IMFDataSeries]:
        """Parse a single data series."""
        try:
            # Extract metadata
            country_code = series_data.get('@REF_AREA', '')
            indicator_code = series_data.get('@INDICATOR', '')
            frequency = series_data.get('@FREQ', 'A')
            unit = series_data.get('@UNIT_MULT', '0')
            scale = series_data.get('@SCALE', '')
            
            # Get country and indicator info
            country = self._country_mapping.get(country_code)
            indicator = self._indicator_mapping.get(indicator_code)
            
            if not country:
                country = IMFCountry(
                    code=country_code,
                    name=f"Country {country_code}",
                    region="Unknown"
                )
            
            if not indicator:
                indicator = IMFIndicatorInfo(
                    code=indicator_code,
                    name=f"Indicator {indicator_code}",
                    description="Unknown indicator",
                    unit=unit,
                    scale=scale,
                    dataset=query.dataset or IMFDataset.WEO,
                    frequency=frequency
                )
            
            # Parse observations
            data_points = []
            observations = series_data.get('Obs', [])
            if not isinstance(observations, list):
                observations = [observations]
            
            for obs in observations:
                if isinstance(obs, dict):
                    time_period = obs.get('@TIME_PERIOD', '')
                    value = obs.get('@OBS_VALUE')
                    
                    # Parse year from time period
                    try:
                        if frequency == 'A':
                            year = int(time_period)
                            period = None
                        elif frequency == 'Q':
                            year = int(time_period[:4])
                            period = time_period[5:]  # Q1, Q2, Q3, Q4
                        elif frequency == 'M':
                            year = int(time_period[:4])
                            period = time_period[5:]  # MM
                        else:
                            year = int(time_period[:4])
                            period = None
                    except:
                        year = 0
                        period = None
                    
                    data_point = IMFDataPoint(
                        country_code=country_code,
                        country_name=country.name,
                        indicator_code=indicator_code,
                        indicator_name=indicator.name,
                        year=year,
                        period=period,
                        value=value,
                        unit=indicator.unit,
                        scale=indicator.scale,
                        dataset=indicator.dataset,
                        last_updated=datetime.now()
                    )
                    
                    data_points.append(data_point)
            
            # Filter by year range if specified
            if query.start_year or query.end_year:
                filtered_points = []
                for point in data_points:
                    if ((query.start_year is None or point.year >= query.start_year) and
                        (query.end_year is None or point.year <= query.end_year)):
                        filtered_points.append(point)
                data_points = filtered_points
            
            return IMFDataSeries(
                country_code=country_code,
                country_name=country.name,
                indicator_code=indicator_code,
                indicator_name=indicator.name,
                dataset=indicator.dataset,
                frequency=frequency,
                unit=indicator.unit,
                scale=indicator.scale,
                data_points=data_points,
                metadata={
                    'country_region': country.region,
                    'indicator_description': indicator.description,
                    'data_source': 'IMF'
                }
            )
            
        except Exception as e:
            logger.error(f"Error parsing series: {str(e)}")
            return None
    
    def _generate_cache_key(self, query: EconomicDataQuery) -> str:
        """Generate cache key for query."""
        import hashlib
        
        key_data = {
            "dataset": query.dataset.value if query.dataset else "WEO",
            "country_codes": sorted(query.country_codes) if query.country_codes else [],
            "indicator_codes": sorted(query.indicator_codes) if query.indicator_codes else [],
            "region": query.region,
            "start_year": query.start_year,
            "end_year": query.end_year,
            "frequency": query.frequency,
        }
        
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    # High-level convenience methods
    
    async def get_gdp_data(
        self,
        country_codes: List[str],
        start_year: int = 2000,
        end_year: Optional[int] = None
    ) -> List[IMFDataSeries]:
        """
        Get GDP data for countries.
        
        Args:
            country_codes: List of country codes
            start_year: Start year
            end_year: End year (defaults to current year)
            
        Returns:
            List of GDP data series
        """
        if end_year is None:
            end_year = datetime.now().year
        
        query = EconomicDataQuery(
            dataset=IMFDataset.WEO,
            country_codes=country_codes,
            indicator_codes=["NGDPD", "NGDP_RPCH", "NGDPDPC"],
            start_year=start_year,
            end_year=end_year
        )
        
        return await self.get_data(query)
    
    async def get_inflation_data(
        self,
        country_codes: List[str],
        start_year: int = 2000,
        end_year: Optional[int] = None
    ) -> List[IMFDataSeries]:
        """
        Get inflation data for countries.
        
        Args:
            country_codes: List of country codes
            start_year: Start year
            end_year: End year (defaults to current year)
            
        Returns:
            List of inflation data series
        """
        if end_year is None:
            end_year = datetime.now().year
        
        query = EconomicDataQuery(
            dataset=IMFDataset.WEO,
            country_codes=country_codes,
            indicator_codes=["PCPI_PCH"],
            start_year=start_year,
            end_year=end_year
        )
        
        return await self.get_data(query)
    
    async def get_fiscal_data(
        self,
        country_codes: List[str],
        start_year: int = 2000,
        end_year: Optional[int] = None
    ) -> List[IMFDataSeries]:
        """
        Get fiscal data for countries.
        
        Args:
            country_codes: List of country codes
            start_year: Start year
            end_year: End year (defaults to current year)
            
        Returns:
            List of fiscal data series
        """
        if end_year is None:
            end_year = datetime.now().year
        
        query = EconomicDataQuery(
            dataset=IMFDataset.WEO,
            country_codes=country_codes,
            indicator_codes=["GGXWDG_NGDP", "GGR_NGDP", "GGX_NGDP", "GGXCNL_NGDP"],
            start_year=start_year,
            end_year=end_year
        )
        
        return await self.get_data(query)
    
    async def get_trade_data(
        self,
        country_codes: List[str],
        start_year: int = 2000,
        end_year: Optional[int] = None
    ) -> List[IMFDataSeries]:
        """
        Get trade data for countries.
        
        Args:
            country_codes: List of country codes
            start_year: Start year
            end_year: End year (defaults to current year)
            
        Returns:
            List of trade data series
        """
        if end_year is None:
            end_year = datetime.now().year
        
        query = EconomicDataQuery(
            dataset=IMFDataset.WEO,
            country_codes=country_codes,
            indicator_codes=["BCA_NGDPD", "TXG_BP6_USD", "TMG_BP6_USD"],
            start_year=start_year,
            end_year=end_year
        )
        
        return await self.get_data(query)
    
    async def get_country_economic_profile(
        self,
        country_code: str
    ) -> Dict[str, Any]:
        """
        Get comprehensive economic profile for a country.
        
        Args:
            country_code: IMF country code
            
        Returns:
            Economic profile dictionary
        """
        # Get multiple data series
        gdp_series = await self.get_gdp_data([country_code], start_year=datetime.now().year - 10)
        inflation_series = await self.get_inflation_data([country_code], start_year=datetime.now().year - 10)
        fiscal_series = await self.get_fiscal_data([country_code], start_year=datetime.now().year - 10)
        trade_series = await self.get_trade_data([country_code], start_year=datetime.now().year - 10)
        
        # Get latest values
        profile = {
            "country_code": country_code,
            "profile_date": datetime.now().isoformat(),
            "indicators": {}
        }
        
        # Helper function to get latest value
        def get_latest_value(series_list, indicator_code):
            for series in series_list:
                if series.indicator_code == indicator_code:
                    latest = series.get_latest_value()
                    if latest and latest.value:
                        return {
                            "value": float(latest.value),
                            "year": latest.year,
                            "unit": latest.unit
                        }
            return None
        
        # GDP indicators
        profile["indicators"]["gdp_current_usd"] = get_latest_value(gdp_series, "NGDPD")
        profile["indicators"]["gdp_growth"] = get_latest_value(gdp_series, "NGDP_RPCH")
        profile["indicators"]["gdp_per_capita"] = get_latest_value(gdp_series, "NGDPDPC")
        
        # Inflation
        profile["indicators"]["inflation"] = get_latest_value(inflation_series, "PCPI_PCH")
        
        # Fiscal indicators
        profile["indicators"]["gov_debt_pct_gdp"] = get_latest_value(fiscal_series, "GGXWDG_NGDP")
        profile["indicators"]["gov_revenue_pct_gdp"] = get_latest_value(fiscal_series, "GGR_NGDP")
        profile["indicators"]["gov_balance_pct_gdp"] = get_latest_value(fiscal_series, "GGXCNL_NGDP")
        
        # Trade indicators
        profile["indicators"]["current_account_pct_gdp"] = get_latest_value(trade_series, "BCA_NGDPD")
        
        return profile
    
    async def compare_countries(
        self,
        country_codes: List[str],
        indicator_code: str,
        start_year: int = 2000,
        end_year: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compare countries on a specific indicator.
        
        Args:
            country_codes: List of country codes to compare
            indicator_code: Indicator code to compare
            start_year: Start year
            end_year: End year
            
        Returns:
            Comparison data
        """
        if end_year is None:
            end_year = datetime.now().year
        
        query = EconomicDataQuery(
            country_codes=country_codes,
            indicator_codes=[indicator_code],
            start_year=start_year,
            end_year=end_year
        )
        
        series_list = await self.get_data(query)
        
        # Get indicator info
        indicator = await self.get_indicator(indicator_code)
        
        # Format comparison data
        comparison = {
            "indicator_code": indicator_code,
            "indicator_name": indicator.name if indicator else indicator_code,
            "unit": indicator.unit if indicator else "Unknown",
            "start_year": start_year,
            "end_year": end_year,
            "countries": {},
            "data_points": []
        }
        
        # Organize data by country
        for series in series_list:
            country_code = series.country_code
            country_name = series.country_name
            
            comparison["countries"][country_code] = {
                "name": country_name,
                "data": []
            }
            
            for point in series.data_points:
                comparison["countries"][country_code]["data"].append({
                    "year": point.year,
                    "value": float(point.value) if point.value else None
                })
                
                # Also store in flat structure for charting
                comparison["data_points"].append({
                    "country_code": country_code,
                    "country_name": country_name,
                    "year": point.year,
                    "value": float(point.value) if point.value else None
                })
        
        return comparison
    
    async def search_indicators(
        self,
        search_term: str,
        dataset: Optional[IMFDataset] = None,
        limit: int = 20
    ) -> List[IMFIndicatorInfo]:
        """
        Search for indicators by name or description.
        
        Args:
            search_term: Search term
            dataset: Filter by dataset
            limit: Maximum results
            
        Returns:
            List of matching indicators
        """
        # Ensure indicator mapping is loaded
        if not self._indicator_mapping:
            await self._load_indicator_mapping()
        
        indicators = list(self._indicator_mapping.values())
        
        # Filter by dataset if specified
        if dataset:
            indicators = [i for i in indicators if i.dataset == dataset]
        
        # Search in name and description
        search_term_lower = search_term.lower()
        matching = []
        
        for indicator in indicators:
            if (search_term_lower in indicator.name.lower() or
                search_term_lower in indicator.description.lower()):
                matching.append(indicator)
                
                if len(matching) >= limit:
                    break
        
        return matching
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on IMF API.
        
        Returns:
            Health status information
        """
        try:
            # Test with a simple dataset request
            start_time = datetime.now()
            
            async with self.http_client.session.get(
                f"{self.base_url}/Dataflow",
                timeout=30
            ) as response:
                end_time = datetime.now()
                
                latency = (end_time - start_time).total_seconds() * 1000
                
                if response.status == 200:
                    return {
                        "status": "healthy",
                        "message": "IMF API is responding",
                        "latency_ms": round(latency, 2),
                        "timestamp": datetime.now().isoformat(),
                        "cached_countries": len(self._country_mapping),
                        "cached_indicators": len(self._indicator_mapping)
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "message": f"API returned status {response.status}",
                        "timestamp": datetime.now().isoformat()
                    }
                    
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }


# Factory function for dependency injection
def get_imf_client() -> IMFClient:
    """
    Factory function to create IMF client.
    
    Returns:
        Configured IMFClient instance
    """
    return IMFClient()