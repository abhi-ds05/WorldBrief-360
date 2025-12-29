# backend/app/integrations/openweather_client.py
"""
OpenWeather API integration for WorldBrief 360.
Provides weather data, forecasts, air quality, and historical weather information.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import math

import aiohttp
from pydantic import BaseModel, Field, validator, HttpUrl
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from cachetools import TTLCache, LRUCache

from app.core.config import settings
from app.core.logging_config import logger
from app.services.utils.http_client import AsyncHTTPClient
from app.schemas.request.weather import WeatherQueryParams


class WeatherUnits(Enum):
    """Temperature units."""
    STANDARD = "standard"  # Kelvin
    METRIC = "metric"      # Celsius
    IMPERIAL = "imperial"  # Fahrenheit


class Language(Enum):
    """Weather description languages."""
    AR = "ar"  # Arabic
    BG = "bg"  # Bulgarian
    CA = "ca"  # Catalan
    CZ = "cz"  # Czech
    DE = "de"  # German
    EL = "el"  # Greek
    EN = "en"  # English
    FA = "fa"  # Persian (Farsi)
    FI = "fi"  # Finnish
    FR = "fr"  # French
    GL = "gl"  # Galician
    HE = "he"  # Hebrew
    HI = "hi"  # Hindi
    HR = "hr"  # Croatian
    HU = "hu"  # Hungarian
    ID = "id"  # Indonesian
    IT = "it"  # Italian
    JA = "ja"  # Japanese
    KR = "kr"  # Korean
    LA = "la"  # Latvian
    LT = "lt"  # Lithuanian
    MK = "mk"  # Macedonian
    NL = "nl"  # Dutch
    PL = "pl"  # Polish
    PT = "pt"  # Portuguese
    RO = "ro"  # Romanian
    RU = "ru"  # Russian
    SE = "se"  # Swedish
    SK = "sk"  # Slovak
    SL = "sl"  # Slovenian
    SP = "sp"  # Spanish
    SR = "sr"  # Serbian
    TH = "th"  # Thai
    TR = "tr"  # Turkish
    UA = "ua"  # Ukrainian
    VI = "vi"  # Vietnamese
    ZH_CN = "zh_cn"  # Chinese Simplified
    ZH_TW = "zh_tw"  # Chinese Traditional


class WeatherCondition(Enum):
    """Weather conditions based on OpenWeather codes."""
    THUNDERSTORM = "Thunderstorm"
    DRIZZLE = "Drizzle"
    RAIN = "Rain"
    SNOW = "Snow"
    ATMOSPHERE = "Atmosphere"
    CLEAR = "Clear"
    CLOUDS = "Clouds"
    EXTREME = "Extreme"
    ADDITIONAL = "Additional"


@dataclass
class Coordinates:
    """Geographic coordinates."""
    lat: float
    lon: float
    
    def to_dict(self) -> Dict[str, float]:
        return {"lat": self.lat, "lon": self.lon}
    
    def distance_to(self, other: 'Coordinates') -> float:
        """Calculate distance in kilometers using Haversine formula."""
        R = 6371  # Earth's radius in kilometers
        
        lat1 = math.radians(self.lat)
        lon1 = math.radians(self.lon)
        lat2 = math.radians(other.lat)
        lon2 = math.radians(other.lon)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c


class WeatherAlert(BaseModel):
    """Weather alert."""
    sender_name: str
    event: str
    start: datetime
    end: datetime
    description: str
    tags: List[str] = Field(default_factory=list)
    
    @validator('start', 'end', pre=True)
    def parse_datetime(cls, v):
        if isinstance(v, int):
            return datetime.fromtimestamp(v, tz=timezone.utc)
        return v


class WeatherConditionDetail(BaseModel):
    """Detailed weather condition."""
    id: int
    main: str
    description: str
    icon: str
    
    def get_icon_url(self, size: str = "2x") -> str:
        """Get icon URL from OpenWeather."""
        return f"https://openweathermap.org/img/wn/{self.icon}@{size}.png"
    
    def get_category(self) -> WeatherCondition:
        """Get weather category."""
        category_map = {
            "Thunderstorm": WeatherCondition.THUNDERSTORM,
            "Drizzle": WeatherCondition.DRIZZLE,
            "Rain": WeatherCondition.RAIN,
            "Snow": WeatherCondition.SNOW,
            "Mist": WeatherCondition.ATMOSPHERE,
            "Smoke": WeatherCondition.ATMOSPHERE,
            "Haze": WeatherCondition.ATMOSPHERE,
            "Dust": WeatherCondition.ATMOSPHERE,
            "Fog": WeatherCondition.ATMOSPHERE,
            "Sand": WeatherCondition.ATMOSPHERE,
            "Ash": WeatherCondition.ATMOSPHERE,
            "Squall": WeatherCondition.ATMOSPHERE,
            "Tornado": WeatherCondition.ATMOSPHERE,
            "Clear": WeatherCondition.CLEAR,
            "Clouds": WeatherCondition.CLOUDS,
            "Extreme": WeatherCondition.EXTREME,
        }
        return category_map.get(self.main, WeatherCondition.ADDITIONAL)


class Temperature(BaseModel):
    """Temperature measurements."""
    temp: float
    feels_like: float
    temp_min: Optional[float] = None
    temp_max: Optional[float] = None
    pressure: Optional[int] = None  # hPa
    humidity: Optional[int] = None  # %
    sea_level: Optional[int] = None  # hPa (at sea level)
    grnd_level: Optional[int] = None  # hPa (at ground level)
    
    def to_celsius(self, unit: WeatherUnits = WeatherUnits.METRIC) -> float:
        """Convert temperature to Celsius."""
        if unit == WeatherUnits.STANDARD:
            return self.temp - 273.15
        elif unit == WeatherUnits.IMPERIAL:
            return (self.temp - 32) * 5/9
        return self.temp
    
    def to_fahrenheit(self, unit: WeatherUnits = WeatherUnits.METRIC) -> float:
        """Convert temperature to Fahrenheit."""
        if unit == WeatherUnits.STANDARD:
            return (self.temp - 273.15) * 9/5 + 32
        elif unit == WeatherUnits.METRIC:
            return self.temp * 9/5 + 32
        return self.temp
    
    def to_kelvin(self, unit: WeatherUnits = WeatherUnits.METRIC) -> float:
        """Convert temperature to Kelvin."""
        if unit == WeatherUnits.METRIC:
            return self.temp + 273.15
        elif unit == WeatherUnits.IMPERIAL:
            return (self.temp - 32) * 5/9 + 273.15
        return self.temp


class Wind(BaseModel):
    """Wind measurements."""
    speed: float  # m/s
    deg: int      # degrees
    gust: Optional[float] = None  # m/s
    
    def get_direction(self) -> str:
        """Get wind direction as cardinal direction."""
        directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
                     'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        idx = round(self.deg / 22.5) % 16
        return directions[idx]
    
    def get_beaufort_scale(self) -> int:
        """Get Beaufort scale number."""
        speeds = [0.3, 1.6, 3.4, 5.5, 8.0, 10.8, 13.9, 17.2, 20.8,
                 24.5, 28.5, 32.7, 36.9, 41.5, 46.2, 51.0, 56.1, 61.3]
        
        for i, speed in enumerate(speeds):
            if self.speed < speed:
                return i
        return 12  # Hurricane


class Precipitation(BaseModel):
    """Precipitation measurements."""
    last_1h: Optional[float] = None  # mm in last hour
    last_3h: Optional[float] = None  # mm in last 3 hours
    snow_last_1h: Optional[float] = None  # mm snow in last hour
    snow_last_3h: Optional[float] = None  # mm snow in last 3 hours
    probability: Optional[float] = None  # % probability of precipitation
    
    def get_total(self) -> float:
        """Get total precipitation."""
        total = 0.0
        if self.last_1h:
            total += self.last_1h
        if self.last_3h:
            total += self.last_3h
        return total


class CloudCover(BaseModel):
    """Cloud cover information."""
    all: int  # percentage
    description: Optional[str] = None
    
    def get_description(self) -> str:
        """Get cloud cover description."""
        if self.all <= 10:
            return "Clear sky"
        elif self.all <= 30:
            return "Few clouds"
        elif self.all <= 60:
            return "Scattered clouds"
        elif self.all <= 85:
            return "Broken clouds"
        else:
            return "Overcast clouds"


class SunTimes(BaseModel):
    """Sunrise and sunset times."""
    sunrise: datetime
    sunset: datetime
    timezone: int  # Shift in seconds from UTC
    
    @validator('sunrise', 'sunset', pre=True)
    def parse_datetime(cls, v):
        if isinstance(v, int):
            return datetime.fromtimestamp(v, tz=timezone.utc)
        return v
    
    def get_daylight_hours(self) -> float:
        """Get daylight hours."""
        delta = self.sunset - self.sunrise
        return delta.total_seconds() / 3600
    
    def is_daytime(self, current_time: Optional[datetime] = None) -> bool:
        """Check if it's currently daytime."""
        if current_time is None:
            current_time = datetime.now(timezone.utc)
        
        # Adjust times to UTC
        sunrise_utc = self.sunrise.astimezone(timezone.utc)
        sunset_utc = self.sunset.astimezone(timezone.utc)
        
        return sunrise_utc <= current_time <= sunset_utc


class AirQualityIndex(BaseModel):
    """Air Quality Index (AQI)."""
    aqi: int  # 1-5 scale
    components: Dict[str, float]  # Pollutant concentrations
    datetime: Optional[datetime] = None # type: ignore
    
    @validator('aqi')
    def validate_aqi(cls, v):
        if not 1 <= v <= 5:
            raise ValueError('AQI must be between 1 and 5')
        return v
    
    def get_description(self) -> str:
        """Get AQI description."""
        descriptions = {
            1: "Good",
            2: "Fair",
            3: "Moderate",
            4: "Poor",
            5: "Very Poor"
        }
        return descriptions.get(self.aqi, "Unknown")
    
    def get_health_implications(self) -> str:
        """Get health implications."""
        implications = {
            1: "Air quality is satisfactory, and air pollution poses little or no risk.",
            2: "Air quality is acceptable. However, there may be a risk for some people.",
            3: "Members of sensitive groups may experience health effects.",
            4: "Health alert: The risk of health effects is increased for everyone.",
            5: "Health warning of emergency conditions."
        }
        return implications.get(self.aqi, "Unknown")


class WeatherForecast(BaseModel):
    """Weather forecast for a specific time."""
    datetime: datetime
    temperature: Temperature
    conditions: List[WeatherConditionDetail] = Field(default_factory=list)
    wind: Wind
    clouds: CloudCover
    precipitation: Optional[Precipitation] = None
    visibility: Optional[int] = None  # meters
    pop: Optional[float] = None  # Probability of precipitation (%)
    humidity: Optional[int] = None  # %
    pressure: Optional[int] = None  # hPa
    uv_index: Optional[float] = None  # UV index
    
    @validator('datetime', pre=True)
    def parse_datetime(cls, v):
        if isinstance(v, int):
            return datetime.fromtimestamp(v, tz=timezone.utc)
        return v


class CurrentWeather(BaseModel):
    """Current weather data."""
    coord: Coordinates
    weather: List[WeatherConditionDetail] = Field(default_factory=list)
    base: str = "stations"
    main: Temperature
    visibility: Optional[int] = None  # meters
    wind: Wind
    clouds: CloudCover
    rain: Optional[Dict[str, float]] = None
    snow: Optional[Dict[str, float]] = None
    dt: datetime  # Time of data calculation
    sys: Dict[str, Any] = Field(default_factory=dict)
    timezone: int  # Shift in seconds from UTC
    id: Optional[int] = None  # City ID
    name: Optional[str] = None  # City name
    cod: Optional[int] = None  # Internal parameter
    
    @validator('dt', pre=True)
    def parse_dt(cls, v):
        if isinstance(v, int):
            return datetime.fromtimestamp(v, tz=timezone.utc)
        return v
    
    def get_sun_times(self) -> Optional[SunTimes]:
        """Get sunrise and sunset times from sys data."""
        sys_data = self.sys
        if 'sunrise' in sys_data and 'sunset' in sys_data:
            return SunTimes(
                sunrise=sys_data['sunrise'],
                sunset=sys_data['sunset'],
                timezone=self.timezone
            )
        return None
    
    def get_precipitation(self) -> Precipitation:
        """Get precipitation data."""
        rain_1h = self.rain.get('1h') if self.rain else None
        rain_3h = self.rain.get('3h') if self.rain else None
        snow_1h = self.snow.get('1h') if self.snow else None
        snow_3h = self.snow.get('3h') if self.snow else None
        
        return Precipitation(
            last_1h=rain_1h,
            last_3h=rain_3h,
            snow_last_1h=snow_1h,
            snow_last_3h=snow_3h
        )


class WeatherForecastResponse(BaseModel):
    """Weather forecast response."""
    lat: float
    lon: float
    timezone: str
    timezone_offset: int
    current: Optional[CurrentWeather] = None
    minutely: List[WeatherForecast] = Field(default_factory=list)  # Next 60 minutes
    hourly: List[WeatherForecast] = Field(default_factory=list)    # Next 48 hours
    daily: List[WeatherForecast] = Field(default_factory=list)     # Next 7 days
    alerts: List[WeatherAlert] = Field(default_factory=list)


class HistoricalWeather(BaseModel):
    """Historical weather data."""
    lat: float
    lon: float
    timezone: str
    timezone_offset: int
    data: List[WeatherForecast] = Field(default_factory=list)


class AirQualityResponse(BaseModel):
    """Air quality response."""
    coord: Coordinates
    list: List[AirQualityIndex] = Field(default_factory=list)


class OpenWeatherClient:
    """
    Client for OpenWeather API.
    Provides current weather, forecasts, historical data, and air quality information.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openweathermap.org/data/2.5",
        timeout: int = 30,
        max_retries: int = 3,
        cache_ttl: int = 600  # 10 minutes cache
    ):
        """
        Initialize OpenWeather client.
        
        Args:
            api_key: OpenWeather API key
            base_url: API base URL
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            cache_ttl: Cache TTL in seconds
        """
        self.api_key = api_key or settings.OPENWEATHER_API_KEY
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        
        self.http_client = AsyncHTTPClient(
            timeout=timeout,
            retries=max_retries
        )
        
        # Caches
        self.current_cache = TTLCache(maxsize=1000, ttl=cache_ttl)
        self.forecast_cache = TTLCache(maxsize=500, ttl=cache_ttl * 3)  # 30 minutes
        self.air_quality_cache = TTLCache(maxsize=500, ttl=cache_ttl * 6)  # 1 hour
        self.geocoding_cache = TTLCache(maxsize=10000, ttl=86400)  # 24 hours
        
        # Rate limiting tracking
        self.calls_per_minute = 60  # Free tier limit
        self.calls_made = 0
        self.last_reset = datetime.now()
        
        if not self.api_key:
            logger.warning("OpenWeather API key not provided")
    
    def is_available(self) -> bool:
        """Check if client is properly configured."""
        return bool(self.api_key)
    
    def _update_rate_limit(self):
        """Update rate limiting tracking."""
        current_time = datetime.now()
        if (current_time - self.last_reset).total_seconds() >= 60:
            self.calls_made = 0
            self.last_reset = current_time
        
        self.calls_made += 1
        
        if self.calls_made >= self.calls_per_minute:
            wait_time = 60 - (current_time - self.last_reset).total_seconds()
            if wait_time > 0:
                logger.warning(f"Rate limit reached. Waiting {wait_time:.1f} seconds.")
                asyncio.sleep(wait_time)
                self.calls_made = 0
                self.last_reset = datetime.now()
    
    def _build_params(
        self,
        base_params: Dict[str, Any],
        units: Optional[WeatherUnits] = None,
        lang: Optional[Language] = None
    ) -> Dict[str, Any]:
        """Build common parameters for OpenWeather API."""
        params = base_params.copy()
        params["appid"] = self.api_key
        
        if units:
            params["units"] = units.value
        
        if lang:
            params["lang"] = lang.value
        
        return params
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError,))
    )
    async def geocode(
        self,
        query: str,
        limit: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Geocode location name to coordinates.
        
        Args:
            query: Location name (city, country, coordinates)
            limit: Maximum number of results
            **kwargs: Additional parameters
            
        Returns:
            List of geocoding results
        """
        if not self.is_available():
            raise ValueError("OpenWeather client is not configured")
        
        cache_key = f"geocode:{query}:{limit}"
        
        if cache_key in self.geocoding_cache:
            logger.debug(f"Cache hit for geocode: {query}")
            return self.geocoding_cache[cache_key]
        
        try:
            self._update_rate_limit()
            
            params = {
                "q": query,
                "limit": limit,
            }
            params.update(kwargs)
            
            url = "http://api.openweathermap.org/geo/1.0/direct"
            
            async with self.http_client.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if isinstance(data, list):
                        results = []
                        for item in data:
                            results.append({
                                "name": item.get("name", ""),
                                "local_names": item.get("local_names", {}),
                                "lat": item.get("lat", 0),
                                "lon": item.get("lon", 0),
                                "country": item.get("country", ""),
                                "state": item.get("state", ""),
                            })
                        
                        self.geocoding_cache[cache_key] = results
                        return results
                    else:
                        return []
                else:
                    error_text = await response.text()
                    logger.error(f"Geocoding failed: {response.status} - {error_text}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error in geocoding: {str(e)}")
            return []
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def reverse_geocode(
        self,
        lat: float,
        lon: float,
        limit: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Reverse geocode coordinates to location name.
        
        Args:
            lat: Latitude
            lon: Longitude
            limit: Maximum number of results
            **kwargs: Additional parameters
            
        Returns:
            List of reverse geocoding results
        """
        if not self.is_available():
            raise ValueError("OpenWeather client is not configured")
        
        cache_key = f"reverse_geocode:{lat}:{lon}:{limit}"
        
        if cache_key in self.geocoding_cache:
            logger.debug(f"Cache hit for reverse geocode: ({lat}, {lon})")
            return self.geocoding_cache[cache_key]
        
        try:
            self._update_rate_limit()
            
            params = {
                "lat": lat,
                "lon": lon,
                "limit": limit,
            }
            params.update(kwargs)
            
            url = "http://api.openweathermap.org/geo/1.0/reverse"
            
            async with self.http_client.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if isinstance(data, list):
                        results = []
                        for item in data:
                            results.append({
                                "name": item.get("name", ""),
                                "local_names": item.get("local_names", {}),
                                "lat": item.get("lat", 0),
                                "lon": item.get("lon", 0),
                                "country": item.get("country", ""),
                                "state": item.get("state", ""),
                            })
                        
                        self.geocoding_cache[cache_key] = results
                        return results
                    else:
                        return []
                else:
                    error_text = await response.text()
                    logger.error(f"Reverse geocoding failed: {response.status} - {error_text}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error in reverse geocoding: {str(e)}")
            return []
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def get_current_weather(
        self,
        location: Union[str, Coordinates],
        units: WeatherUnits = WeatherUnits.METRIC,
        lang: Language = Language.EN,
        **kwargs
    ) -> Optional[CurrentWeather]:
        """
        Get current weather for a location.
        
        Args:
            location: City name or coordinates
            units: Temperature units
            lang: Language for descriptions
            **kwargs: Additional parameters
            
        Returns:
            Current weather data or None if not found
        """
        if not self.is_available():
            raise ValueError("OpenWeather client is not configured")
        
        # Generate cache key
        if isinstance(location, Coordinates):
            cache_key = f"current:{location.lat}:{location.lon}:{units.value}:{lang.value}"
        else:
            cache_key = f"current:{location}:{units.value}:{lang.value}"
        
        if cache_key in self.current_cache:
            logger.debug(f"Cache hit for current weather: {cache_key}")
            return self.current_cache[cache_key]
        
        try:
            self._update_rate_limit()
            
            # Build parameters
            params = self._build_params({}, units, lang)
            
            if isinstance(location, Coordinates):
                params["lat"] = location.lat
                params["lon"] = location.lon
            else:
                params["q"] = location
            
            params.update(kwargs)
            
            url = f"{self.base_url}/weather"
            
            async with self.http_client.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Parse current weather
                    current_weather = self._parse_current_weather(data)
                    
                    # Cache the result
                    self.current_cache[cache_key] = current_weather
                    
                    return current_weather
                elif response.status == 404:
                    logger.warning(f"Location not found: {location}")
                    return None
                else:
                    error_text = await response.text()
                    logger.error(f"Current weather failed: {response.status} - {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting current weather: {str(e)}")
            return None
    
    def _parse_current_weather(self, data: Dict[str, Any]) -> CurrentWeather:
        """Parse current weather data from OpenWeather API."""
        # Parse coordinates
        coord_data = data.get("coord", {})
        coord = Coordinates(
            lat=coord_data.get("lat", 0),
            lon=coord_data.get("lon", 0)
        )
        
        # Parse weather conditions
        weather_conditions = []
        for condition_data in data.get("weather", []):
            weather_conditions.append(WeatherConditionDetail(**condition_data))
        
        # Parse main temperature data
        main_data = data.get("main", {})
        temperature = Temperature(
            temp=main_data.get("temp", 0),
            feels_like=main_data.get("feels_like", 0),
            temp_min=main_data.get("temp_min"),
            temp_max=main_data.get("temp_max"),
            pressure=main_data.get("pressure"),
            humidity=main_data.get("humidity"),
            sea_level=main_data.get("sea_level"),
            grnd_level=main_data.get("grnd_level")
        )
        
        # Parse wind data
        wind_data = data.get("wind", {})
        wind = Wind(
            speed=wind_data.get("speed", 0),
            deg=wind_data.get("deg", 0),
            gust=wind_data.get("gust")
        )
        
        # Parse cloud data
        clouds_data = data.get("clouds", {})
        clouds = CloudCover(all=clouds_data.get("all", 0))
        
        return CurrentWeather(
            coord=coord,
            weather=weather_conditions,
            base=data.get("base", "stations"),
            main=temperature,
            visibility=data.get("visibility"),
            wind=wind,
            clouds=clouds,
            rain=data.get("rain"),
            snow=data.get("snow"),
            dt=data.get("dt", 0),
            sys=data.get("sys", {}),
            timezone=data.get("timezone", 0),
            id=data.get("id"),
            name=data.get("name"),
            cod=data.get("cod")
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def get_weather_forecast(
        self,
        location: Union[str, Coordinates],
        units: WeatherUnits = WeatherUnits.METRIC,
        lang: Language = Language.EN,
        exclude: Optional[List[str]] = None,
        **kwargs
    ) -> Optional[WeatherForecastResponse]:
        """
        Get weather forecast for a location.
        
        Args:
            location: City name or coordinates
            units: Temperature units
            lang: Language for descriptions
            exclude: Parts to exclude (current, minutely, hourly, daily, alerts)
            **kwargs: Additional parameters
            
        Returns:
            Weather forecast response or None if not found
        """
        if not self.is_available():
            raise ValueError("OpenWeather client is not configured")
        
        # Generate cache key
        if isinstance(location, Coordinates):
            cache_key = f"forecast:{location.lat}:{location.lon}:{units.value}:{lang.value}"
        else:
            cache_key = f"forecast:{location}:{units.value}:{lang.value}"
        
        if exclude:
            cache_key += f":{','.join(sorted(exclude))}"
        
        if cache_key in self.forecast_cache:
            logger.debug(f"Cache hit for forecast: {cache_key}")
            return self.forecast_cache[cache_key]
        
        try:
            self._update_rate_limit()
            
            # Build parameters
            params = self._build_params({}, units, lang)
            
            if isinstance(location, Coordinates):
                params["lat"] = location.lat
                params["lon"] = location.lon
            else:
                params["q"] = location
            
            if exclude:
                params["exclude"] = ",".join(exclude)
            
            params.update(kwargs)
            
            url = f"{self.base_url}/onecall"
            
            async with self.http_client.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Parse forecast response
                    forecast_response = self._parse_forecast_response(data)
                    
                    # Cache the result
                    self.forecast_cache[cache_key] = forecast_response
                    
                    return forecast_response
                elif response.status == 404:
                    logger.warning(f"Location not found for forecast: {location}")
                    return None
                else:
                    error_text = await response.text()
                    logger.error(f"Weather forecast failed: {response.status} - {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting weather forecast: {str(e)}")
            return None
    
    def _parse_forecast_response(self, data: Dict[str, Any]) -> WeatherForecastResponse:
        """Parse forecast response from OpenWeather API."""
        # Parse current weather if present
        current = None
        if "current" in data:
            # Convert to CurrentWeather format
            current_data = data["current"]
            current_data["coord"] = {"lat": data["lat"], "lon": data["lon"]}
            current_data["sys"] = {}
            current_data["timezone"] = data["timezone_offset"]
            current = self._parse_current_weather(current_data)
        
        # Parse minutely forecasts
        minutely_forecasts = []
        for forecast_data in data.get("minutely", []):
            minutely_forecasts.append(self._parse_weather_forecast(forecast_data))
        
        # Parse hourly forecasts
        hourly_forecasts = []
        for forecast_data in data.get("hourly", []):
            hourly_forecasts.append(self._parse_weather_forecast(forecast_data))
        
        # Parse daily forecasts
        daily_forecasts = []
        for forecast_data in data.get("daily", []):
            daily_forecasts.append(self._parse_weather_forecast(forecast_data))
        
        # Parse alerts
        alerts = []
        for alert_data in data.get("alerts", []):
            alerts.append(WeatherAlert(**alert_data))
        
        return WeatherForecastResponse(
            lat=data.get("lat", 0),
            lon=data.get("lon", 0),
            timezone=data.get("timezone", ""),
            timezone_offset=data.get("timezone_offset", 0),
            current=current,
            minutely=minutely_forecasts,
            hourly=hourly_forecasts,
            daily=daily_forecasts,
            alerts=alerts
        )
    
    def _parse_weather_forecast(self, data: Dict[str, Any]) -> WeatherForecast:
        """Parse weather forecast data."""
        # Parse datetime
        dt = data.get("dt", 0)
        
        # Parse temperature
        temp_data = data.get("temp", {})
        if isinstance(temp_data, dict):
            temperature = Temperature(
                temp=temp_data.get("day") if "day" in temp_data else temp_data.get("temp", 0),
                feels_like=temp_data.get("feels_like", {}).get("day") if "feels_like" in temp_data else data.get("feels_like", 0),
                temp_min=temp_data.get("min"),
                temp_max=temp_data.get("max"),
                pressure=data.get("pressure"),
                humidity=data.get("humidity")
            )
        else:
            temperature = Temperature(
                temp=temp_data,
                feels_like=data.get("feels_like", {}).get("temp") if isinstance(data.get("feels_like"), dict) else data.get("feels_like", 0),
                pressure=data.get("pressure"),
                humidity=data.get("humidity")
            )
        
        # Parse weather conditions
        conditions = []
        for condition_data in data.get("weather", []):
            conditions.append(WeatherConditionDetail(**condition_data))
        
        # Parse wind
        wind_data = data.get("wind", {})
        wind = Wind(
            speed=wind_data.get("speed", 0) if isinstance(wind_data, dict) else wind_data,
            deg=wind_data.get("deg", 0) if isinstance(wind_data, dict) else data.get("wind_deg", 0),
            gust=wind_data.get("gust") if isinstance(wind_data, dict) else data.get("wind_gust")
        )
        
        # Parse clouds
        clouds_data = data.get("clouds", {})
        clouds = CloudCover(
            all=clouds_data if isinstance(clouds_data, (int, float)) else clouds_data.get("all", 0)
        )
        
        # Parse precipitation
        precipitation = None
        if "rain" in data or "snow" in data:
            rain_1h = data.get("rain", {}).get("1h") if isinstance(data.get("rain"), dict) else data.get("rain")
            rain_3h = data.get("rain", {}).get("3h") if isinstance(data.get("rain"), dict) else None
            snow_1h = data.get("snow", {}).get("1h") if isinstance(data.get("snow"), dict) else data.get("snow")
            snow_3h = data.get("snow", {}).get("3h") if isinstance(data.get("snow"), dict) else None
            
            precipitation = Precipitation(
                last_1h=rain_1h,
                last_3h=rain_3h,
                snow_last_1h=snow_1h,
                snow_last_3h=snow_3h,
                probability=data.get("pop")  # Probability of precipitation
            )
        
        return WeatherForecast(
            datetime=dt,
            temperature=temperature,
            conditions=conditions,
            wind=wind,
            clouds=clouds,
            precipitation=precipitation,
            visibility=data.get("visibility"),
            pop=data.get("pop"),
            humidity=data.get("humidity"),
            pressure=data.get("pressure"),
            uv_index=data.get("uvi")
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def get_air_quality(
        self,
        location: Union[str, Coordinates],
        **kwargs
    ) -> Optional[AirQualityResponse]:
        """
        Get air quality data for a location.
        
        Args:
            location: City name or coordinates
            **kwargs: Additional parameters
            
        Returns:
            Air quality response or None if not found
        """
        if not self.is_available():
            raise ValueError("OpenWeather client is not configured")
        
        # Generate cache key
        if isinstance(location, Coordinates):
            cache_key = f"air_quality:{location.lat}:{location.lon}"
        else:
            cache_key = f"air_quality:{location}"
        
        if cache_key in self.air_quality_cache:
            logger.debug(f"Cache hit for air quality: {cache_key}")
            return self.air_quality_cache[cache_key]
        
        try:
            self._update_rate_limit()
            
            # Build parameters
            params = {"appid": self.api_key}
            
            if isinstance(location, Coordinates):
                params["lat"] = location.lat
                params["lon"] = location.lon
            else:
                # Geocode first to get coordinates
                geocode_results = await self.geocode(location, limit=1)
                if not geocode_results:
                    logger.warning(f"Location not found for air quality: {location}")
                    return None
                
                params["lat"] = geocode_results[0]["lat"]
                params["lon"] = geocode_results[0]["lon"]
            
            params.update(kwargs)
            
            url = f"{self.base_url}/air_pollution"
            
            async with self.http_client.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Parse air quality response
                    air_quality_response = self._parse_air_quality_response(data)
                    
                    # Cache the result
                    self.air_quality_cache[cache_key] = air_quality_response
                    
                    return air_quality_response
                else:
                    error_text = await response.text()
                    logger.error(f"Air quality failed: {response.status} - {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting air quality: {str(e)}")
            return None
    
    def _parse_air_quality_response(self, data: Dict[str, Any]) -> AirQualityResponse:
        """Parse air quality response from OpenWeather API."""
        # Parse coordinates
        coord_data = data.get("coord", {})
        coord = Coordinates(
            lat=coord_data.get("lat", 0),
            lon=coord_data.get("lon", 0)
        )
        
        # Parse air quality data
        aqi_list = []
        for aqi_data in data.get("list", []):
            main_data = aqi_data.get("main", {})
            components_data = aqi_data.get("components", {})
            
            aqi = AirQualityIndex(
                aqi=main_data.get("aqi", 1),
                components=components_data,
                datetime=aqi_data.get("dt")
            )
            aqi_list.append(aqi)
        
        return AirQualityResponse(
            coord=coord,
            list=aqi_list
        )
    
    async def get_historical_weather(
        self,
        location: Union[str, Coordinates],
        date: datetime,
        units: WeatherUnits = WeatherUnits.METRIC,
        lang: Language = Language.EN,
        **kwargs
    ) -> Optional[HistoricalWeather]:
        """
        Get historical weather data for a location and date.
        
        Args:
            location: City name or coordinates
            date: Date for historical data
            units: Temperature units
            lang: Language for descriptions
            **kwargs: Additional parameters
            
        Returns:
            Historical weather data or None if not found
        """
        if not self.is_available():
            raise ValueError("OpenWeather client is not configured")
        
        # Note: Historical data requires paid subscription
        # This is a placeholder implementation
        
        try:
            # Convert date to timestamp
            timestamp = int(date.timestamp())
            
            # Build parameters
            params = self._build_params({}, units, lang)
            
            if isinstance(location, Coordinates):
                params["lat"] = location.lat
                params["lon"] = location.lon
            else:
                params["q"] = location
            
            params["dt"] = timestamp
            params.update(kwargs)
            
            url = f"{self.base_url}/onecall/timemachine"
            
            async with self.http_client.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Parse historical data
                    forecasts = []
                    for forecast_data in data.get("hourly", []):
                        forecasts.append(self._parse_weather_forecast(forecast_data))
                    
                    return HistoricalWeather(
                        lat=data.get("lat", 0),
                        lon=data.get("lon", 0),
                        timezone=data.get("timezone", ""),
                        timezone_offset=data.get("timezone_offset", 0),
                        data=forecasts
                    )
                else:
                    error_text = await response.text()
                    logger.error(f"Historical weather failed: {response.status} - {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting historical weather: {str(e)}")
            return None
    
    async def get_weather_summary(
        self,
        location: Union[str, Coordinates],
        units: WeatherUnits = WeatherUnits.METRIC,
        lang: Language = Language.EN,
        include_forecast: bool = True,
        include_air_quality: bool = False
    ) -> Dict[str, Any]:
        """
        Get comprehensive weather summary for a location.
        
        Args:
            location: City name or coordinates
            units: Temperature units
            lang: Language for descriptions
            include_forecast: Whether to include forecast
            include_air_quality: Whether to include air quality
            
        Returns:
            Weather summary dictionary
        """
        summary = {
            "location": str(location),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "units": units.value,
            "language": lang.value,
        }
        
        # Get current weather
        current_weather = await self.get_current_weather(
            location=location,
            units=units,
            lang=lang
        )
        
        if current_weather:
            summary["current"] = self._format_current_summary(current_weather, units)
            
            # Get forecast if requested
            if include_forecast:
                forecast = await self.get_weather_forecast(
                    location=location,
                    units=units,
                    lang=lang,
                    exclude=["minutely", "alerts"]
                )
                
                if forecast:
                    summary["forecast"] = self._format_forecast_summary(forecast)
            
            # Get air quality if requested
            if include_air_quality:
                air_quality = await self.get_air_quality(location)
                
                if air_quality and air_quality.list:
                    summary["air_quality"] = self._format_air_quality_summary(air_quality.list[0])
        
        return summary
    
    def _format_current_summary(
        self,
        current: CurrentWeather,
        units: WeatherUnits
    ) -> Dict[str, Any]:
        """Format current weather for summary."""
        main_condition = current.weather[0] if current.weather else None
        
        return {
            "temperature": {
                "current": current.main.temp,
                "feels_like": current.main.feels_like,
                "min": current.main.temp_min,
                "max": current.main.temp_max,
                "unit": "°C" if units == WeatherUnits.METRIC else "°F" if units == WeatherUnits.IMPERIAL else "K"
            },
            "condition": {
                "main": main_condition.main if main_condition else "Unknown",
                "description": main_condition.description if main_condition else "",
                "icon": main_condition.get_icon_url() if main_condition else ""
            },
            "wind": {
                "speed": current.wind.speed,
                "direction": current.wind.get_direction(),
                "gust": current.wind.gust
            },
            "humidity": current.main.humidity,
            "pressure": current.main.pressure,
            "visibility": current.visibility,
            "clouds": current.clouds.all,
            "sun_times": self._format_sun_times(current.get_sun_times()),
            "location": {
                "name": current.name,
                "coordinates": current.coord.to_dict(),
                "timezone": current.timezone
            }
        }
    
    def _format_forecast_summary(self, forecast: WeatherForecastResponse) -> Dict[str, Any]:
        """Format forecast for summary."""
        hourly_summary = []
        for hour in forecast.hourly[:12]:  # Next 12 hours
            hourly_summary.append({
                "time": hour.datetime.isoformat(),
                "temp": hour.temperature.temp,
                "condition": hour.conditions[0].main if hour.conditions else "Unknown",
                "precipitation_probability": hour.pop,
            })
        
        daily_summary = []
        for day in forecast.daily[:5]:  # Next 5 days
            daily_summary.append({
                "date": day.datetime.date().isoformat(),
                "temp_min": day.temperature.temp_min,
                "temp_max": day.temperature.temp_max,
                "condition": day.conditions[0].main if day.conditions else "Unknown",
                "precipitation_probability": day.pop,
            })
        
        return {
            "hourly": hourly_summary,
            "daily": daily_summary,
            "alerts": [
                {
                    "event": alert.event,
                    "description": alert.description,
                    "start": alert.start.isoformat(),
                    "end": alert.end.isoformat()
                }
                for alert in forecast.alerts
            ]
        }
    
    def _format_air_quality_summary(self, aqi: AirQualityIndex) -> Dict[str, Any]:
        """Format air quality for summary."""
        return {
            "aqi": aqi.aqi,
            "description": aqi.get_description(),
            "health_implications": aqi.get_health_implications(),
            "pollutants": aqi.components,
            "timestamp": aqi.datetime.isoformat() if aqi.datetime else None
        }
    
    def _format_sun_times(self, sun_times: Optional[SunTimes]) -> Optional[Dict[str, str]]:
        """Format sun times for summary."""
        if not sun_times:
            return None
        
        return {
            "sunrise": sun_times.sunrise.isoformat(),
            "sunset": sun_times.sunset.isoformat(),
            "daylight_hours": round(sun_times.get_daylight_hours(), 1),
            "is_daytime": sun_times.is_daytime()
        }
    
    async def compare_locations(
        self,
        locations: List[Union[str, Coordinates]],
        units: WeatherUnits = WeatherUnits.METRIC,
        lang: Language = Language.EN
    ) -> Dict[str, Any]:
        """
        Compare weather between multiple locations.
        
        Args:
            locations: List of locations to compare
            units: Temperature units
            lang: Language for descriptions
            
        Returns:
            Comparison results
        """
        comparison = {
            "locations": [],
            "comparison_time": datetime.now(timezone.utc).isoformat(),
            "units": units.value,
        }
        
        weather_data = []
        
        # Get weather for each location
        for location in locations:
            weather = await self.get_current_weather(
                location=location,
                units=units,
                lang=lang
            )
            
            if weather:
                location_info = {
                    "name": weather.name or str(location),
                    "coordinates": weather.coord.to_dict(),
                    "weather": {
                        "temp": weather.main.temp,
                        "feels_like": weather.main.feels_like,
                        "condition": weather.weather[0].main if weather.weather else "Unknown",
                        "humidity": weather.main.humidity,
                        "wind_speed": weather.wind.speed,
                        "wind_direction": weather.wind.get_direction(),
                    }
                }
                weather_data.append(location_info)
        
        comparison["locations"] = weather_data
        
        # Find extremes
        if weather_data:
            temps = [loc["weather"]["temp"] for loc in weather_data]
            humidities = [loc["weather"]["humidity"] for loc in weather_data]
            wind_speeds = [loc["weather"]["wind_speed"] for loc in weather_data]
            
            comparison["extremes"] = {
                "hottest": {
                    "location": weather_data[temps.index(max(temps))]["name"],
                    "temperature": max(temps)
                },
                "coldest": {
                    "location": weather_data[temps.index(min(temps))]["name"],
                    "temperature": min(temps)
                },
                "most_humid": {
                    "location": weather_data[humidities.index(max(humidities))]["name"],
                    "humidity": max(humidities)
                },
                "windiest": {
                    "location": weather_data[wind_speeds.index(max(wind_speeds))]["name"],
                    "wind_speed": max(wind_speeds)
                }
            }
        
        return comparison
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on OpenWeather API.
        
        Returns:
            Health status information
        """
        if not self.is_available():
            return {
                "status": "disabled",
                "message": "Client not configured (no API key)",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Test with a simple current weather request
            start_time = datetime.now()
            
            params = {
                "appid": self.api_key,
                "q": "London",
                "units": "metric",
            }
            
            url = f"{self.base_url}/weather"
            
            async with self.http_client.session.get(
                url, params=params, timeout=10
            ) as response:
                end_time = datetime.now()
                
                latency = (end_time - start_time).total_seconds() * 1000
                
                if response.status == 200:
                    return {
                        "status": "healthy",
                        "message": "API is responding",
                        "latency_ms": round(latency, 2),
                        "calls_made_last_minute": self.calls_made,
                        "timestamp": datetime.now().isoformat()
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
def get_openweather_client() -> OpenWeatherClient:
    """
    Factory function to create OpenWeather client.
    
    Returns:
        Configured OpenWeatherClient instance
    """
    return OpenWeatherClient()