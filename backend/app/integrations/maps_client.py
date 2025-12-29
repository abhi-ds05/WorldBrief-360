# backend/app/integrations/maps_client.py
"""
Maps and geospatial services integration for WorldBrief 360.
Supports multiple providers: Google Maps, Mapbox, OpenStreetMap, and HERE Maps.
"""

import asyncio
import json
import math
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import hashlib
from urllib.parse import quote, urlencode

import aiohttp
import geopy.distance
from pydantic import BaseModel, Field, validator, HttpUrl, confloat, conint
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from cachetools import TTLCache, LRUCache
import polyline

from app.core.config import settings
from app.core.logging_config import logger
from app.services.utils.http_client import AsyncHTTPClient
from app.schemas.request.geo import (
    GeocodeRequest, ReverseGeocodeRequest, 
    DirectionsRequest, PlaceSearchRequest,
    DistanceMatrixRequest
)


class MapProvider(Enum):
    """Supported map providers."""
    GOOGLE_MAPS = "google"
    MAPBOX = "mapbox"
    OPENSTREETMAP = "osm"
    HERE_MAPS = "here"
    OPENROUTESERVICE = "ors"


class TravelMode(Enum):
    """Travel modes for directions and distance calculations."""
    DRIVING = "driving"
    WALKING = "walking"
    BICYCLING = "bicycling"
    TRANSIT = "transit"
    FLYING = "flying"  # For air distance


class AvoidType(Enum):
    """Types of features to avoid in routing."""
    TOLLS = "tolls"
    HIGHWAYS = "highways"
    FERRIES = "ferries"
    INDOOR = "indoor"


class PlaceType(Enum):
    """Place types for search and filtering."""
    ACCOUNTING = "accounting"
    AIRPORT = "airport"
    AMUSEMENT_PARK = "amusement_park"
    AQUARIUM = "aquarium"
    ART_GALLERY = "art_gallery"
    ATM = "atm"
    BAKERY = "bakery"
    BANK = "bank"
    BAR = "bar"
    BEAUTY_SALON = "beauty_salon"
    BICYCLE_STORE = "bicycle_store"
    BOOK_STORE = "book_store"
    BOWLING_ALLEY = "bowling_alley"
    BUS_STATION = "bus_station"
    CAFE = "cafe"
    CAMPGROUND = "campground"
    CAR_DEALER = "car_dealer"
    CAR_RENTAL = "car_rental"
    CAR_REPAIR = "car_repair"
    CAR_WASH = "car_wash"
    CASINO = "casino"
    CEMETERY = "cemetery"
    CHURCH = "church"
    CITY_HALL = "city_hall"
    CLOTHING_STORE = "clothing_store"
    CONVENIENCE_STORE = "convenience_store"
    COURTHOUSE = "courthouse"
    DENTIST = "dentist"
    DEPARTMENT_STORE = "department_store"
    DOCTOR = "doctor"
    DRUGSTORE = "drugstore"
    ELECTRICIAN = "electrician"
    ELECTRONICS_STORE = "electronics_store"
    EMBASSY = "embassy"
    FIRE_STATION = "fire_station"
    FLORIST = "florist"
    FUNERAL_HOME = "funeral_home"
    FURNITURE_STORE = "furniture_store"
    GAS_STATION = "gas_station"
    GYM = "gym"
    HAIR_CARE = "hair_care"
    HARDWARE_STORE = "hardware_store"
    HINDU_TEMPLE = "hindu_temple"
    HOME_GOODS_STORE = "home_goods_store"
    HOSPITAL = "hospital"
    INSURANCE_AGENCY = "insurance_agency"
    JEWELRY_STORE = "jewelry_store"
    LAUNDRY = "laundry"
    LAWYER = "lawyer"
    LIBRARY = "library"
    LIGHT_RAIL_STATION = "light_rail_station"
    LIQUOR_STORE = "liquor_store"
    LOCAL_GOVERNMENT_OFFICE = "local_government_office"
    LOCKSMITH = "locksmith"
    LODGING = "lodging"
    MEAL_DELIVERY = "meal_delivery"
    MEAL_TAKEAWAY = "meal_takeaway"
    MOSQUE = "mosque"
    MOVIE_RENTAL = "movie_rental"
    MOVIE_THEATER = "movie_theater"
    MOVING_COMPANY = "moving_company"
    MUSEUM = "museum"
    NIGHT_CLUB = "night_club"
    PAINTER = "painter"
    PARK = "park"
    PARKING = "parking"
    PET_STORE = "pet_store"
    PHARMACY = "pharmacy"
    PHYSIOTHERAPIST = "physiotherapist"
    PLUMBER = "plumber"
    POLICE = "police"
    POST_OFFICE = "post_office"
    PRIMARY_SCHOOL = "primary_school"
    REAL_ESTATE_AGENCY = "real_estate_agency"
    RESTAURANT = "restaurant"
    ROOFING_CONTRACTOR = "roofing_contractor"
    RV_PARK = "rv_park"
    SCHOOL = "school"
    SECONDARY_SCHOOL = "secondary_school"
    SHOE_STORE = "shoe_store"
    SHOPPING_MALL = "shopping_mall"
    SPA = "spa"
    STADIUM = "stadium"
    STORAGE = "storage"
    STORE = "store"
    SUBWAY_STATION = "subway_station"
    SUPERMARKET = "supermarket"
    SYNAGOGUE = "synagogue"
    TAXI_STAND = "taxi_stand"
    TOURIST_ATTRACTION = "tourist_attraction"
    TRAIN_STATION = "train_station"
    TRANSIT_STATION = "transit_station"
    TRAVEL_AGENCY = "travel_agency"
    UNIVERSITY = "university"
    VETERINARY_CARE = "veterinary_care"
    ZOO = "zoo"


@dataclass
class Coordinates:
    """Geographic coordinates."""
    lat: float
    lng: float
    
    def to_dict(self) -> Dict[str, float]:
        return {"lat": self.lat, "lng": self.lng}
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.lat, self.lng)
    
    def distance_to(self, other: 'Coordinates') -> float:
        """Calculate distance in kilometers using Haversine formula."""
        return geopy.distance.distance(
            (self.lat, self.lng),
            (other.lat, other.lng)
        ).km
    
    def bearing_to(self, other: 'Coordinates') -> float:
        """Calculate initial bearing in degrees."""
        lat1 = math.radians(self.lat)
        lat2 = math.radians(other.lat)
        diff_lng = math.radians(other.lng - self.lng)
        
        x = math.sin(diff_lng) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (
            math.sin(lat1) * math.cos(lat2) * math.cos(diff_lng)
        )
        
        initial_bearing = math.atan2(x, y)
        return (math.degrees(initial_bearing) + 360) % 360


@dataclass
class BoundingBox:
    """Geographic bounding box."""
    north: float
    south: float
    east: float
    west: float
    
    def contains(self, coords: Coordinates) -> bool:
        """Check if coordinates are within bounding box."""
        return (
            self.south <= coords.lat <= self.north and
            self.west <= coords.lng <= self.east
        )
    
    def center(self) -> Coordinates:
        """Get center coordinates of bounding box."""
        center_lat = (self.north + self.south) / 2
        center_lng = (self.east + self.west) / 2
        return Coordinates(center_lat, center_lng)
    
    def area(self) -> float:
        """Calculate area in square kilometers (approximate)."""
        # Simple approximation for small areas
        lat_distance = geopy.distance.distance(
            (self.north, self.west),
            (self.south, self.west)
        ).km
        
        lng_distance = geopy.distance.distance(
            (self.north, self.west),
            (self.north, self.east)
        ).km
        
        return lat_distance * lng_distance


class AddressComponent(BaseModel):
    """Component of an address."""
    long_name: str
    short_name: str
    types: List[str]
    
    def get_type_value(self, type_name: str) -> Optional[str]:
        """Get value for a specific type."""
        if type_name in self.types:
            return self.long_name
        return None


class Geometry(BaseModel):
    """Geometric information for a location."""
    location: Coordinates
    location_type: str
    viewport: Optional[BoundingBox] = None
    bounds: Optional[BoundingBox] = None
    
    class Config:
        arbitrary_types_allowed = True


class Place(BaseModel):
    """A place/location from maps API."""
    place_id: str
    name: str
    formatted_address: str
    geometry: Geometry
    types: List[str] = Field(default_factory=list)
    rating: Optional[float] = None
    user_ratings_total: Optional[int] = None
    price_level: Optional[int] = None  # 0-4
    opening_hours: Optional[Dict[str, Any]] = None
    photos: List[Dict[str, Any]] = Field(default_factory=list)
    permanently_closed: Optional[bool] = None
    business_status: Optional[str] = None
    icon: Optional[str] = None
    icon_background_color: Optional[str] = None
    icon_mask_base_uri: Optional[str] = None
    plus_code: Optional[Dict[str, str]] = None
    scope: Optional[str] = None
    vicinity: Optional[str] = None
    website: Optional[str] = None
    phone_number: Optional[str] = None
    international_phone_number: Optional[str] = None
    reviews: List[Dict[str, Any]] = Field(default_factory=list)
    utc_offset: Optional[int] = None  # minutes
    adr_address: Optional[str] = None
    html_attributions: List[str] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True
    
    def is_open_now(self) -> Optional[bool]:
        """Check if place is open now (if opening hours available)."""
        if not self.opening_hours:
            return None
        
        open_now = self.opening_hours.get('open_now')
        return open_now if open_now is not None else None


class GeocodeResult(BaseModel):
    """Result from geocoding (address to coordinates)."""
    formatted_address: str
    geometry: Geometry
    place_id: str
    types: List[str] = Field(default_factory=list)
    address_components: List[AddressComponent] = Field(default_factory=list)
    partial_match: bool = False
    postcode_localities: Optional[List[str]] = None
    
    def get_component(self, type_name: str) -> Optional[AddressComponent]:
        """Get address component by type."""
        for component in self.address_components:
            if type_name in component.types:
                return component
        return None
    
    def get_country(self) -> Optional[str]:
        """Get country name."""
        country_component = self.get_component("country")
        return country_component.long_name if country_component else None
    
    def get_city(self) -> Optional[str]:
        """Get city name."""
        for type_name in ["locality", "administrative_area_level_2", "postal_town"]:
            component = self.get_component(type_name)
            if component:
                return component.long_name
        return None


class RouteStep(BaseModel):
    """A step in a route."""
    distance: Dict[str, Any]  # {"text": "0.5 km", "value": 500}
    duration: Dict[str, Any]  # {"text": "5 mins", "value": 300}
    start_location: Coordinates
    end_location: Coordinates
    html_instructions: Optional[str] = None
    maneuver: Optional[str] = None
    polyline: Optional[str] = None  # Encoded polyline
    travel_mode: TravelMode
    steps: List['RouteStep'] = Field(default_factory=list)  # For complex steps
    
    class Config:
        arbitrary_types_allowed = True
    
    def get_coordinates(self) -> List[Coordinates]:
        """Decode polyline to get coordinates."""
        if not self.polyline:
            return [self.start_location, self.end_location]
        
        try:
            # polyline library expects (lat, lng) tuples
            points = polyline.decode(self.polyline)
            return [Coordinates(lat=lat, lng=lng) for lat, lng in points]
        except:
            return [self.start_location, self.end_location]


class RouteLeg(BaseModel):
    """A leg of a route (between waypoints)."""
    distance: Dict[str, Any]
    duration: Dict[str, Any]
    start_address: str
    end_address: str
    start_location: Coordinates
    end_location: Coordinates
    steps: List[RouteStep] = Field(default_factory=list)
    via_waypoints: List[Coordinates] = Field(default_factory=list)
    traffic_speed_entry: List[Any] = Field(default_factory=list)
    via_waypoint: List[Any] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True
    
    def total_distance_km(self) -> float:
        """Get total distance in kilometers."""
        distance_m = self.distance.get("value", 0)
        return distance_m / 1000
    
    def total_duration_minutes(self) -> float:
        """Get total duration in minutes."""
        duration_s = self.duration.get("value", 0)
        return duration_s / 60


class Route(BaseModel):
    """A complete route."""
    summary: str
    legs: List[RouteLeg] = Field(default_factory=list)
    waypoint_order: List[int] = Field(default_factory=list)
    overview_polyline: Optional[str] = None
    bounds: Optional[BoundingBox] = None
    copyrights: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    fare: Optional[Dict[str, Any]] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def total_distance_km(self) -> float:
        """Get total route distance in kilometers."""
        return sum(leg.total_distance_km() for leg in self.legs)
    
    def total_duration_minutes(self) -> float:
        """Get total route duration in minutes."""
        return sum(leg.total_duration_minutes() for leg in self.legs)
    
    def get_coordinates(self) -> List[Coordinates]:
        """Get all coordinates along the route."""
        coords = []
        for leg in self.legs:
            for step in leg.steps:
                coords.extend(step.get_coordinates())
        return coords


class DistanceMatrixElement(BaseModel):
    """Element in a distance matrix."""
    status: str
    distance: Optional[Dict[str, Any]] = None
    duration: Optional[Dict[str, Any]] = None
    duration_in_traffic: Optional[Dict[str, Any]] = None
    fare: Optional[Dict[str, Any]] = None


class DistanceMatrixRow(BaseModel):
    """Row in a distance matrix."""
    elements: List[DistanceMatrixElement]


class DistanceMatrixResponse(BaseModel):
    """Distance matrix response."""
    origin_addresses: List[str]
    destination_addresses: List[str]
    rows: List[DistanceMatrixRow]
    
    def get_distance(
        self, 
        origin_index: int, 
        destination_index: int
    ) -> Optional[Dict[str, Any]]:
        """Get distance between origin and destination."""
        if origin_index < len(self.rows) and destination_index < len(self.rows[origin_index].elements):
            element = self.rows[origin_index].elements[destination_index]
            return element.distance
        return None
    
    def get_duration(
        self, 
        origin_index: int, 
        destination_index: int
    ) -> Optional[Dict[str, Any]]:
        """Get duration between origin and destination."""
        if origin_index < len(self.rows) and destination_index < len(self.rows[origin_index].elements):
            element = self.rows[origin_index].elements[destination_index]
            return element.duration
        return None


class BaseMapsProvider:
    """Base class for maps providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
        self.http_client = AsyncHTTPClient()
        
    async def geocode(self, request: GeocodeRequest) -> List[GeocodeResult]:
        """Convert address to coordinates."""
        raise NotImplementedError("Subclasses must implement geocode")
    
    async def reverse_geocode(self, request: ReverseGeocodeRequest) -> List[GeocodeResult]:
        """Convert coordinates to address."""
        raise NotImplementedError("Subclasses must implement reverse_geocode")
    
    async def get_directions(self, request: DirectionsRequest) -> List[Route]:
        """Get directions between locations."""
        raise NotImplementedError("Subclasses must implement get_directions")
    
    async def get_distance_matrix(self, request: DistanceMatrixRequest) -> DistanceMatrixResponse:
        """Get distance matrix between multiple origins and destinations."""
        raise NotImplementedError("Subclasses must implement get_distance_matrix")
    
    async def search_places(self, request: PlaceSearchRequest) -> List[Place]:
        """Search for places."""
        raise NotImplementedError("Subclasses must implement search_places")
    
    async def get_place_details(self, place_id: str) -> Optional[Place]:
        """Get detailed information about a place."""
        raise NotImplementedError("Subclasses must implement get_place_details")
    
    async def autocomplete(self, query: str, session_token: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get place autocomplete suggestions."""
        raise NotImplementedError("Subclasses must implement autocomplete")
    
    async def validate_connection(self) -> bool:
        """Validate connection to maps provider."""
        raise NotImplementedError("Subclasses must implement validate_connection")


class GoogleMapsProvider(BaseMapsProvider):
    """Google Maps provider."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {
                "api_key": settings.GOOGLE_MAPS_API_KEY,
                "base_url": "https://maps.googleapis.com/maps/api",
                "geocode_url": "/geocode/json",
                "directions_url": "/directions/json",
                "distance_matrix_url": "/distancematrix/json",
                "places_url": "/place",
                "default_language": "en",
                "region": "us",
            }
        super().__init__(config)
        
        self.base_url = self.config["base_url"]
        self.api_key = self.config["api_key"]
        
        if not self.api_key:
            logger.warning("Google Maps API key not provided")
    
    def is_available(self) -> bool:
        """Check if provider is properly configured."""
        return bool(self.api_key)
    
    def _build_params(self, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Build common parameters for Google Maps API."""
        params = base_params.copy()
        params["key"] = self.api_key
        
        if "language" not in params:
            params["language"] = self.config.get("default_language", "en")
        
        if "region" not in params and "region" in self.config:
            params["region"] = self.config["region"]
        
        return params
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError,))
    )
    async def geocode(self, request: GeocodeRequest) -> List[GeocodeResult]:
        """Geocode address using Google Maps."""
        if not self.is_available():
            raise ValueError("Google Maps provider is not configured")
        
        try:
            params = self._build_params({
                "address": request.address,
                "bounds": request.bounds,
                "components": self._format_components(request.components),
                "language": request.language,
            })
            
            url = f"{self.base_url}/geocode/json"
            
            async with self.http_client.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get("status") == "OK":
                        results = data.get("results", [])
                        return [self._parse_geocode_result(r) for r in results]
                    else:
                        logger.error(f"Geocoding failed: {data.get('status')} - {data.get('error_message')}")
                        return []
                else:
                    logger.error(f"Geocoding failed with status: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error in geocoding: {str(e)}")
            return []
    
    def _format_components(self, components: Optional[Dict[str, str]]) -> Optional[str]:
        """Format components dictionary for Google Maps API."""
        if not components:
            return None
        
        component_parts = []
        for key, value in components.items():
            if value:
                component_parts.append(f"{key}:{value}")
        
        return "|".join(component_parts) if component_parts else None
    
    def _parse_geocode_result(self, data: Dict[str, Any]) -> GeocodeResult:
        """Parse Google Maps geocode result."""
        # Parse address components
        address_components = []
        for comp_data in data.get("address_components", []):
            address_components.append(AddressComponent(
                long_name=comp_data.get("long_name", ""),
                short_name=comp_data.get("short_name", ""),
                types=comp_data.get("types", [])
            ))
        
        # Parse geometry
        geometry_data = data.get("geometry", {})
        location_data = geometry_data.get("location", {})
        location = Coordinates(
            lat=location_data.get("lat", 0),
            lng=location_data.get("lng", 0)
        )
        
        # Parse viewport
        viewport = None
        viewport_data = geometry_data.get("viewport")
        if viewport_data:
            viewport = BoundingBox(
                north=viewport_data.get("northeast", {}).get("lat", 0),
                south=viewport_data.get("southwest", {}).get("lat", 0),
                east=viewport_data.get("northeast", {}).get("lng", 0),
                west=viewport_data.get("southwest", {}).get("lng", 0)
            )
        
        # Parse bounds
        bounds = None
        bounds_data = geometry_data.get("bounds")
        if bounds_data:
            bounds = BoundingBox(
                north=bounds_data.get("northeast", {}).get("lat", 0),
                south=bounds_data.get("southwest", {}).get("lat", 0),
                east=bounds_data.get("northeast", {}).get("lng", 0),
                west=bounds_data.get("southwest", {}).get("lng", 0)
            )
        
        geometry = Geometry(
            location=location,
            location_type=geometry_data.get("location_type", ""),
            viewport=viewport,
            bounds=bounds
        )
        
        return GeocodeResult(
            formatted_address=data.get("formatted_address", ""),
            geometry=geometry,
            place_id=data.get("place_id", ""),
            types=data.get("types", []),
            address_components=address_components,
            partial_match=data.get("partial_match", False),
            postcode_localities=data.get("postcode_localities")
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def reverse_geocode(self, request: ReverseGeocodeRequest) -> List[GeocodeResult]:
        """Reverse geocode coordinates using Google Maps."""
        if not self.is_available():
            raise ValueError("Google Maps provider is not configured")
        
        try:
            latlng = f"{request.location.lat},{request.location.lng}"
            
            params = self._build_params({
                "latlng": latlng,
                "result_type": "|".join(request.result_types) if request.result_types else None,
                "location_type": "|".join(request.location_types) if request.location_types else None,
                "language": request.language,
            })
            
            url = f"{self.base_url}/geocode/json"
            
            async with self.http_client.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get("status") == "OK":
                        results = data.get("results", [])
                        return [self._parse_geocode_result(r) for r in results]
                    else:
                        logger.error(f"Reverse geocoding failed: {data.get('status')}")
                        return []
                else:
                    logger.error(f"Reverse geocoding failed with status: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error in reverse geocoding: {str(e)}")
            return []
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def get_directions(self, request: DirectionsRequest) -> List[Route]:
        """Get directions using Google Maps."""
        if not self.is_available():
            raise ValueError("Google Maps provider is not configured")
        
        try:
            # Build origin and destination
            origin = self._format_location(request.origin)
            destination = self._format_location(request.destination)
            
            # Build parameters
            params = self._build_params({
                "origin": origin,
                "destination": destination,
                "mode": request.travel_mode.value if request.travel_mode else "driving",
                "waypoints": self._format_waypoints(request.waypoints) if request.waypoints else None,
                "alternatives": request.alternatives,
                "avoid": "|".join([a.value for a in request.avoid]) if request.avoid else None,
                "language": request.language,
                "units": request.units or "metric",
                "departure_time": request.departure_time,
                "arrival_time": request.arrival_time,
                "traffic_model": request.traffic_model,
                "transit_mode": "|".join(request.transit_modes) if request.transit_modes else None,
                "transit_routing_preference": request.transit_routing_preference,
            })
            
            url = f"{self.base_url}/directions/json"
            
            async with self.http_client.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get("status") == "OK":
                        routes_data = data.get("routes", [])
                        return [self._parse_route(r) for r in routes_data]
                    else:
                        logger.error(f"Directions failed: {data.get('status')}")
                        return []
                else:
                    logger.error(f"Directions failed with status: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting directions: {str(e)}")
            return []
    
    def _format_location(self, location: Union[str, Coordinates, Place]) -> str:
        """Format location for Google Maps API."""
        if isinstance(location, str):
            return location
        elif isinstance(location, Coordinates):
            return f"{location.lat},{location.lng}"
        elif isinstance(location, Place):
            return location.place_id
        else:
            return str(location)
    
    def _format_waypoints(self, waypoints: List[Union[str, Coordinates, Place]]) -> str:
        """Format waypoints for Google Maps API."""
        formatted_waypoints = []
        for waypoint in waypoints:
            if isinstance(waypoint, Coordinates):
                formatted_waypoints.append(f"{waypoint.lat},{waypoint.lng}")
            elif isinstance(waypoint, Place):
                formatted_waypoints.append(f"place_id:{waypoint.place_id}")
            else:
                formatted_waypoints.append(str(waypoint))
        
        return "|".join(formatted_waypoints)
    
    def _parse_route(self, route_data: Dict[str, Any]) -> Route:
        """Parse Google Maps route data."""
        # Parse legs
        legs = []
        for leg_data in route_data.get("legs", []):
            legs.append(self._parse_route_leg(leg_data))
        
        # Parse bounds
        bounds = None
        bounds_data = route_data.get("bounds")
        if bounds_data:
            bounds = BoundingBox(
                north=bounds_data.get("northeast", {}).get("lat", 0),
                south=bounds_data.get("southwest", {}).get("lat", 0),
                east=bounds_data.get("northeast", {}).get("lng", 0),
                west=bounds_data.get("southwest", {}).get("lng", 0)
            )
        
        return Route(
            summary=route_data.get("summary", ""),
            legs=legs,
            waypoint_order=route_data.get("waypoint_order", []),
            overview_polyline=route_data.get("overview_polyline", {}).get("points"),
            bounds=bounds,
            copyrights=route_data.get("copyrights"),
            warnings=route_data.get("warnings", []),
            fare=route_data.get("fare")
        )
    
    def _parse_route_leg(self, leg_data: Dict[str, Any]) -> RouteLeg:
        """Parse route leg data."""
        # Parse start and end locations
        start_location_data = leg_data.get("start_location", {})
        end_location_data = leg_data.get("end_location", {})
        
        start_location = Coordinates(
            lat=start_location_data.get("lat", 0),
            lng=start_location_data.get("lng", 0)
        )
        end_location = Coordinates(
            lat=end_location_data.get("lat", 0),
            lng=end_location_data.get("lng", 0)
        )
        
        # Parse steps
        steps = []
        for step_data in leg_data.get("steps", []):
            steps.append(self._parse_route_step(step_data))
        
        # Parse via waypoints
        via_waypoints = []
        for wp_data in leg_data.get("via_waypoint", []):
            location_data = wp_data.get("location", {})
            via_waypoints.append(Coordinates(
                lat=location_data.get("lat", 0),
                lng=location_data.get("lng", 0)
            ))
        
        return RouteLeg(
            distance=leg_data.get("distance", {}),
            duration=leg_data.get("duration", {}),
            start_address=leg_data.get("start_address", ""),
            end_address=leg_data.get("end_address", ""),
            start_location=start_location,
            end_location=end_location,
            steps=steps,
            via_waypoints=via_waypoints,
            traffic_speed_entry=leg_data.get("traffic_speed_entry", []),
            via_waypoint=leg_data.get("via_waypoint", [])
        )
    
    def _parse_route_step(self, step_data: Dict[str, Any]) -> RouteStep:
        """Parse route step data."""
        # Parse start and end locations
        start_location_data = step_data.get("start_location", {})
        end_location_data = step_data.get("end_location", {})
        
        start_location = Coordinates(
            lat=start_location_data.get("lat", 0),
            lng=start_location_data.get("lng", 0)
        )
        end_location = Coordinates(
            lat=end_location_data.get("lat", 0),
            lng=end_location_data.get("lng", 0)
        )
        
        # Parse substeps if they exist
        substeps = []
        for substep_data in step_data.get("steps", []):
            substeps.append(self._parse_route_step(substep_data))
        
        return RouteStep(
            distance=step_data.get("distance", {}),
            duration=step_data.get("duration", {}),
            start_location=start_location,
            end_location=end_location,
            html_instructions=step_data.get("html_instructions"),
            maneuver=step_data.get("maneuver"),
            polyline=step_data.get("polyline", {}).get("points"),
            travel_mode=TravelMode(step_data.get("travel_mode", "driving")),
            steps=substeps
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def get_distance_matrix(self, request: DistanceMatrixRequest) -> DistanceMatrixResponse:
        """Get distance matrix using Google Maps."""
        if not self.is_available():
            raise ValueError("Google Maps provider is not configured")
        
        try:
            # Format origins and destinations
            origins = [self._format_location(origin) for origin in request.origins]
            destinations = [self._format_location(destination) for destination in request.destinations]
            
            # Build parameters
            params = self._build_params({
                "origins": "|".join(origins),
                "destinations": "|".join(destinations),
                "mode": request.travel_mode.value if request.travel_mode else "driving",
                "avoid": "|".join([a.value for a in request.avoid]) if request.avoid else None,
                "units": request.units or "metric",
                "departure_time": request.departure_time,
                "arrival_time": request.arrival_time,
                "traffic_model": request.traffic_model,
                "transit_mode": "|".join(request.transit_modes) if request.transit_modes else None,
                "transit_routing_preference": request.transit_routing_preference,
            })
            
            url = f"{self.base_url}/distancematrix/json"
            
            async with self.http_client.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get("status") == "OK":
                        return self._parse_distance_matrix(data)
                    else:
                        logger.error(f"Distance matrix failed: {data.get('status')}")
                        return DistanceMatrixResponse(
                            origin_addresses=[],
                            destination_addresses=[],
                            rows=[]
                        )
                else:
                    logger.error(f"Distance matrix failed with status: {response.status}")
                    return DistanceMatrixResponse(
                        origin_addresses=[],
                        destination_addresses=[],
                        rows=[]
                    )
                    
        except Exception as e:
            logger.error(f"Error getting distance matrix: {str(e)}")
            return DistanceMatrixResponse(
                origin_addresses=[],
                destination_addresses=[],
                rows=[]
            )
    
    def _parse_distance_matrix(self, data: Dict[str, Any]) -> DistanceMatrixResponse:
        """Parse distance matrix response."""
        rows = []
        for row_data in data.get("rows", []):
            elements = []
            for element_data in row_data.get("elements", []):
                elements.append(DistanceMatrixElement(
                    status=element_data.get("status", ""),
                    distance=element_data.get("distance"),
                    duration=element_data.get("duration"),
                    duration_in_traffic=element_data.get("duration_in_traffic"),
                    fare=element_data.get("fare")
                ))
            rows.append(DistanceMatrixRow(elements=elements))
        
        return DistanceMatrixResponse(
            origin_addresses=data.get("origin_addresses", []),
            destination_addresses=data.get("destination_addresses", []),
            rows=rows
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def search_places(self, request: PlaceSearchRequest) -> List[Place]:
        """Search for places using Google Maps."""
        if not self.is_available():
            raise ValueError("Google Maps provider is not configured")
        
        try:
            # Build parameters
            params = self._build_params({
                "query": request.query,
                "location": f"{request.location.lat},{request.location.lng}" if request.location else None,
                "radius": request.radius,
                "type": request.place_type.value if request.place_type else None,
                "keyword": request.keyword,
                "language": request.language,
                "minprice": request.min_price,
                "maxprice": request.max_price,
                "opennow": request.open_now,
                "pagetoken": request.page_token,
                "region": request.region,
            })
            
            url = f"{self.base_url}/place/textsearch/json"
            
            async with self.http_client.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get("status") == "OK":
                        results = data.get("results", [])
                        return [self._parse_place(r) for r in results]
                    else:
                        logger.error(f"Place search failed: {data.get('status')}")
                        return []
                else:
                    logger.error(f"Place search failed with status: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error searching places: {str(e)}")
            return []
    
    def _parse_place(self, data: Dict[str, Any]) -> Place:
        """Parse place data from Google Maps."""
        # Parse geometry
        geometry_data = data.get("geometry", {})
        location_data = geometry_data.get("location", {})
        location = Coordinates(
            lat=location_data.get("lat", 0),
            lng=location_data.get("lng", 0)
        )
        
        # Parse viewport
        viewport = None
        viewport_data = geometry_data.get("viewport")
        if viewport_data:
            viewport = BoundingBox(
                north=viewport_data.get("northeast", {}).get("lat", 0),
                south=viewport_data.get("southwest", {}).get("lat", 0),
                east=viewport_data.get("northeast", {}).get("lng", 0),
                west=viewport_data.get("southwest", {}).get("lng", 0)
            )
        
        geometry = Geometry(
            location=location,
            location_type=geometry_data.get("location_type", ""),
            viewport=viewport
        )
        
        return Place(
            place_id=data.get("place_id", ""),
            name=data.get("name", ""),
            formatted_address=data.get("formatted_address", ""),
            geometry=geometry,
            types=data.get("types", []),
            rating=data.get("rating"),
            user_ratings_total=data.get("user_ratings_total"),
            price_level=data.get("price_level"),
            opening_hours=data.get("opening_hours"),
            photos=data.get("photos", []),
            permanently_closed=data.get("permanently_closed"),
            business_status=data.get("business_status"),
            icon=data.get("icon"),
            icon_background_color=data.get("icon_background_color"),
            icon_mask_base_uri=data.get("icon_mask_base_uri"),
            plus_code=data.get("plus_code"),
            scope=data.get("scope"),
            vicinity=data.get("vicinity")
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def get_place_details(self, place_id: str, language: str = "en") -> Optional[Place]:
        """Get place details using Google Maps."""
        if not self.is_available():
            raise ValueError("Google Maps provider is not configured")
        
        try:
            params = self._build_params({
                "place_id": place_id,
                "language": language,
                "fields": ",".join([
                    "address_component", "adr_address", "business_status",
                    "formatted_address", "geometry", "icon", "name",
                    "permanently_closed", "photo", "place_id", "plus_code",
                    "type", "url", "utc_offset", "vicinity",
                    "formatted_phone_number", "international_phone_number",
                    "opening_hours", "website", "price_level", "rating",
                    "review", "user_ratings_total"
                ])
            })
            
            url = f"{self.base_url}/place/details/json"
            
            async with self.http_client.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get("status") == "OK":
                        result = data.get("result", {})
                        return self._parse_place_details(result)
                    else:
                        logger.error(f"Place details failed: {data.get('status')}")
                        return None
                else:
                    logger.error(f"Place details failed with status: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting place details: {str(e)}")
            return None
    
    def _parse_place_details(self, data: Dict[str, Any]) -> Place:
        """Parse detailed place data from Google Maps."""
        # Parse basic place info (reuse _parse_place for common fields)
        place = self._parse_place(data)
        
        # Update with additional fields
        place.website = data.get("website")
        place.phone_number = data.get("formatted_phone_number")
        place.international_phone_number = data.get("international_phone_number")
        place.utc_offset = data.get("utc_offset")
        place.adr_address = data.get("adr_address")
        place.reviews = data.get("reviews", [])
        
        return place
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def autocomplete(self, query: str, session_token: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get place autocomplete suggestions."""
        if not self.is_available():
            raise ValueError("Google Maps provider is not configured")
        
        try:
            params = self._build_params({
                "input": query,
                "sessiontoken": session_token,
            })
            
            url = f"{self.base_url}/place/autocomplete/json"
            
            async with self.http_client.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get("status") == "OK":
                        return data.get("predictions", [])
                    else:
                        logger.error(f"Autocomplete failed: {data.get('status')}")
                        return []
                else:
                    logger.error(f"Autocomplete failed with status: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error in autocomplete: {str(e)}")
            return []
    
    async def validate_connection(self) -> bool:
        """Validate connection to Google Maps API."""
        if not self.is_available():
            return False
        
        try:
            # Test with a simple geocode request
            params = self._build_params({
                "address": "Googleplex, Mountain View, CA",
            })
            
            url = f"{self.base_url}/geocode/json"
            
            async with self.http_client.session.get(url, params=params, timeout=10) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"Google Maps connection validation failed: {str(e)}")
            return False


class OpenStreetMapProvider(BaseMapsProvider):
    """OpenStreetMap/Nominatim provider (free, no API key required)."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {
                "base_url": "https://nominatim.openstreetmap.org",
                "user_agent": "WorldBrief360/1.0",
                "email": settings.OSM_EMAIL if hasattr(settings, 'OSM_EMAIL') else None,
                "delay": 1.0,  # Rate limiting delay
            }
        super().__init__(config)
        
        self.base_url = self.config["base_url"]
        self.headers = {
            "User-Agent": self.config["user_agent"]
        }
        
        if self.config.get("email"):
            self.headers["From"] = self.config["email"]
    
    def is_available(self) -> bool:
        """OpenStreetMap is always available."""
        return True
    
    async def _rate_limit(self):
        """Respect OSM rate limiting."""
        await asyncio.sleep(self.config.get("delay", 1.0))
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def geocode(self, request: GeocodeRequest) -> List[GeocodeResult]:
        """Geocode address using OpenStreetMap/Nominatim."""
        try:
            await self._rate_limit()
            
            params = {
                "q": request.address,
                "format": "jsonv2",
                "addressdetails": 1,
                "limit": 10,
                "accept-language": request.language or "en",
            }
            
            if request.bounds:
                params["viewbox"] = (
                    f"{request.bounds.west},{request.bounds.south},"
                    f"{request.bounds.east},{request.bounds.north}"
                )
                params["bounded"] = 1
            
            url = f"{self.base_url}/search"
            
            async with self.http_client.session.get(
                url, params=params, headers=self.headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if isinstance(data, list):
                        return [self._parse_osm_result(r) for r in data]
                    else:
                        return []
                else:
                    logger.error(f"OSM geocoding failed with status: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error in OSM geocoding: {str(e)}")
            return []
    
    def _parse_osm_result(self, data: Dict[str, Any]) -> GeocodeResult:
        """Parse OSM/Nominatim result."""
        # Parse address components
        address_components = []
        address_data = data.get("address", {})
        
        for key, value in address_data.items():
            if isinstance(value, str):
                address_components.append(AddressComponent(
                    long_name=value,
                    short_name=value,
                    types=[key]
                ))
        
        # Parse coordinates
        lat = float(data.get("lat", 0))
        lng = float(data.get("lon", 0))
        location = Coordinates(lat=lat, lng=lng)
        
        # Parse bounding box
        boundingbox = data.get("boundingbox", [])
        viewport = None
        if len(boundingbox) >= 4:
            viewport = BoundingBox(
                north=float(boundingbox[1]),
                south=float(boundingbox[0]),
                east=float(boundingbox[3]),
                west=float(boundingbox[2])
            )
        
        geometry = Geometry(
            location=location,
            location_type=data.get("type", ""),
            viewport=viewport
        )
        
        return GeocodeResult(
            formatted_address=data.get("display_name", ""),
            geometry=geometry,
            place_id=str(data.get("place_id", "")),
            types=[data.get("type", ""), data.get("class", "")],
            address_components=address_components,
            partial_match=False
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def reverse_geocode(self, request: ReverseGeocodeRequest) -> List[GeocodeResult]:
        """Reverse geocode using OpenStreetMap/Nominatim."""
        try:
            await self._rate_limit()
            
            params = {
                "lat": request.location.lat,
                "lon": request.location.lng,
                "format": "jsonv2",
                "addressdetails": 1,
                "zoom": 18,  # Most detailed
                "accept-language": request.language or "en",
            }
            
            url = f"{self.base_url}/reverse"
            
            async with self.http_client.session.get(
                url, params=params, headers=self.headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return [self._parse_osm_result(data)]
                else:
                    logger.error(f"OSM reverse geocoding failed with status: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error in OSM reverse geocoding: {str(e)}")
            return []
    
    # Note: OSM doesn't have built-in directions/distance matrix APIs
    # We'd need to use OSRM or similar for routing
    
    async def get_directions(self, request: DirectionsRequest) -> List[Route]:
        """OpenStreetMap doesn't provide directions directly."""
        logger.warning("OpenStreetMap provider doesn't support directions directly. Use OSRM instead.")
        return []
    
    async def get_distance_matrix(self, request: DistanceMatrixRequest) -> DistanceMatrixResponse:
        """OpenStreetMap doesn't provide distance matrix directly."""
        logger.warning("OpenStreetMap provider doesn't support distance matrix directly.")
        return DistanceMatrixResponse(origin_addresses=[], destination_addresses=[], rows=[])
    
    async def search_places(self, request: PlaceSearchRequest) -> List[Place]:
        """Search places using OpenStreetMap."""
        # For OSM, we can use the geocode endpoint with special parameters
        geocode_request = GeocodeRequest(
            address=request.query or "",
            bounds=request.bounds,
            language=request.language
        )
        
        geocode_results = await self.geocode(geocode_request)
        
        # Convert GeocodeResult to Place
        places = []
        for result in geocode_results:
            place = Place(
                place_id=result.place_id,
                name=result.formatted_address.split(",")[0] if result.formatted_address else "",
                formatted_address=result.formatted_address,
                geometry=result.geometry,
                types=result.types
            )
            places.append(place)
        
        return places
    
    async def get_place_details(self, place_id: str) -> Optional[Place]:
        """Get place details from OpenStreetMap."""
        # OSM doesn't have a direct place details endpoint
        # We could use the reverse geocoding with the place's coordinates
        # For now, return None
        return None
    
    async def autocomplete(self, query: str, session_token: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get autocomplete suggestions from OpenStreetMap."""
        try:
            await self._rate_limit()
            
            params = {
                "q": query,
                "format": "json",
                "addressdetails": 0,
                "limit": 5,
            }
            
            url = f"{self.base_url}/search"
            
            async with self.http_client.session.get(
                url, params=params, headers=self.headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if isinstance(data, list):
                        suggestions = []
                        for item in data:
                            suggestions.append({
                                "description": item.get("display_name", ""),
                                "place_id": str(item.get("place_id", "")),
                                "types": [item.get("type", "")],
                            })
                        return suggestions
                    else:
                        return []
                else:
                    logger.error(f"OSM autocomplete failed with status: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error in OSM autocomplete: {str(e)}")
            return []
    
    async def validate_connection(self) -> bool:
        """Validate connection to OpenStreetMap API."""
        try:
            # Test with a simple search
            params = {
                "q": "London",
                "format": "json",
                "limit": 1,
            }
            
            url = f"{self.base_url}/search"
            
            async with self.http_client.session.get(
                url, params=params, headers=self.headers, timeout=10
            ) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"OSM connection validation failed: {str(e)}")
            return False


class MapsClient:
    """
    Main maps client for WorldBrief 360.
    Supports multiple providers with fallback logic.
    """
    
    def __init__(
        self,
        default_provider: Optional[MapProvider] = None,
        providers_config: Optional[Dict[MapProvider, Dict[str, Any]]] = None
    ):
        self.default_provider = default_provider or MapProvider.GOOGLE_MAPS
        self.providers: Dict[MapProvider, BaseMapsProvider] = {}
        self._initialize_providers(providers_config or {})
        
        # Caches
        self.geocode_cache = TTLCache(maxsize=10000, ttl=86400)  # 24 hours
        self.directions_cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour
        self.places_cache = TTLCache(maxsize=5000, ttl=3600)  # 1 hour
        
        logger.info(f"Maps client initialized with default provider: {self.default_provider.value}")
    
    def _initialize_providers(self, providers_config: Dict[MapProvider, Dict[str, Any]]):
        """Initialize map providers."""
        provider_classes = {
            MapProvider.GOOGLE_MAPS: GoogleMapsProvider,
            MapProvider.OPENSTREETMAP: OpenStreetMapProvider,
        }
        
        for provider_type, provider_class in provider_classes.items():
            try:
                config = providers_config.get(provider_type, {})
                provider = provider_class(config)
                
                # Validate provider if it has API key requirements
                if hasattr(provider, 'is_available'):
                    if provider.is_available():
                        self.providers[provider_type] = provider
                        logger.info(f"Initialized maps provider: {provider_type.value}")
                    else:
                        logger.warning(f"Skipping unavailable maps provider: {provider_type.value}")
                else:
                    # Provider doesn't have availability check
                    self.providers[provider_type] = provider
                    logger.info(f"Initialized maps provider: {provider_type.value}")
                    
            except Exception as e:
                logger.error(f"Failed to initialize {provider_type.value}: {str(e)}")
    
    def get_provider(self, provider_type: Optional[MapProvider] = None) -> BaseMapsProvider:
        """
        Get maps provider.
        
        Args:
            provider_type: Provider type (uses default if None)
            
        Returns:
            Maps provider instance
            
        Raises:
            ValueError: If provider not available
        """
        provider_type = provider_type or self.default_provider
        
        if provider_type not in self.providers:
            # Try to get any available provider
            if self.providers:
                return next(iter(self.providers.values()))
            raise ValueError(f"No maps providers available. Requested: {provider_type.value}")
        
        return self.providers[provider_type]
    
    def get_available_providers(self) -> List[Dict[str, Any]]:
        """Get list of available maps providers."""
        available = []
        
        for provider_type, provider in self.providers.items():
            available.append({
                "provider": provider_type.value,
                "name": provider_type.name,
                "enabled": True,
                "description": str(provider)
            })
        
        return available
    
    def _generate_cache_key(self, prefix: str, data: Any) -> str:
        """Generate cache key."""
        import json
        
        key_str = json.dumps(data, sort_keys=True, default=str)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        return f"{prefix}:{key_hash}"
    
    async def geocode(
        self,
        address: str,
        provider: Optional[MapProvider] = None,
        **kwargs
    ) -> List[GeocodeResult]:
        """
        Geocode an address.
        
        Args:
            address: Address to geocode
            provider: Maps provider to use
            **kwargs: Additional parameters for GeocodeRequest
            
        Returns:
            List of geocode results
        """
        cache_key = self._generate_cache_key("geocode", {"address": address, **kwargs})
        
        if cache_key in self.geocode_cache:
            logger.debug(f"Cache hit for geocode: {address}")
            return self.geocode_cache[cache_key]
        
        try:
            request = GeocodeRequest(address=address, **kwargs)
            provider_instance = self.get_provider(provider)
            results = await provider_instance.geocode(request)
            
            # Cache the results
            self.geocode_cache[cache_key] = results
            
            return results
            
        except Exception as e:
            logger.error(f"Geocoding failed for address '{address}': {str(e)}")
            return []
    
    async def reverse_geocode(
        self,
        latitude: float,
        longitude: float,
        provider: Optional[MapProvider] = None,
        **kwargs
    ) -> List[GeocodeResult]:
        """
        Reverse geocode coordinates.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            provider: Maps provider to use
            **kwargs: Additional parameters for ReverseGeocodeRequest
            
        Returns:
            List of reverse geocode results
        """
        cache_key = self._generate_cache_key(
            "reverse_geocode", 
            {"lat": latitude, "lng": longitude, **kwargs}
        )
        
        if cache_key in self.geocode_cache:
            logger.debug(f"Cache hit for reverse geocode: ({latitude}, {longitude})")
            return self.geocode_cache[cache_key]
        
        try:
            location = Coordinates(lat=latitude, lng=longitude)
            request = ReverseGeocodeRequest(location=location, **kwargs)
            provider_instance = self.get_provider(provider)
            results = await provider_instance.reverse_geocode(request)
            
            # Cache the results
            self.geocode_cache[cache_key] = results
            
            return results
            
        except Exception as e:
            logger.error(f"Reverse geocoding failed for coordinates ({latitude}, {longitude}): {str(e)}")
            return []
    
    async def get_directions(
        self,
        origin: Union[str, Coordinates, Place],
        destination: Union[str, Coordinates, Place],
        provider: Optional[MapProvider] = None,
        **kwargs
    ) -> List[Route]:
        """
        Get directions between two locations.
        
        Args:
            origin: Starting location
            destination: Destination location
            provider: Maps provider to use
            **kwargs: Additional parameters for DirectionsRequest
            
        Returns:
            List of routes
        """
        cache_key = self._generate_cache_key(
            "directions",
            {"origin": str(origin), "destination": str(destination), **kwargs}
        )
        
        if cache_key in self.directions_cache:
            logger.debug(f"Cache hit for directions: {origin} -> {destination}")
            return self.directions_cache[cache_key]
        
        try:
            request = DirectionsRequest(
                origin=origin,
                destination=destination,
                **kwargs
            )
            provider_instance = self.get_provider(provider)
            routes = await provider_instance.get_directions(request)
            
            # Cache the results
            self.directions_cache[cache_key] = routes
            
            return routes
            
        except Exception as e:
            logger.error(f"Directions failed from {origin} to {destination}: {str(e)}")
            return []
    
    async def get_distance_matrix(
        self,
        origins: List[Union[str, Coordinates, Place]],
        destinations: List[Union[str, Coordinates, Place]],
        provider: Optional[MapProvider] = None,
        **kwargs
    ) -> DistanceMatrixResponse:
        """
        Get distance matrix between multiple origins and destinations.
        
        Args:
            origins: List of origin locations
            destinations: List of destination locations
            provider: Maps provider to use
            **kwargs: Additional parameters for DistanceMatrixRequest
            
        Returns:
            Distance matrix response
        """
        try:
            request = DistanceMatrixRequest(
                origins=origins,
                destinations=destinations,
                **kwargs
            )
            provider_instance = self.get_provider(provider)
            return await provider_instance.get_distance_matrix(request)
            
        except Exception as e:
            logger.error(f"Distance matrix failed: {str(e)}")
            return DistanceMatrixResponse(
                origin_addresses=[],
                destination_addresses=[],
                rows=[]
            )
    
    async def search_places(
        self,
        query: str,
        provider: Optional[MapProvider] = None,
        **kwargs
    ) -> List[Place]:
        """
        Search for places.
        
        Args:
            query: Search query
            provider: Maps provider to use
            **kwargs: Additional parameters for PlaceSearchRequest
            
        Returns:
            List of places
        """
        cache_key = self._generate_cache_key("places_search", {"query": query, **kwargs})
        
        if cache_key in self.places_cache:
            logger.debug(f"Cache hit for places search: {query}")
            return self.places_cache[cache_key]
        
        try:
            request = PlaceSearchRequest(query=query, **kwargs)
            provider_instance = self.get_provider(provider)
            places = await provider_instance.search_places(request)
            
            # Cache the results
            self.places_cache[cache_key] = places
            
            return places
            
        except Exception as e:
            logger.error(f"Place search failed for query '{query}': {str(e)}")
            return []
    
    async def get_place_details(
        self,
        place_id: str,
        provider: Optional[MapProvider] = None,
        **kwargs
    ) -> Optional[Place]:
        """
        Get detailed information about a place.
        
        Args:
            place_id: Place identifier
            provider: Maps provider to use
            **kwargs: Additional parameters
            
        Returns:
            Place details or None if not found
        """
        cache_key = self._generate_cache_key("place_details", {"place_id": place_id, **kwargs})
        
        if cache_key in self.places_cache:
            logger.debug(f"Cache hit for place details: {place_id}")
            return self.places_cache[cache_key]
        
        try:
            provider_instance = self.get_provider(provider)
            place = await provider_instance.get_place_details(place_id, **kwargs)
            
            # Cache the result
            if place:
                self.places_cache[cache_key] = place
            
            return place
            
        except Exception as e:
            logger.error(f"Failed to get place details for {place_id}: {str(e)}")
            return None
    
    async def calculate_distance(
        self,
        point1: Coordinates,
        point2: Coordinates,
        travel_mode: TravelMode = TravelMode.DRIVING
    ) -> float:
        """
        Calculate distance between two points.
        
        Args:
            point1: First point
            point2: Second point
            travel_mode: Travel mode for distance calculation
            
        Returns:
            Distance in kilometers
        """
        if travel_mode == TravelMode.FLYING:
            # Use Haversine formula for air distance
            return point1.distance_to(point2)
        else:
            # Use distance matrix for ground transportation
            matrix = await self.get_distance_matrix(
                origins=[point1],
                destinations=[point2],
                travel_mode=travel_mode
            )
            
            distance_element = matrix.get_distance(0, 0)
            if distance_element:
                distance_m = distance_element.get("value", 0)
                return distance_m / 1000  # Convert to km
            else:
                # Fall back to air distance
                return point1.distance_to(point2)
    
    async def find_nearby_places(
        self,
        location: Coordinates,
        radius: float = 1000,  # meters
        place_type: Optional[PlaceType] = None,
        keyword: Optional[str] = None,
        limit: int = 20,
        provider: Optional[MapProvider] = None
    ) -> List[Place]:
        """
        Find places near a location.
        
        Args:
            location: Center location
            radius: Search radius in meters
            place_type: Type of place to search for
            keyword: Additional keyword filter
            limit: Maximum number of results
            provider: Maps provider to use
            
        Returns:
            List of nearby places
        """
        try:
            # Convert radius from meters to kilometers for queries
            radius_km = radius / 1000
            
            # First, reverse geocode to get location name
            reverse_results = await self.reverse_geocode(
                latitude=location.lat,
                longitude=location.lng,
                provider=provider
            )
            
            if not reverse_results:
                return []
            
            location_name = reverse_results[0].formatted_address
            
            # Build search query
            query_parts = []
            if keyword:
                query_parts.append(keyword)
            if place_type:
                query_parts.append(place_type.value.replace("_", " "))
            query_parts.append(f"near {location_name}")
            
            query = " ".join(query_parts)
            
            # Search for places
            places = await self.search_places(
                query=query,
                location=location,
                radius=radius,
                place_type=place_type,
                keyword=keyword,
                provider=provider
            )
            
            # Filter by distance and limit
            filtered_places = []
            for place in places:
                distance = location.distance_to(place.geometry.location)
                if distance <= radius_km:
                    filtered_places.append(place)
                
                if len(filtered_places) >= limit:
                    break
            
            return filtered_places
            
        except Exception as e:
            logger.error(f"Failed to find nearby places: {str(e)}")
            return []
    
    async def batch_geocode(
        self,
        addresses: List[str],
        provider: Optional[MapProvider] = None,
        max_concurrent: int = 10,
        **kwargs
    ) -> Dict[str, List[GeocodeResult]]:
        """
        Geocode multiple addresses concurrently.
        
        Args:
            addresses: List of addresses to geocode
            provider: Maps provider to use
            max_concurrent: Maximum concurrent requests
            **kwargs: Additional parameters for geocoding
            
        Returns:
            Dictionary mapping addresses to geocode results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def geocode_with_semaphore(address):
            async with semaphore:
                return address, await self.geocode(address, provider=provider, **kwargs)
        
        tasks = [geocode_with_semaphore(address) for address in addresses]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        geocode_results = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch geocoding failed: {str(result)}")
            elif isinstance(result, tuple) and len(result) == 2:
                address, geocodes = result
                geocode_results[address] = geocodes
        
        return geocode_results
    
    async def validate_all_connections(self) -> Dict[str, bool]:
        """Validate connections for all providers."""
        results = {}
        
        for provider_type, provider in self.providers.items():
            try:
                results[provider_type.value] = await provider.validate_connection()
            except Exception as e:
                results[provider_type.value] = False
                logger.error(f"Connection validation failed for {provider_type.value}: {str(e)}")
        
        return results


# Factory function for dependency injection
def get_maps_client() -> MapsClient:
    """
    Factory function to create maps client.
    
    Returns:
        Configured MapsClient instance
    """
    return MapsClient()