# backend/app/integrations/wikipedia_client.py
"""
Wikipedia and Wikidata API client.
Provides access to:
- Wikipedia articles (multiple languages)
- Wikidata structured data
- Wikipedia search
- Article summaries and extracts
- Images and media
- Citations and references
- Historical revisions
- Related articles and categories
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Tuple
from urllib.parse import urlencode, quote, unquote

import aiohttp
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, validator, HttpUrl
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.config import settings
from app.core.logging_config import get_logger
from app.cache.redis_client import get_redis_client
from app.utils.http_client import get_http_client

logger = get_logger(__name__)


# ============ Data Models ============

class WikipediaLanguage(str, Enum):
    """Wikipedia language codes (top 20 by article count + common)."""
    EN = "en"  # English
    ES = "es"  # Spanish
    DE = "de"  # German
    FR = "fr"  # French
    RU = "ru"  # Russian
    JA = "ja"  # Japanese
    PT = "pt"  # Portuguese
    IT = "it"  # Italian
    ZH = "zh"  # Chinese
    AR = "ar"  # Arabic
    HI = "hi"  # Hindi
    BN = "bn"  # Bengali
    KO = "ko"  # Korean
    NL = "nl"  # Dutch
    TR = "tr"  # Turkish
    PL = "pl"  # Polish
    UK = "uk"  # Ukrainian
    FA = "fa"  # Persian
    VI = "vi"  # Vietnamese
    SV = "sv"  # Swedish
    MULTI = "multi"  # Multi-language (for Wikidata)


class ContentType(str, Enum):
    """Types of Wikipedia content."""
    ARTICLE = "article"
    SUMMARY = "summary"
    EXTRACT = "extract"
    FULL_TEXT = "full_text"
    INFOBOX = "infobox"
    CATEGORIES = "categories"
    REFERENCES = "references"
    IMAGES = "images"
    LINKS = "links"
    GEO = "geo"  # Geographic coordinates
    REVISION = "revision"
    WIKIDATA = "wikidata"
    SEARCH = "search"
    RANDOM = "random"
    FEATURED = "featured"  # Featured articles
    TRENDING = "trending"  # Trending articles


class WikipediaRequest(BaseModel):
    """Wikipedia API request model."""
    query: str  # Article title, search query, or Wikidata ID
    language: WikipediaLanguage = WikipediaLanguage.EN
    content_type: ContentType = ContentType.SUMMARY
    page_id: Optional[int] = None
    wikidata_id: Optional[str] = None  # Q-number
    limit: int = Field(default=10, ge=1, le=500)
    offset: int = 0
    extract_length: Optional[int] = None  # Characters for extract
    include_html: bool = False
    include_coordinates: bool = True
    include_images: bool = True
    include_references: bool = False
    include_categories: bool = False
    include_links: bool = False
    redirect: bool = True  # Follow redirects
    resolve_wikidata: bool = True  # Resolve Wikidata IDs
    cache_ttl: int = 86400  # Cache TTL in seconds (24 hours default)
    
    @validator('extract_length')
    def validate_extract_length(cls, v):
        if v is not None and (v < 50 or v > 10000):
            raise ValueError('Extract length must be between 50 and 10000 characters')
        return v
    
    @validator('wikidata_id')
    def validate_wikidata_id(cls, v):
        if v is not None:
            if not v.startswith('Q'):
                raise ValueError('Wikidata ID must start with Q')
            try:
                int(v[1:])
            except ValueError:
                raise ValueError('Wikidata ID must be Q followed by a number')
        return v


class WikipediaResponse(BaseModel):
    """Wikipedia API response model."""
    request_id: str = Field(default_factory=lambda: f"wiki_{int(time.time() * 1000)}")
    success: bool
    query: str
    language: WikipediaLanguage
    content_type: ContentType
    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    pagination: Optional[Dict[str, Any]] = None
    total_count: Optional[int] = None
    cached: bool = False
    cache_key: Optional[str] = None
    execution_time_ms: Optional[float] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WikipediaArticle(BaseModel):
    """Wikipedia article model."""
    title: str
    page_id: int
    language: WikipediaLanguage
    url: HttpUrl
    wikidata_id: Optional[str] = None
    summary: Optional[str] = None
    extract: Optional[str] = None
    full_text: Optional[str] = None
    infobox: Optional[Dict[str, Any]] = None
    images: List[Dict[str, Any]] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    references: List[Dict[str, Any]] = Field(default_factory=list)
    links: List[Dict[str, Any]] = Field(default_factory=list)
    coordinates: Optional[Dict[str, float]] = None  # lat, lon
    thumbnail: Optional[Dict[str, Any]] = None
    pageviews: Optional[Dict[str, int]] = None  # Last 30 days views
    last_modified: Optional[datetime] = None
    revision_id: Optional[int] = None
    disambiguation: bool = False
    featured: bool = False
    importance: Optional[str] = None  # High, Medium, Low
    article_length: Optional[int] = None  # Characters


class WikidataEntity(BaseModel):
    """Wikidata entity model."""
    id: str  # Q-number
    label: Optional[str] = None
    description: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)
    claims: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    sitelinks: Dict[str, Dict[str, str]] = Field(default_factory=dict)  # Wikipedia links
    type: str = "item"  # item, property, lexeme
    lastrevid: Optional[int] = None
    modified: Optional[datetime] = None
    
    def get_claim_value(self, property_id: str) -> Optional[Any]:
        """Get value for a Wikidata property."""
        if property_id in self.claims:
            claims = self.claims[property_id]
            if claims:
                # Return the first claim's main value
                return claims[0].get('mainsnak', {}).get('datavalue', {}).get('value')
        return None
    
    def get_wikipedia_link(self, language: str = "en") -> Optional[str]:
        """Get Wikipedia URL for a specific language."""
        sitelink = self.sitelinks.get(f"{language}wiki")
        if sitelink:
            return sitelink.get('url')
        return None


class WikipediaSearchResult(BaseModel):
    """Wikipedia search result."""
    title: str
    page_id: int
    language: WikipediaLanguage
    url: HttpUrl
    snippet: Optional[str] = None
    size: Optional[int] = None  # Article size in bytes
    word_count: Optional[int] = None
    timestamp: Optional[datetime] = None  # Last edit
    categories: List[str] = Field(default_factory=list)
    score: Optional[float] = None  # Search relevance score
    thumbnail: Optional[Dict[str, Any]] = None
    description: Optional[str] = None


class WikipediaPageview(BaseModel):
    """Wikipedia pageview statistics."""
    article: str
    language: WikipediaLanguage
    views: Dict[str, int] = Field(default_factory=dict)  # date -> views
    total_views: int = 0
    average_views: float = 0.0
    peak_views: int = 0
    peak_date: Optional[str] = None
    trend: Optional[str] = None  # increasing, decreasing, stable


# ============ Wikipedia Client ============

class WikipediaClient:
    """
    Wikipedia and Wikidata API client with caching and rate limiting.
    """
    
    def __init__(self):
        self.http_client = get_http_client()
        self.redis = get_redis_client()
        self.default_cache_ttl = 86400  # 24 hours
        self.initialized = False
        
        # API endpoints
        self.endpoints = {
            "api": {
                "base": "https://{language}.wikipedia.org/w/api.php",
                "action": "query",
                "format": "json",
                "formatversion": "2",
            },
            "wikidata": {
                "base": "https://www.wikidata.org/w/api.php",
                "entity": "https://www.wikidata.org/wiki/Special:EntityData/{id}.json",
            },
            "pageviews": {
                "base": "https://wikimedia.org/api/rest_v1/metrics/pageviews",
            },
            "geo": {
                "base": "https://{language}.wikipedia.org/wiki/Special:GoToLinkedPage",
            }
        }
        
        # Rate limiting (Wikipedia requests per minute)
        self.rate_limits = {
            WikipediaLanguage.EN: 100,
            WikipediaLanguage.DE: 50,
            WikipediaLanguage.FR: 50,
            WikipediaLanguage.ES: 50,
            WikipediaLanguage.RU: 30,
            WikipediaLanguage.JA: 30,
            WikipediaLanguage.ZH: 30,
            "default": 25,
        }
        
        # Common properties mapping
        self.wikidata_properties = {
            # Basic identifiers
            "instance_of": "P31",
            "subclass_of": "P279",
            "part_of": "P361",
            
            # Geography
            "coordinate_location": "P625",
            "country": "P17",
            "located_in": "P131",
            "continent": "P30",
            "capital": "P36",
            "population": "P1082",
            "area": "P2046",
            "elevation": "P2044",
            
            # People
            "date_of_birth": "P569",
            "date_of_death": "P570",
            "place_of_birth": "P19",
            "place_of_death": "P20",
            "occupation": "P106",
            "gender": "P21",
            
            # Organizations
            "founded_by": "P112",
            "inception": "P571",
            "dissolved": "P576",
            "headquarters": "P159",
            
            # Events
            "point_in_time": "P585",
            "start_time": "P580",
            "end_time": "P582",
            "location": "P276",
            
            # Creative works
            "author": "P50",
            "publication_date": "P577",
            "language": "P407",
            "genre": "P136",
            
            # Scientific
            "taxon_name": "P225",
            "atomic_number": "P1086",
            "chemical_formula": "P274",
            
            # Media
            "image": "P18",
            "official_website": "P856",
            "twitter_username": "P2002",
            "instagram_username": "P2003",
            
            # WorldBrief 360 specific
            "news_events": "P1196",  # related to current events
            "economic_indicator": "custom_001",  # Custom mapping
            "political_entity": "P1142",  # political ideology
        }
    
    async def initialize(self) -> None:
        """Initialize the Wikipedia client."""
        if self.initialized:
            return
        
        logger.info("Initializing Wikipedia API client...")
        
        # Load custom mappings if available
        await self._load_custom_mappings()
        
        self.initialized = True
        logger.info("Wikipedia API client initialized")
    
    async def _load_custom_mappings(self) -> None:
        """Load custom Wikipedia/Wikidata mappings."""
        # This could load from database or configuration
        # For now, initialize empty
        self.custom_mappings = {}
    
    async def _get_cache_key(self, request: WikipediaRequest) -> str:
        """Generate cache key for request."""
        request_dict = request.dict(exclude={'cache_ttl'})
        request_json = json.dumps(request_dict, sort_keys=True)
        request_hash = hashlib.md5(request_json.encode()).hexdigest()
        return f"wiki:{request.language.value}:{request.content_type.value}:{request_hash}"
    
    async def fetch(self, request: WikipediaRequest) -> WikipediaResponse:
        """
        Fetch data from Wikipedia/Wikidata.
        
        Args:
            request: Wikipedia request
            
        Returns:
            WikipediaResponse with data
        """
        start_time = datetime.utcnow()
        
        # Check cache first
        cache_key = await self._get_cache_key(request)
        cached_data = await self.redis.get(cache_key)
        if cached_data:
            response_data = json.loads(cached_data)
            response_data['cached'] = True
            response_data['cache_key'] = cache_key
            response_data['execution_time_ms'] = (datetime.utcnow() - start_time).total_seconds() * 1000
            return WikipediaResponse(**response_data)
        
        try:
            # Route to appropriate handler
            handler_map = {
                ContentType.ARTICLE: self._fetch_article,
                ContentType.SUMMARY: self._fetch_summary,
                ContentType.EXTRACT: self._fetch_extract,
                ContentType.FULL_TEXT: self._fetch_full_text,
                ContentType.INFOBOX: self._fetch_infobox,
                ContentType.CATEGORIES: self._fetch_categories,
                ContentType.REFERENCES: self._fetch_references,
                ContentType.IMAGES: self._fetch_images,
                ContentType.LINKS: self._fetch_links,
                ContentType.GEO: self._fetch_geo,
                ContentType.REVISION: self._fetch_revision,
                ContentType.WIKIDATA: self._fetch_wikidata,
                ContentType.SEARCH: self._fetch_search,
                ContentType.RANDOM: self._fetch_random,
                ContentType.FEATURED: self._fetch_featured,
                ContentType.TRENDING: self._fetch_trending,
            }
            
            handler = handler_map.get(request.content_type)
            if not handler:
                raise ValueError(f"Unsupported content type: {request.content_type}")
            
            data = await handler(request)
            
            # Create response
            response = WikipediaResponse(
                success=True,
                query=request.query,
                language=request.language,
                content_type=request.content_type,
                data=data.get("data", {}),
                metadata=data.get("metadata", {}),
                pagination=data.get("pagination"),
                total_count=data.get("total_count"),
                cached=False,
                cache_key=cache_key,
                execution_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
            
            # Cache the response
            cache_ttl = request.cache_ttl or self.default_cache_ttl
            await self.redis.setex(
                cache_key,
                cache_ttl,
                json.dumps(response.dict())
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to fetch Wikipedia data: {str(e)}", exc_info=True)
            return WikipediaResponse(
                success=False,
                query=request.query,
                language=request.language,
                content_type=request.content_type,
                error=str(e),
                execution_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    # ============ Content Handlers ============
    
    async def _fetch_summary(self, request: WikipediaRequest) -> Dict[str, Any]:
        """Fetch article summary."""
        # Try to get from page summary API
        base_url = self.endpoints["api"]["base"].format(language=request.language.value)
        
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "extracts|pageimages|info|coordinates",
            "exintro": "1",
            "explaintext": "1",
            "piprop": "thumbnail|name",
            "pithumbsize": "300",
            "inprop": "url",
            "colimit": "max",
            "titles": request.query,
            "redirects": "1" if request.redirect else "0",
        }
        
        if request.extract_length:
            params["exchars"] = min(request.extract_length, 500)  # Max 500 for summary
        
        # Make request
        async with self.http_client.get(base_url, params=params) as response:
            if response.status != 200:
                raise Exception(f"Wikipedia API error: {response.status}")
            
            data = await response.json()
            
            # Parse response
            pages = data.get("query", {}).get("pages", [])
            if not pages:
                raise Exception(f"Article not found: {request.query}")
            
            page = pages[0]
            if "missing" in page:
                raise Exception(f"Article not found: {request.query}")
            
            # Build article object
            article = self._parse_page_data(page, request.language)
            
            # Get Wikidata if requested
            if request.resolve_wikidata and not article.wikidata_id:
                wikidata_id = await self._get_wikidata_id(article.title, request.language)
                article.wikidata_id = wikidata_id
            
            return {
                "data": article.dict(),
                "metadata": {
                    "api_timestamp": data.get("curtimestamp"),
                    "api_version": data.get("version", {}).get("version"),
                }
            }
    
    async def _fetch_article(self, request: WikipediaRequest) -> Dict[str, Any]:
        """Fetch full article with all requested components."""
        # Get basic article info
        summary_request = WikipediaRequest(
            query=request.query,
            language=request.language,
            content_type=ContentType.SUMMARY,
            redirect=request.redirect,
            resolve_wikidata=request.resolve_wikidata,
        )
        
        summary_data = await self._fetch_summary(summary_request)
        article_data = summary_data["data"]
        
        # Fetch additional components in parallel
        tasks = []
        
        if request.include_images:
            images_request = WikipediaRequest(
                query=request.query,
                language=request.language,
                content_type=ContentType.IMAGES,
                page_id=article_data.get("page_id"),
            )
            tasks.append(self._fetch_images(images_request))
        
        if request.include_categories:
            categories_request = WikipediaRequest(
                query=request.query,
                language=request.language,
                content_type=ContentType.CATEGORIES,
                page_id=article_data.get("page_id"),
            )
            tasks.append(self._fetch_categories(categories_request))
        
        if request.include_references:
            references_request = WikipediaRequest(
                query=request.query,
                language=request.language,
                content_type=ContentType.REFERENCES,
                page_id=article_data.get("page_id"),
            )
            tasks.append(self._fetch_references(references_request))
        
        if request.include_links:
            links_request = WikipediaRequest(
                query=request.query,
                language=request.language,
                content_type=ContentType.LINKS,
                page_id=article_data.get("page_id"),
            )
            tasks.append(self._fetch_links(links_request))
        
        # Execute parallel requests
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Merge results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error fetching component: {str(result)}")
                    continue
                
                if tasks[i].__name__ == "_fetch_images":
                    article_data["images"] = result["data"].get("images", [])
                elif tasks[i].__name__ == "_fetch_categories":
                    article_data["categories"] = result["data"].get("categories", [])
                elif tasks[i].__name__ == "_fetch_references":
                    article_data["references"] = result["data"].get("references", [])
                elif tasks[i].__name__ == "_fetch_links":
                    article_data["links"] = result["data"].get("links", [])
        
        # Get pageviews if available
        try:
            pageviews = await self.get_pageviews(
                article_data["title"],
                request.language
            )
            article_data["pageviews"] = pageviews.dict() if pageviews else None
        except Exception as e:
            logger.warning(f"Failed to fetch pageviews: {str(e)}")
        
        return {
            "data": article_data,
            "metadata": summary_data.get("metadata", {})
        }
    
    async def _fetch_extract(self, request: WikipediaRequest) -> Dict[str, Any]:
        """Fetch article extract (longer than summary)."""
        base_url = self.endpoints["api"]["base"].format(language=request.language.value)
        
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "extracts",
            "explaintext": "1",
            "titles": request.query,
            "redirects": "1" if request.redirect else "0",
        }
        
        if request.extract_length:
            params["exchars"] = min(request.extract_length, 20000)  # Max 20k for extract
        else:
            params["exlimit"] = "1"
        
        # Make request
        async with self.http_client.get(base_url, params=params) as response:
            if response.status != 200:
                raise Exception(f"Wikipedia API error: {response.status}")
            
            data = await response.json()
            
            pages = data.get("query", {}).get("pages", [])
            if not pages or "missing" in pages[0]:
                raise Exception(f"Article not found: {request.query}")
            
            page = pages[0]
            extract = page.get("extract", "")
            
            return {
                "data": {
                    "title": page.get("title"),
                    "page_id": page.get("pageid"),
                    "extract": extract,
                    "extract_length": len(extract),
                },
                "metadata": {
                    "has_more": len(extract) >= (request.extract_length or 20000)
                }
            }
    
    async def _fetch_full_text(self, request: WikipediaRequest) -> Dict[str, Any]:
        """Fetch full article text (HTML or plain text)."""
        base_url = self.endpoints["api"]["base"].format(language=request.language.value)
        
        params = {
            "action": "parse",
            "format": "json",
            "page": request.query,
            "prop": "text|images|links|categories",
            "redirects": "1" if request.redirect else "0",
        }
        
        if not request.include_html:
            params["section"] = "0"  # Only main content
            params["disableeditsection"] = "1"
            params["disabletoc"] = "1"
        
        # Make request
        async with self.http_client.get(base_url, params=params) as response:
            if response.status != 200:
                raise Exception(f"Wikipedia API error: {response.status}")
            
            data = await response.json()
            
            if "error" in data:
                raise Exception(f"Wikipedia API error: {data['error'].get('info')}")
            
            parse_data = data.get("parse", {})
            
            # Extract text
            if request.include_html:
                content = parse_data.get("text", {}).get("*", "")
                # Clean up HTML
                soup = BeautifulSoup(content, 'html.parser')
                # Remove edit links, table of contents, etc.
                for element in soup.find_all(['span', 'div'], class_=re.compile(r'edit|toc|mw')):
                    element.decompose()
                content = str(soup)
            else:
                # Get plain text
                text_params = params.copy()
                text_params["prop"] = "text"
                text_params["section"] = "0"
                
                async with self.http_client.get(base_url, params=text_params) as text_response:
                    text_data = await text_response.json()
                    text_parse = text_data.get("parse", {})
                    content = text_parse.get("text", {}).get("*", "")
                    
                    # Extract text from HTML
                    soup = BeautifulSoup(content, 'html.parser')
                    content = soup.get_text(separator='\n', strip=True)
            
            return {
                "data": {
                    "title": parse_data.get("title"),
                    "page_id": parse_data.get("pageid"),
                    "content": content,
                    "content_type": "html" if request.include_html else "text",
                    "content_length": len(content),
                    "images": parse_data.get("images", []),
                    "links": parse_data.get("links", []),
                    "categories": parse_data.get("categories", []),
                }
            }
    
    async def _fetch_infobox(self, request: WikipediaRequest) -> Dict[str, Any]:
        """Fetch article infobox data."""
        # First get page HTML to extract infobox
        base_url = f"https://{request.language.value}.wikipedia.org/wiki/{quote(request.query)}"
        
        async with self.http_client.get(base_url) as response:
            if response.status != 200:
                raise Exception(f"Wikipedia page error: {response.status}")
            
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find infobox
            infobox = soup.find('table', class_='infobox')
            if not infobox:
                # Try other infobox classes
                infobox = soup.find('table', class_=re.compile(r'infobox'))
            
            if not infobox:
                return {"data": {"infobox": {}, "has_infobox": False}}
            
            # Parse infobox
            infobox_data = self._parse_infobox(infobox)
            
            return {
                "data": {
                    "infobox": infobox_data,
                    "has_infobox": True,
                }
            }
    
    def _parse_infobox(self, infobox) -> Dict[str, Any]:
        """Parse Wikipedia infobox HTML."""
        data = {}
        
        # Extract caption/title
        caption = infobox.find('caption')
        if caption:
            data['caption'] = caption.get_text(strip=True)
        
        # Extract rows
        rows = infobox.find_all('tr')
        for row in rows:
            # Skip header rows
            if row.find('th', colspan=True):
                continue
            
            # Get key (th) and value (td)
            th = row.find('th')
            td = row.find('td')
            
            if th and td:
                key = th.get_text(strip=True).lower().replace(' ', '_')
                value = td.get_text(separator=' ', strip=True)
                
                # Clean up value
                value = re.sub(r'\[\d+\]', '', value)  # Remove citation markers
                value = value.strip()
                
                if value:
                    data[key] = value
        
        return data
    
    async def _fetch_categories(self, request: WikipediaRequest) -> Dict[str, Any]:
        """Fetch article categories."""
        base_url = self.endpoints["api"]["base"].format(language=request.language.value)
        
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "categories",
            "cllimit": request.limit,
            "clshow": "!hidden",  # Exclude hidden categories
            "titles": request.query if not request.page_id else "",
            "pageids": request.page_id if request.page_id else "",
            "redirects": "1" if request.redirect else "0",
        }
        
        # Make request
        async with self.http_client.get(base_url, params=params) as response:
            if response.status != 200:
                raise Exception(f"Wikipedia API error: {response.status}")
            
            data = await response.json()
            
            pages = data.get("query", {}).get("pages", [])
            if not pages:
                raise Exception(f"Article not found")
            
            page = pages[0]
            categories = page.get("categories", [])
            
            # Extract category names
            category_names = [cat.get("title", "").replace("Category:", "") for cat in categories]
            
            return {
                "data": {
                    "categories": category_names,
                    "count": len(category_names),
                },
                "pagination": {
                    "continue": data.get("continue", {}).get("clcontinue"),
                }
            }
    
    async def _fetch_references(self, request: WikipediaRequest) -> Dict[str, Any]:
        """Fetch article references/citations."""
        # Use action=parse to get references
        base_url = self.endpoints["api"]["base"].format(language=request.language.value)
        
        params = {
            "action": "parse",
            "format": "json",
            "page": request.query if not request.page_id else "",
            "pageid": request.page_id if request.page_id else "",
            "prop": "externallinks",
            "redirects": "1" if request.redirect else "0",
        }
        
        # Make request
        async with self.http_client.get(base_url, params=params) as response:
            if response.status != 200:
                raise Exception(f"Wikipedia API error: {response.status}")
            
            data = await response.json()
            
            if "error" in data:
                raise Exception(f"Wikipedia API error: {data['error'].get('info')}")
            
            parse_data = data.get("parse", {})
            externallinks = parse_data.get("externallinks", [])
            
            # Also try to get references from HTML
            ref_params = params.copy()
            ref_params["prop"] = "text"
            ref_params["section"] = "0"
            
            async with self.http_client.get(base_url, params=ref_params) as ref_response:
                ref_data = await ref_response.json()
                ref_parse = ref_data.get("parse", {})
                html_content = ref_parse.get("text", {}).get("*", "")
                
                # Extract references from HTML
                soup = BeautifulSoup(html_content, 'html.parser')
                ref_elements = soup.find_all('sup', class_='reference')
                
                references = []
                for ref in ref_elements:
                    link = ref.find('a', href=True)
                    if link:
                        references.append({
                            "text": link.get_text(strip=True),
                            "href": link['href'],
                            "id": link.get('id'),
                        })
            
            return {
                "data": {
                    "externallinks": externallinks[:request.limit],
                    "footnotes": references[:request.limit],
                    "total_externallinks": len(externallinks),
                    "total_footnotes": len(references),
                }
            }
    
    async def _fetch_images(self, request: WikipediaRequest) -> Dict[str, Any]:
        """Fetch article images."""
        base_url = self.endpoints["api"]["base"].format(language=request.language.value)
        
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "images|pageimages",
            "imlimit": request.limit,
            "titles": request.query if not request.page_id else "",
            "pageids": request.page_id if request.page_id else "",
            "redirects": "1" if request.redirect else "0",
        }
        
        # Make request
        async with self.http_client.get(base_url, params=params) as response:
            if response.status != 200:
                raise Exception(f"Wikipedia API error: {response.status}")
            
            data = await response.json()
            
            pages = data.get("query", {}).get("pages", [])
            if not pages:
                raise Exception(f"Article not found")
            
            page = pages[0]
            images = page.get("images", [])
            
            # Get thumbnail if available
            thumbnail = page.get("thumbnail")
            
            # Fetch image info for each image
            image_details = []
            for img in images[:10]:  # Limit to 10 images for performance
                img_name = img.get("title", "")
                if img_name.startswith("File:"):
                    img_details = await self._get_image_info(img_name, request.language)
                    if img_details:
                        image_details.append(img_details)
            
            return {
                "data": {
                    "images": image_details,
                    "thumbnail": thumbnail,
                    "count": len(image_details),
                },
                "pagination": {
                    "continue": data.get("continue", {}).get("imcontinue"),
                }
            }
    
    async def _get_image_info(self, image_name: str, language: WikipediaLanguage) -> Optional[Dict[str, Any]]:
        """Get detailed information about an image."""
        base_url = self.endpoints["api"]["base"].format(language=language.value)
        
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "titles": image_name,
            "prop": "imageinfo",
            "iiprop": "url|size|mime|thumbmime|user|timestamp|extmetadata",
            "iiurlwidth": "300",
        }
        
        async with self.http_client.get(base_url, params=params) as response:
            if response.status != 200:
                return None
            
            data = await response.json()
            pages = data.get("query", {}).get("pages", [])
            if not pages:
                return None
            
            page = pages[0]
            imageinfo = page.get("imageinfo", [])
            if not imageinfo:
                return None
            
            info = imageinfo[0]
            return {
                "title": page.get("title", "").replace("File:", ""),
                "url": info.get("url"),
                "thumbnail_url": info.get("thumburl"),
                "description_url": info.get("descriptionurl"),
                "size": info.get("size"),
                "width": info.get("width"),
                "height": info.get("height"),
                "mime": info.get("mime"),
                "user": info.get("user"),
                "timestamp": info.get("timestamp"),
                "metadata": info.get("extmetadata", {}),
            }
    
    async def _fetch_links(self, request: WikipediaRequest) -> Dict[str, Any]:
        """Fetch article links."""
        base_url = self.endpoints["api"]["base"].format(language=request.language.value)
        
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "links",
            "pllimit": request.limit,
            "plnamespace": "0",  # Only main namespace articles
            "titles": request.query if not request.page_id else "",
            "pageids": request.page_id if request.page_id else "",
            "redirects": "1" if request.redirect else "0",
        }
        
        # Make request
        async with self.http_client.get(base_url, params=params) as response:
            if response.status != 200:
                raise Exception(f"Wikipedia API error: {response.status}")
            
            data = await response.json()
            
            pages = data.get("query", {}).get("pages", [])
            if not pages:
                raise Exception(f"Article not found")
            
            page = pages[0]
            links = page.get("links", [])
            
            # Get basic info about linked pages
            linked_pages = []
            for link in links[:50]:  # Limit to 50 for performance
                linked_pages.append({
                    "title": link.get("title"),
                    "page_id": link.get("pageid"),
                })
            
            return {
                "data": {
                    "links": linked_pages,
                    "count": len(linked_pages),
                },
                "pagination": {
                    "continue": data.get("continue", {}).get("plcontinue"),
                }
            }
    
    async def _fetch_geo(self, request: WikipediaRequest) -> Dict[str, Any]:
        """Fetch geographic coordinates."""
        base_url = self.endpoints["api"]["base"].format(language=request.language.value)
        
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "coordinates",
            "colimit": "max",
            "titles": request.query,
            "redirects": "1" if request.redirect else "0",
        }
        
        # Make request
        async with self.http_client.get(base_url, params=params) as response:
            if response.status != 200:
                raise Exception(f"Wikipedia API error: {response.status}")
            
            data = await response.json()
            
            pages = data.get("query", {}).get("pages", [])
            if not pages:
                raise Exception(f"Article not found: {request.query}")
            
            page = pages[0]
            coordinates = page.get("coordinates", [])
            
            if not coordinates:
                return {"data": {"coordinates": None, "has_coordinates": False}}
            
            coord = coordinates[0]
            return {
                "data": {
                    "coordinates": {
                        "lat": coord.get("lat"),
                        "lon": coord.get("lon"),
                        "globe": coord.get("globe", "earth"),
                    },
                    "has_coordinates": True,
                }
            }
    
    async def _fetch_revision(self, request: WikipediaRequest) -> Dict[str, Any]:
        """Fetch article revision history."""
        base_url = self.endpoints["api"]["base"].format(language=request.language.value)
        
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "revisions",
            "rvlimit": request.limit,
            "rvprop": "ids|timestamp|user|comment|size",
            "rvdir": "newer",  # or "older"
            "titles": request.query,
            "redirects": "1" if request.redirect else "0",
        }
        
        # Make request
        async with self.http_client.get(base_url, params=params) as response:
            if response.status != 200:
                raise Exception(f"Wikipedia API error: {response.status}")
            
            data = await response.json()
            
            pages = data.get("query", {}).get("pages", [])
            if not pages:
                raise Exception(f"Article not found: {request.query}")
            
            page = pages[0]
            revisions = page.get("revisions", [])
            
            return {
                "data": {
                    "revisions": revisions,
                    "count": len(revisions),
                    "latest_revid": page.get("lastrevid"),
                },
                "pagination": {
                    "continue": data.get("continue", {}).get("rvcontinue"),
                }
            }
    
    async def _fetch_wikidata(self, request: WikipediaRequest) -> Dict[str, Any]:
        """Fetch Wikidata entity."""
        # Determine Wikidata ID
        wikidata_id = request.wikidata_id
        
        if not wikidata_id and request.query:
            # Try to get Wikidata ID from Wikipedia article
            wikidata_id = await self._get_wikidata_id(request.query, request.language)
        
        if not wikidata_id:
            raise Exception("Could not determine Wikidata ID")
        
        # Fetch entity data
        entity_url = self.endpoints["wikidata"]["entity"].format(id=wikidata_id)
        
        async with self.http_client.get(entity_url) as response:
            if response.status != 200:
                raise Exception(f"Wikidata API error: {response.status}")
            
            data = await response.json()
            entity_data = data.get("entities", {}).get(wikidata_id)
            
            if not entity_data:
                raise Exception(f"Wikidata entity not found: {wikidata_id}")
            
            # Parse entity
            entity = self._parse_wikidata_entity(entity_data)
            
            return {
                "data": entity.dict(),
                "metadata": {
                    "fetched_from": "wikidata",
                    "entity_id": wikidata_id,
                }
            }
    
    async def _get_wikidata_id(self, title: str, language: WikipediaLanguage) -> Optional[str]:
        """Get Wikidata ID for a Wikipedia article."""
        cache_key = f"wiki:wikidata_id:{language.value}:{hashlib.md5(title.encode()).hexdigest()}"
        
        # Check cache
        cached = await self.redis.get(cache_key)
        if cached:
            return cached.decode() if cached != "null" else None
        
        base_url = self.endpoints["api"]["base"].format(language=language.value)
        
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "pageprops",
            "titles": title,
            "ppprop": "wikibase_item",
        }
        
        async with self.http_client.get(base_url, params=params) as response:
            if response.status != 200:
                return None
            
            data = await response.json()
            pages = data.get("query", {}).get("pages", [])
            
            if not pages:
                return None
            
            page = pages[0]
            wikidata_id = page.get("pageprops", {}).get("wikibase_item")
            
            # Cache result
            await self.redis.setex(cache_key, 604800, wikidata_id or "null")  # 7 days
            
            return wikidata_id
    
    def _parse_wikidata_entity(self, entity_data: Dict[str, Any]) -> WikidataEntity:
        """Parse Wikidata entity data."""
        return WikidataEntity(
            id=entity_data.get("id"),
            label=entity_data.get("labels", {}).get("en", {}).get("value"),
            description=entity_data.get("descriptions", {}).get("en", {}).get("value"),
            aliases=[alias.get("value") for alias in entity_data.get("aliases", {}).get("en", [])],
            claims=entity_data.get("claims", {}),
            sitelinks=entity_data.get("sitelinks", {}),
            type=entity_data.get("type"),
            lastrevid=entity_data.get("lastrevid"),
            modified=entity_data.get("modified"),
        )
    
    async def _fetch_search(self, request: WikipediaRequest) -> Dict[str, Any]:
        """Search Wikipedia."""
        base_url = self.endpoints["api"]["base"].format(language=request.language.value)
        
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "list": "search",
            "srsearch": request.query,
            "srlimit": request.limit,
            "sroffset": request.offset,
            "srprop": "size|wordcount|timestamp|snippet",
            "srwhat": "text",  # text, title, nearmatch
        }
        
        # Make request
        async with self.http_client.get(base_url, params=params) as response:
            if response.status != 200:
                raise Exception(f"Wikipedia API error: {response.status}")
            
            data = await response.json()
            
            search_results = data.get("query", {}).get("search", [])
            total_hits = data.get("query", {}).get("searchinfo", {}).get("totalhits", 0)
            
            # Format results
            results = []
            for result in search_results:
                results.append(WikipediaSearchResult(
                    title=result.get("title"),
                    page_id=result.get("pageid"),
                    language=request.language,
                    url=f"https://{request.language.value}.wikipedia.org/wiki/{quote(result.get('title'))}",
                    snippet=result.get("snippet"),
                    size=result.get("size"),
                    word_count=result.get("wordcount"),
                    timestamp=result.get("timestamp"),
                ).dict())
            
            return {
                "data": {
                    "results": results,
                    "query": request.query,
                },
                "metadata": {
                    "total_hits": total_hits,
                    "search_time": data.get("query", {}).get("searchinfo", {}).get("searchtime"),
                },
                "pagination": {
                    "offset": request.offset,
                    "limit": request.limit,
                    "total": total_hits,
                    "has_more": request.offset + len(results) < total_hits,
                },
                "total_count": total_hits,
            }
    
    async def _fetch_random(self, request: WikipediaRequest) -> Dict[str, Any]:
        """Get random Wikipedia articles."""
        base_url = self.endpoints["api"]["base"].format(language=request.language.value)
        
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "list": "random",
            "rnlimit": request.limit,
            "rnnamespace": "0",  # Main namespace only
        }
        
        # Make request
        async with self.http_client.get(base_url, params=params) as response:
            if response.status != 200:
                raise Exception(f"Wikipedia API error: {response.status}")
            
            data = await response.json()
            
            random_pages = data.get("query", {}).get("random", [])
            
            # Get basic info for each page
            articles = []
            for page in random_pages:
                article_request = WikipediaRequest(
                    query=page.get("title"),
                    language=request.language,
                    content_type=ContentType.SUMMARY,
                )
                
                try:
                    article_data = await self._fetch_summary(article_request)
                    articles.append(article_data["data"])
                except Exception as e:
                    logger.warning(f"Failed to fetch random article {page.get('title')}: {str(e)}")
            
            return {
                "data": {
                    "articles": articles,
                    "count": len(articles),
                }
            }
    
    async def _fetch_featured(self, request: WikipediaRequest) -> Dict[str, Any]:
        """Get featured Wikipedia articles."""
        # This would use Wikipedia's "Featured articles" category
        # Simplified implementation
        base_url = self.endpoints["api"]["base"].format(language=request.language.value)
        
        # Get articles from "Featured articles" category
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "list": "categorymembers",
            "cmtitle": "Category:Featured articles",
            "cmlimit": request.limit,
            "cmtype": "page",
        }
        
        async with self.http_client.get(base_url, params=params) as response:
            if response.status != 200:
                # Try different category name for different languages
                params["cmtitle"] = "Category:FA-Class articles"
                async with self.http_client.get(base_url, params=params) as response2:
                    if response2.status != 200:
                        raise Exception("Could not fetch featured articles")
                    data = await response2.json()
            else:
                data = await response.json()
            
            category_members = data.get("query", {}).get("categorymembers", [])
            
            # Get basic info for featured articles
            articles = []
            for member in category_members[:min(10, len(category_members))]:  # Limit to 10
                article_request = WikipediaRequest(
                    query=member.get("title"),
                    language=request.language,
                    content_type=ContentType.SUMMARY,
                )
                
                try:
                    article_data = await self._fetch_summary(article_request)
                    article_data["featured"] = True
                    articles.append(article_data["data"])
                except Exception as e:
                    logger.warning(f"Failed to fetch featured article {member.get('title')}: {str(e)}")
            
            return {
                "data": {
                    "articles": articles,
                    "count": len(articles),
                }
            }
    
    async def _fetch_trending(self, request: WikipediaRequest) -> Dict[str, Any]:
        """Get trending Wikipedia articles (most viewed)."""
        # Use pageviews API
        base_url = self.endpoints["pageviews"]["base"]
        
        # Get top viewed articles for yesterday
        yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y/%m/%d")
        
        url = f"{base_url}/top/{request.language.value}.wikipedia/all-access/{yesterday}"
        
        async with self.http_client.get(url) as response:
            if response.status != 200:
                raise Exception(f"Pageviews API error: {response.status}")
            
            data = await response.json()
            
            items = data.get("items", [])
            if not items:
                return {"data": {"trending": [], "count": 0}}
            
            top_articles = items[0].get("articles", [])[:request.limit]
            
            # Get article details
            trending_articles = []
            for article in top_articles:
                article_title = article.get("article")
                
                # Skip non-articles (Main_Page, Special:, etc.)
                if any(x in article_title for x in ["Main_Page", "Special:", "Portal:", "Wikipedia:"]):
                    continue
                
                article_request = WikipediaRequest(
                    query=unquote(article_title.replace("_", " ")),
                    language=request.language,
                    content_type=ContentType.SUMMARY,
                )
                
                try:
                    article_data = await self._fetch_summary(article_request)
                    article_data["views"] = article.get("views")
                    article_data["rank"] = article.get("rank")
                    trending_articles.append(article_data["data"])
                except Exception as e:
                    logger.debug(f"Failed to fetch trending article {article_title}: {str(e)}")
            
            return {
                "data": {
                    "trending": trending_articles,
                    "date": yesterday,
                    "count": len(trending_articles),
                }
            }
    
    # ============ Utility Methods ============
    
    def _parse_page_data(self, page_data: Dict[str, Any], language: WikipediaLanguage) -> WikipediaArticle:
        """Parse Wikipedia API page data into article object."""
        return WikipediaArticle(
            title=page_data.get("title"),
            page_id=page_data.get("pageid"),
            language=language,
            url=page_data.get("fullurl", f"https://{language.value}.wikipedia.org/wiki/{quote(page_data.get('title', ''))}"),
            wikidata_id=page_data.get("pageprops", {}).get("wikibase_item"),
            summary=page_data.get("extract"),
            extract=page_data.get("extract"),
            coordinates=page_data.get("coordinates", [{}])[0] if page_data.get("coordinates") else None,
            thumbnail=page_data.get("thumbnail"),
            last_modified=page_data.get("touched"),
            revision_id=page_data.get("lastrevid"),
            article_length=len(page_data.get("extract", "")),
        )
    
    async def get_pageviews(self, article: str, language: WikipediaLanguage, days: int = 30) -> Optional[WikipediaPageview]:
        """Get pageview statistics for an article."""
        base_url = self.endpoints["pageviews"]["base"]
        
        # Calculate date range
        end_date = datetime.utcnow() - timedelta(days=1)  # Yesterday (most recent complete day)
        start_date = end_date - timedelta(days=days)
        
        end_str = end_date.strftime("%Y%m%d")
        start_str = start_date.strftime("%Y%m%d")
        
        url = f"{base_url}/per-article/{language.value}.wikipedia/all-access/all-agents/{quote(article)}/daily/{start_str}/{end_str}"
        
        try:
            async with self.http_client.get(url) as response:
                if response.status != 200:
                    return None
                
                data = await response.json()
                items = data.get("items", [])
                
                if not items:
                    return None
                
                # Process pageviews
                views_by_date = {}
                total_views = 0
                peak_views = 0
                peak_date = None
                
                for item in items:
                    date = item.get("timestamp", "")[:8]  # YYYYMMDD
                    views = item.get("views", 0)
                    
                    views_by_date[date] = views
                    total_views += views
                    
                    if views > peak_views:
                        peak_views = views
                        peak_date = date
                
                average_views = total_views / len(items) if items else 0
                
                # Calculate trend (last 7 days vs previous 7 days)
                if len(items) >= 14:
                    recent = sum(item.get("views", 0) for item in items[-7:])
                    previous = sum(item.get("views", 0) for item in items[-14:-7])
                    
                    if previous > 0:
                        trend_percent = ((recent - previous) / previous) * 100
                        if trend_percent > 10:
                            trend = "increasing"
                        elif trend_percent < -10:
                            trend = "decreasing"
                        else:
                            trend = "stable"
                    else:
                        trend = "stable"
                else:
                    trend = None
                
                return WikipediaPageview(
                    article=article,
                    language=language,
                    views=views_by_date,
                    total_views=total_views,
                    average_views=average_views,
                    peak_views=peak_views,
                    peak_date=peak_date,
                    trend=trend,
                )
                
        except Exception as e:
            logger.warning(f"Failed to fetch pageviews for {article}: {str(e)}")
            return None
    
    async def get_article_by_wikidata(self, wikidata_id: str, language: WikipediaLanguage = WikipediaLanguage.EN) -> Optional[WikipediaArticle]:
        """Get Wikipedia article from Wikidata ID."""
        # First get Wikidata entity to find Wikipedia link
        request = WikipediaRequest(
            query=wikidata_id,
            language=language,
            content_type=ContentType.WIKIDATA,
            wikidata_id=wikidata_id,
        )
        
        response = await self.fetch(request)
        if not response.success:
            return None
        
        entity_data = response.data
        sitelink = entity_data.get("sitelinks", {}).get(f"{language.value}wiki")
        
        if not sitelink:
            return None
        
        article_title = sitelink.get("title")
        if not article_title:
            return None
        
        # Fetch the article
        article_request = WikipediaRequest(
            query=article_title,
            language=language,
            content_type=ContentType.ARTICLE,
        )
        
        article_response = await self.fetch(article_request)
        if not article_response.success:
            return None
        
        article_data = article_response.data
        return WikipediaArticle(**article_data)
    
    async def get_multilingual_articles(self, title: str, source_language: WikipediaLanguage) -> Dict[str, WikipediaArticle]:
        """Get same article in multiple languages."""
        # Get Wikidata ID first
        wikidata_id = await self._get_wikidata_id(title, source_language)
        if not wikidata_id:
            return {}
        
        # Get Wikidata entity
        wikidata_request = WikipediaRequest(
            query=wikidata_id,
            language=source_language,
            content_type=ContentType.WIKIDATA,
            wikidata_id=wikidata_id,
        )
        
        wikidata_response = await self.fetch(wikidata_request)
        if not wikidata_response.success:
            return {}
        
        entity_data = wikidata_response.data
        sitelinks = entity_data.get("sitelinks", {})
        
        # Fetch articles for each language
        articles = {}
        for sitelink_key, sitelink_data in sitelinks.items():
            # Extract language from sitelink key (e.g., "enwiki" -> "en")
            match = re.match(r"^(\w{2,3})wiki$", sitelink_key)
            if match:
                language_code = match.group(1)
                try:
                    language = WikipediaLanguage(language_code)
                    article_title = sitelink_data.get("title")
                    
                    # Fetch article summary
                    article_request = WikipediaRequest(
                        query=article_title,
                        language=language,
                        content_type=ContentType.SUMMARY,
                    )
                    
                    article_response = await self.fetch(article_request)
                    if article_response.success:
                        articles[language_code] = WikipediaArticle(**article_response.data)
                except ValueError:
                    # Unknown language code, skip
                    continue
        
        return articles
    
    async def get_related_articles(self, title: str, language: WikipediaLanguage, limit: int = 10) -> List[WikipediaArticle]:
        """Get articles related to a given article (via categories)."""
        # Get article categories
        categories_request = WikipediaRequest(
            query=title,
            language=language,
            content_type=ContentType.CATEGORIES,
            limit=20,
        )
        
        categories_response = await self.fetch(categories_request)
        if not categories_response.success:
            return []
        
        categories = categories_response.data.get("categories", [])
        
        # Get articles from the first category
        if not categories:
            return []
        
        first_category = categories[0]
        
        # Get category members
        base_url = self.endpoints["api"]["base"].format(language=language.value)
        
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "list": "categorymembers",
            "cmtitle": f"Category:{first_category}",
            "cmlimit": limit,
            "cmtype": "page",
            "cmnamespace": "0",
        }
        
        async with self.http_client.get(base_url, params=params) as response:
            if response.status != 200:
                return []
            
            data = await response.json()
            members = data.get("query", {}).get("categorymembers", [])
            
            # Get summary for each member (excluding the original article)
            related_articles = []
            for member in members:
                if member.get("title") == title:
                    continue
                
                article_request = WikipediaRequest(
                    query=member.get("title"),
                    language=language,
                    content_type=ContentType.SUMMARY,
                )
                
                try:
                    article_response = await self.fetch(article_request)
                    if article_response.success:
                        related_articles.append(WikipediaArticle(**article_response.data))
                    
                    if len(related_articles) >= limit:
                        break
                except Exception as e:
                    logger.debug(f"Failed to fetch related article {member.get('title')}: {str(e)}")
            
            return related_articles
    
    async def verify_citation(self, url: str) -> Dict[str, Any]:
        """Verify if a URL is cited in Wikipedia."""
        # This would search Wikipedia for the URL
        # Simplified implementation
        base_url = self.endpoints["api"]["base"].format(language="en")
        
        # Search for URL in Wikipedia
        params = {
            "action": "query",
            "format": "json",
            "list": "exturlusage",
            "euquery": url,
            "eulimit": 5,
            "eunamespace": "0",
        }
        
        async with self.http_client.get(base_url, params=params) as response:
            if response.status != 200:
                return {"verified": False, "error": "API error"}
            
            data = await response.json()
            usages = data.get("query", {}).get("exturlusage", [])
            
            if usages:
                articles = [{"title": u.get("title"), "page_id": u.get("pageid")} for u in usages]
                return {
                    "verified": True,
                    "citation_count": len(usages),
                    "articles": articles,
                    "url": url,
                }
            else:
                return {"verified": False, "citation_count": 0, "articles": []}
    
    async def clear_cache(self, pattern: str = "wiki:*") -> int:
        """Clear Wikipedia cache."""
        keys = await self.redis.keys(pattern)
        if keys:
            deleted = await self.redis.delete(*keys)
            logger.info(f"Cleared {deleted} Wikipedia cache keys")
            return deleted
        return 0


# ============ Factory Function ============

_wiki_client: Optional[WikipediaClient] = None

async def get_wikipedia_client() -> WikipediaClient:
    """
    Get or create a Wikipedia client singleton.
    
    Returns:
        WikipediaClient instance
    """
    global _wiki_client
    
    if _wiki_client is None:
        _wiki_client = WikipediaClient()
        await _wiki_client.initialize()
    
    return _wiki_client


# ============ Utility Functions ============

async def get_wikipedia_summary(topic: str, language: str = "en", extract_length: int = 500) -> Optional[Dict[str, Any]]:
    """
    Get Wikipedia summary for a topic.
    
    Args:
        topic: Topic to look up
        language: Language code
        extract_length: Length of extract
        
    Returns:
        Summary data or None
    """
    try:
        client = await get_wikipedia_client()
        
        request = WikipediaRequest(
            query=topic,
            language=WikipediaLanguage(language),
            content_type=ContentType.SUMMARY,
            extract_length=extract_length,
        )
        
        response = await client.fetch(request)
        
        if response.success:
            return response.data
        return None
        
    except Exception as e:
        logger.error(f"Failed to get Wikipedia summary: {str(e)}")
        return None


async def search_wikipedia(query: str, language: str = "en", limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search Wikipedia.
    
    Args:
        query: Search query
        language: Language code
        limit: Maximum results
        
    Returns:
        List of search results
    """
    try:
        client = await get_wikipedia_client()
        
        request = WikipediaRequest(
            query=query,
            language=WikipediaLanguage(language),
            content_type=ContentType.SEARCH,
            limit=limit,
        )
        
        response = await client.fetch(request)
        
        if response.success:
            return response.data.get("results", [])
        return []
        
    except Exception as e:
        logger.error(f"Failed to search Wikipedia: {str(e)}")
        return []


async def get_trending_topics(language: str = "en", limit: int = 20) -> List[Dict[str, Any]]:
    """
    Get trending Wikipedia topics.
    
    Args:
        language: Language code
        limit: Maximum results
        
    Returns:
        List of trending topics
    """
    try:
        client = await get_wikipedia_client()
        
        request = WikipediaRequest(
            query="trending",
            language=WikipediaLanguage(language),
            content_type=ContentType.TRENDING,
            limit=limit,
        )
        
        response = await client.fetch(request)
        
        if response.success:
            return response.data.get("trending", [])
        return []
        
    except Exception as e:
        logger.error(f"Failed to get trending topics: {str(e)}")
        return []


async def enrich_with_wikipedia(data: Dict[str, Any], language: str = "en") -> Dict[str, Any]:
    """
    Enrich data with Wikipedia information.
    
    Args:
        data: Data to enrich (should contain 'name' or 'title')
        language: Language code
        
    Returns:
        Enriched data
    """
    enriched = data.copy()
    
    # Determine what to look up
    lookup_terms = []
    
    if "name" in data:
        lookup_terms.append(data["name"])
    if "title" in data:
        lookup_terms.append(data["title"])
    if "location" in data:
        lookup_terms.append(data["location"])
    
    if not lookup_terms:
        return enriched
    
    client = await get_wikipedia_client()
    
    for term in lookup_terms:
        try:
            request = WikipediaRequest(
                query=term,
                language=WikipediaLanguage(language),
                content_type=ContentType.SUMMARY,
                extract_length=200,
            )
            
            response = await client.fetch(request)
            
            if response.success:
                wiki_data = response.data
                
                # Add Wikipedia data
                enriched["wikipedia"] = {
                    "summary": wiki_data.get("summary"),
                    "url": wiki_data.get("url"),
                    "page_id": wiki_data.get("page_id"),
                    "wikidata_id": wiki_data.get("wikidata_id"),
                }
                
                # Add coordinates if available
                if wiki_data.get("coordinates"):
                    enriched["coordinates"] = wiki_data["coordinates"]
                
                # Add thumbnail if available
                if wiki_data.get("thumbnail"):
                    enriched["thumbnail"] = wiki_data["thumbnail"]
                
                break  # Stop at first successful lookup
                
        except Exception as e:
            logger.debug(f"Failed to enrich {term} with Wikipedia: {str(e)}")
            continue
    
    return enriched


async def verify_facts_with_wikipedia(facts: List[str], language: str = "en") -> List[Dict[str, Any]]:
    """
    Verify facts against Wikipedia.
    
    Args:
        facts: List of facts to verify
        language: Language code
        
    Returns:
        List of verification results
    """
    client = await get_wikipedia_client()
    results = []
    
    for fact in facts:
        try:
            # Search for the fact
            search_request = WikipediaRequest(
                query=fact,
                language=WikipediaLanguage(language),
                content_type=ContentType.SEARCH,
                limit=3,
            )
            
            search_response = await client.fetch(search_request)
            
            if not search_response.success or not search_response.data.get("results"):
                results.append({
                    "fact": fact,
                    "verified": False,
                    "confidence": 0.0,
                    "sources": [],
                })
                continue
            
            # Check search results
            search_results = search_response.data["results"]
            confidence = min(1.0, len(search_results) / 10.0)  # Simple confidence score
            
            results.append({
                "fact": fact,
                "verified": confidence > 0.3,
                "confidence": confidence,
                "sources": [{"title": r["title"], "url": r["url"]} for r in search_results[:3]],
            })
            
        except Exception as e:
            logger.error(f"Failed to verify fact '{fact}': {str(e)}")
            results.append({
                "fact": fact,
                "verified": False,
                "confidence": 0.0,
                "error": str(e),
            })
    
    return results