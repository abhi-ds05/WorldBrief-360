# backend/app/integrations/news_api_client.py
"""
News API integration for WorldBrief 360.
Supports multiple news sources: NewsAPI.org, GNews, Mediastack, and more.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
from urllib.parse import urlencode

import aiohttp
from pydantic import BaseModel, Field, validator, HttpUrl
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from cachetools import TTLCache, LRUCache
import dateutil.parser

from app.core.config import settings
from app.core.logging_config import logger
from app.services.utils.http_client import AsyncHTTPClient
from app.schemas.request.news import NewsQueryParams


class NewsSource(Enum):
    """Supported news sources."""
    NEWSAPI = "newsapi"
    GNEWS = "gnews"
    MEDIASTACK = "mediastack"
    NEWS_DATA = "newsdata"
    CURRENT_EVENTS = "currents"  # Currents API
    CONTEXTUAL = "contextual"  # Contextual Web Search
    WEB_SEARCH = "websearch"  # Generic web search


class NewsCategory(Enum):
    """News categories."""
    BUSINESS = "business"
    ENTERTAINMENT = "entertainment"
    GENERAL = "general"
    HEALTH = "health"
    SCIENCE = "science"
    SPORTS = "sports"
    TECHNOLOGY = "technology"
    POLITICS = "politics"
    ENVIRONMENT = "environment"
    WORLD = "world"
    NATIONAL = "national"
    LOCAL = "local"


class NewsLanguage(Enum):
    """News languages."""
    AR = "ar"  # Arabic
    DE = "de"  # German
    EN = "en"  # English
    ES = "es"  # Spanish
    FR = "fr"  # French
    HE = "he"  # Hebrew
    IT = "it"  # Italian
    NL = "nl"  # Dutch
    NO = "no"  # Norwegian
    PT = "pt"  # Portuguese
    RU = "ru"  # Russian
    SE = "se"  # Swedish
    UD = "ud"  # Urdu
    ZH = "zh"  # Chinese


class NewsCountry(Enum):
    """News countries (ISO 3166-1 alpha-2)."""
    AE = "ae"  # UAE
    AR = "ar"  # Argentina
    AT = "at"  # Austria
    AU = "au"  # Australia
    BE = "be"  # Belgium
    BG = "bg"  # Bulgaria
    BR = "br"  # Brazil
    CA = "ca"  # Canada
    CH = "ch"  # Switzerland
    CN = "cn"  # China
    CO = "co"  # Colombia
    CU = "cu"  # Cuba
    CZ = "cz"  # Czech Republic
    DE = "de"  # Germany
    EG = "eg"  # Egypt
    FR = "fr"  # France
    GB = "gb"  # United Kingdom
    GR = "gr"  # Greece
    HK = "hk"  # Hong Kong
    HU = "hu"  # Hungary
    ID = "id"  # Indonesia
    IE = "ie"  # Ireland
    IL = "il"  # Israel
    IN = "in"  # India
    IT = "it"  # Italy
    JP = "jp"  # Japan
    KR = "kr"  # South Korea
    LT = "lt"  # Lithuania
    LV = "lv"  # Latvia
    MA = "ma"  # Morocco
    MX = "mx"  # Mexico
    MY = "my"  # Malaysia
    NG = "ng"  # Nigeria
    NL = "nl"  # Netherlands
    NO = "no"  # Norway
    NZ = "nz"  # New Zealand
    PH = "ph"  # Philippines
    PL = "pl"  # Poland
    PT = "pt"  # Portugal
    RO = "ro"  # Romania
    RS = "rs"  # Serbia
    RU = "ru"  # Russia
    SA = "sa"  # Saudi Arabia
    SE = "se"  # Sweden
    SG = "sg"  # Singapore
    SI = "si"  # Slovenia
    SK = "sk"  # Slovakia
    TH = "th"  # Thailand
    TR = "tr"  # Turkey
    TW = "tw"  # Taiwan
    UA = "ua"  # Ukraine
    US = "us"  # United States
    VE = "ve"  # Venezuela
    ZA = "za"  # South Africa


class NewsArticle(BaseModel):
    """Standardized news article representation."""
    id: str
    title: str
    description: Optional[str] = None
    content: Optional[str] = None
    summary: Optional[str] = None
    url: HttpUrl
    source_id: Optional[str] = None
    source_name: str
    source_url: Optional[HttpUrl] = None
    author: Optional[str] = None
    published_at: datetime
    updated_at: Optional[datetime] = None
    category: Optional[NewsCategory] = None
    language: NewsLanguage = NewsLanguage.EN
    country: Optional[NewsCountry] = None
    image_url: Optional[HttpUrl] = None
    video_url: Optional[HttpUrl] = None
    keywords: List[str] = Field(default_factory=list)
    sentiment_score: Optional[float] = None  # -1 to 1
    relevance_score: Optional[float] = None  # 0 to 1
    fact_check_score: Optional[float] = None  # 0 to 1
    read_time_minutes: Optional[int] = None
    word_count: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True
    
    @validator('published_at', pre=True)
    def parse_published_at(cls, v):
        if isinstance(v, str):
            try:
                return dateutil.parser.parse(v)
            except:
                return datetime.now()
        return v
    
    @validator('sentiment_score')
    def validate_sentiment_score(cls, v):
        if v is not None and (v < -1 or v > 1):
            raise ValueError('Sentiment score must be between -1 and 1')
        return v
    
    @validator('relevance_score', 'fact_check_score')
    def validate_score(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError('Score must be between 0 and 1')
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'content': self.content,
            'summary': self.summary,
            'url': str(self.url),
            'source_id': self.source_id,
            'source_name': self.source_name,
            'source_url': str(self.source_url) if self.source_url else None,
            'author': self.author,
            'published_at': self.published_at.isoformat(),
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'category': self.category.value if self.category else None,
            'language': self.language.value,
            'country': self.country.value if self.country else None,
            'image_url': str(self.image_url) if self.image_url else None,
            'video_url': str(self.video_url) if self.video_url else None,
            'keywords': self.keywords,
            'sentiment_score': self.sentiment_score,
            'relevance_score': self.relevance_score,
            'fact_check_score': self.fact_check_score,
            'read_time_minutes': self.read_time_minutes,
            'word_count': self.word_count,
            'metadata': self.metadata,
        }


class NewsResponse(BaseModel):
    """News API response."""
    articles: List[NewsArticle]
    total_results: int
    page: int
    page_size: int
    has_more: bool
    query_params: Dict[str, Any] = Field(default_factory=dict)
    source: NewsSource
    requested_at: datetime = Field(default_factory=datetime.now)
    processing_time_ms: Optional[float] = None


class BaseNewsProvider:
    """Base class for news providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
        self.http_client = AsyncHTTPClient()
        
    async def get_top_headlines(
        self,
        query: Optional[str] = None,
        category: Optional[NewsCategory] = None,
        country: Optional[NewsCountry] = None,
        language: Optional[NewsLanguage] = None,
        page: int = 1,
        page_size: int = 20,
        **kwargs
    ) -> NewsResponse:
        """Get top headlines."""
        raise NotImplementedError("Subclasses must implement get_top_headlines")
    
    async def search_articles(
        self,
        query: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        language: Optional[NewsLanguage] = None,
        sort_by: str = "relevancy",
        page: int = 1,
        page_size: int = 20,
        **kwargs
    ) -> NewsResponse:
        """Search for articles."""
        raise NotImplementedError("Subclasses must implement search_articles")
    
    async def get_sources(
        self,
        category: Optional[NewsCategory] = None,
        country: Optional[NewsCountry] = None,
        language: Optional[NewsLanguage] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Get available news sources."""
        raise NotImplementedError("Subclasses must implement get_sources")
    
    async def validate_connection(self) -> bool:
        """Validate connection to news provider."""
        raise NotImplementedError("Subclasses must implement validate_connection")
    
    def _generate_article_id(self, article_data: Dict[str, Any]) -> str:
        """Generate unique article ID."""
        unique_str = f"{article_data.get('url', '')}_{article_data.get('published_at', '')}"
        return hashlib.md5(unique_str.encode()).hexdigest()


class NewsAPIProvider(BaseNewsProvider):
    """NewsAPI.org provider."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {
                "api_key": settings.NEWSAPI_API_KEY,
                "base_url": "https://newsapi.org/v2",
                "user_agent": "WorldBrief360/1.0",
                "timeout": 30,
                "max_retries": 3,
            }
        super().__init__(config)
        
        self.base_url = self.config["base_url"]
        self.api_key = self.config["api_key"]
        self.headers = {
            "User-Agent": self.config["user_agent"],
        }
        
        if not self.api_key:
            logger.warning("NewsAPI.org API key not provided")
    
    def is_available(self) -> bool:
        """Check if provider is properly configured."""
        return bool(self.api_key)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError,))
    )
    async def get_top_headlines(
        self,
        query: Optional[str] = None,
        category: Optional[NewsCategory] = None,
        country: Optional[NewsCountry] = None,
        language: Optional[NewsLanguage] = None,
        page: int = 1,
        page_size: int = 20,
        **kwargs
    ) -> NewsResponse:
        """Get top headlines from NewsAPI.org."""
        if not self.is_available():
            raise ValueError("NewsAPI.org provider is not configured")
        
        start_time = datetime.now()
        
        try:
            params = {
                "apiKey": self.api_key,
                "page": page,
                "pageSize": page_size,
            }
            
            if query:
                params["q"] = query
            if category:
                params["category"] = category.value
            if country:
                params["country"] = country.value
            if language:
                params["language"] = language.value
            
            # Add any additional parameters
            params.update(kwargs)
            
            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}
            
            url = f"{self.base_url}/top-headlines"
            
            async with self.http_client.session.get(
                url, params=params, headers=self.headers
            ) as response:
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get("status") == "ok":
                        articles = self._parse_articles(data.get("articles", []))
                        
                        return NewsResponse(
                            articles=articles,
                            total_results=data.get("totalResults", 0),
                            page=page,
                            page_size=page_size,
                            has_more=len(articles) == page_size,
                            query_params=params,
                            source=NewsSource.NEWSAPI,
                            processing_time_ms=processing_time
                        )
                    else:
                        logger.error(f"NewsAPI error: {data.get('code')} - {data.get('message')}")
                        return NewsResponse(
                            articles=[],
                            total_results=0,
                            page=page,
                            page_size=page_size,
                            has_more=False,
                            query_params=params,
                            source=NewsSource.NEWSAPI,
                            processing_time_ms=processing_time
                        )
                else:
                    error_text = await response.text()
                    logger.error(f"NewsAPI request failed: {response.status} - {error_text}")
                    return NewsResponse(
                        articles=[],
                        total_results=0,
                        page=page,
                        page_size=page_size,
                        has_more=False,
                        query_params=params,
                        source=NewsSource.NEWSAPI,
                        processing_time_ms=processing_time
                    )
                    
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Error getting top headlines from NewsAPI: {str(e)}")
            return NewsResponse(
                articles=[],
                total_results=0,
                page=page,
                page_size=page_size,
                has_more=False,
                query_params={},
                source=NewsSource.NEWSAPI,
                processing_time_ms=processing_time
            )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def search_articles(
        self,
        query: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        language: Optional[NewsLanguage] = None,
        sort_by: str = "relevancy",
        page: int = 1,
        page_size: int = 20,
        **kwargs
    ) -> NewsResponse:
        """Search for articles on NewsAPI.org."""
        if not self.is_available():
            raise ValueError("NewsAPI.org provider is not configured")
        
        start_time = datetime.now()
        
        try:
            params = {
                "apiKey": self.api_key,
                "q": query,
                "page": page,
                "pageSize": page_size,
                "sortBy": sort_by,
            }
            
            if from_date:
                params["from"] = from_date.strftime("%Y-%m-%d")
            if to_date:
                params["to"] = to_date.strftime("%Y-%m-%d")
            if language:
                params["language"] = language.value
            
            # Add any additional parameters
            params.update(kwargs)
            
            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}
            
            url = f"{self.base_url}/everything"
            
            async with self.http_client.session.get(
                url, params=params, headers=self.headers
            ) as response:
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get("status") == "ok":
                        articles = self._parse_articles(data.get("articles", []))
                        
                        return NewsResponse(
                            articles=articles,
                            total_results=data.get("totalResults", 0),
                            page=page,
                            page_size=page_size,
                            has_more=len(articles) == page_size,
                            query_params=params,
                            source=NewsSource.NEWSAPI,
                            processing_time_ms=processing_time
                        )
                    else:
                        logger.error(f"NewsAPI search error: {data.get('code')} - {data.get('message')}")
                        return NewsResponse(
                            articles=[],
                            total_results=0,
                            page=page,
                            page_size=page_size,
                            has_more=False,
                            query_params=params,
                            source=NewsSource.NEWSAPI,
                            processing_time_ms=processing_time
                        )
                else:
                    error_text = await response.text()
                    logger.error(f"NewsAPI search failed: {response.status} - {error_text}")
                    return NewsResponse(
                        articles=[],
                        total_results=0,
                        page=page,
                        page_size=page_size,
                        has_more=False,
                        query_params=params,
                        source=NewsSource.NEWSAPI,
                        processing_time_ms=processing_time
                    )
                    
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Error searching articles on NewsAPI: {str(e)}")
            return NewsResponse(
                articles=[],
                total_results=0,
                page=page,
                page_size=page_size,
                has_more=False,
                query_params={},
                source=NewsSource.NEWSAPI,
                processing_time_ms=processing_time
            )
    
    def _parse_articles(self, articles_data: List[Dict[str, Any]]) -> List[NewsArticle]:
        """Parse raw article data from NewsAPI."""
        articles = []
        
        for article_data in articles_data:
            try:
                # Generate unique ID
                article_id = self._generate_article_id(article_data)
                
                # Parse source
                source_data = article_data.get("source", {})
                source_id = source_data.get("id")
                source_name = source_data.get("name", "Unknown")
                
                # Parse dates
                published_at_str = article_data.get("publishedAt")
                published_at = dateutil.parser.parse(published_at_str) if published_at_str else datetime.now()
                
                # Parse URL
                url = article_data.get("url", "")
                if not url.startswith("http"):
                    continue  # Skip articles without valid URLs
                
                # Estimate read time
                content = article_data.get("content", "") or article_data.get("description", "")
                word_count = len(content.split())
                read_time = max(1, word_count // 200)  # 200 words per minute
                
                # Create article
                article = NewsArticle(
                    id=article_id,
                    title=article_data.get("title", "").strip() or "No title",
                    description=article_data.get("description", "").strip(),
                    content=article_data.get("content", "").strip(),
                    url=url,
                    source_id=source_id,
                    source_name=source_name,
                    author=article_data.get("author"),
                    published_at=published_at,
                    image_url=article_data.get("urlToImage"),
                    language=NewsLanguage.EN,  # NewsAPI doesn't provide language
                    word_count=word_count,
                    read_time_minutes=read_time,
                    metadata={
                        "raw_source": "newsapi",
                        "source_id": source_id,
                    }
                )
                
                articles.append(article)
                
            except Exception as e:
                logger.error(f"Error parsing article: {str(e)}")
                continue
        
        return articles
    
    async def get_sources(
        self,
        category: Optional[NewsCategory] = None,
        country: Optional[NewsCountry] = None,
        language: Optional[NewsLanguage] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Get available news sources from NewsAPI.org."""
        if not self.is_available():
            raise ValueError("NewsAPI.org provider is not configured")
        
        try:
            params = {
                "apiKey": self.api_key,
            }
            
            if category:
                params["category"] = category.value
            if country:
                params["country"] = country.value
            if language:
                params["language"] = language.value
            
            params.update(kwargs)
            
            url = f"{self.base_url}/sources"
            
            async with self.http_client.session.get(
                url, params=params, headers=self.headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get("status") == "ok":
                        sources = data.get("sources", [])
                        
                        formatted_sources = []
                        for source in sources:
                            formatted_sources.append({
                                "id": source.get("id"),
                                "name": source.get("name"),
                                "description": source.get("description"),
                                "url": source.get("url"),
                                "category": source.get("category"),
                                "language": source.get("language"),
                                "country": source.get("country"),
                            })
                        
                        return formatted_sources
                    else:
                        logger.error(f"NewsAPI sources error: {data.get('code')} - {data.get('message')}")
                        return []
                else:
                    error_text = await response.text()
                    logger.error(f"NewsAPI sources failed: {response.status} - {error_text}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting sources from NewsAPI: {str(e)}")
            return []
    
    async def validate_connection(self) -> bool:
        """Validate connection to NewsAPI.org."""
        if not self.is_available():
            return False
        
        try:
            # Test with a simple top headlines request
            params = {
                "apiKey": self.api_key,
                "country": "us",
                "pageSize": 1,
            }
            
            url = f"{self.base_url}/top-headlines"
            
            async with self.http_client.session.get(
                url, params=params, headers=self.headers, timeout=10
            ) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"NewsAPI connection validation failed: {str(e)}")
            return False


class GNewsProvider(BaseNewsProvider):
    """GNews provider (free alternative)."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {
                "api_key": settings.GNEWS_API_KEY,
                "base_url": "https://gnews.io/api/v4",
                "timeout": 30,
                "max_retries": 3,
            }
        super().__init__(config)
        
        self.base_url = self.config["base_url"]
        self.api_key = self.config["api_key"]
        
        if not self.api_key:
            logger.warning("GNews API key not provided")
    
    def is_available(self) -> bool:
        """Check if provider is properly configured."""
        return bool(self.api_key)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def get_top_headlines(
        self,
        query: Optional[str] = None,
        category: Optional[NewsCategory] = None,
        country: Optional[NewsCountry] = None,
        language: Optional[NewsLanguage] = None,
        page: int = 1,
        page_size: int = 20,
        **kwargs
    ) -> NewsResponse:
        """Get top headlines from GNews."""
        if not self.is_available():
            raise ValueError("GNews provider is not configured")
        
        start_time = datetime.now()
        
        try:
            params = {
                "token": self.api_key,
                "max": page_size,
            }
            
            if query:
                params["q"] = query
            if category:
                # Map our categories to GNews categories
                category_map = {
                    NewsCategory.BUSINESS: "business",
                    NewsCategory.ENTERTAINMENT: "entertainment",
                    NewsCategory.HEALTH: "health",
                    NewsCategory.SCIENCE: "science",
                    NewsCategory.SPORTS: "sports",
                    NewsCategory.TECHNOLOGY: "technology",
                }
                gnews_category = category_map.get(category)
                if gnews_category:
                    params["topic"] = gnews_category
            
            if country:
                params["country"] = country.value
            
            if language:
                params["lang"] = language.value
            
            # GNews uses different pagination
            if page > 1:
                params["page"] = page
            
            # Add any additional parameters
            params.update(kwargs)
            
            url = f"{self.base_url}/top-headlines"
            
            async with self.http_client.session.get(url, params=params) as response:
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                
                if response.status == 200:
                    data = await response.json()
                    
                    articles = self._parse_articles(data.get("articles", []))
                    total_results = data.get("totalArticles", len(articles))
                    
                    return NewsResponse(
                        articles=articles,
                        total_results=total_results,
                        page=page,
                        page_size=page_size,
                        has_more=len(articles) == page_size,
                        query_params=params,
                        source=NewsSource.GNEWS,
                        processing_time_ms=processing_time
                    )
                else:
                    error_text = await response.text()
                    logger.error(f"GNews request failed: {response.status} - {error_text}")
                    return NewsResponse(
                        articles=[],
                        total_results=0,
                        page=page,
                        page_size=page_size,
                        has_more=False,
                        query_params=params,
                        source=NewsSource.GNEWS,
                        processing_time_ms=processing_time
                    )
                    
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Error getting top headlines from GNews: {str(e)}")
            return NewsResponse(
                articles=[],
                total_results=0,
                page=page,
                page_size=page_size,
                has_more=False,
                query_params={},
                source=NewsSource.GNEWS,
                processing_time_ms=processing_time
            )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def search_articles(
        self,
        query: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        language: Optional[NewsLanguage] = None,
        sort_by: str = "relevancy",
        page: int = 1,
        page_size: int = 20,
        **kwargs
    ) -> NewsResponse:
        """Search for articles on GNews."""
        if not self.is_available():
            raise ValueError("GNews provider is not configured")
        
        start_time = datetime.now()
        
        try:
            params = {
                "token": self.api_key,
                "q": query,
                "max": page_size,
            }
            
            if from_date:
                params["from"] = from_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            if to_date:
                params["to"] = to_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            if language:
                params["lang"] = language.value
            
            # Map sort_by
            sort_map = {
                "relevancy": "relevance",
                "publishedAt": "publishedAt",
                "popularity": "popularity",
            }
            params["sortby"] = sort_map.get(sort_by, "relevance")
            
            # GNews uses different pagination
            if page > 1:
                params["page"] = page
            
            # Add any additional parameters
            params.update(kwargs)
            
            url = f"{self.base_url}/search"
            
            async with self.http_client.session.get(url, params=params) as response:
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                
                if response.status == 200:
                    data = await response.json()
                    
                    articles = self._parse_articles(data.get("articles", []))
                    total_results = data.get("totalArticles", len(articles))
                    
                    return NewsResponse(
                        articles=articles,
                        total_results=total_results,
                        page=page,
                        page_size=page_size,
                        has_more=len(articles) == page_size,
                        query_params=params,
                        source=NewsSource.GNEWS,
                        processing_time_ms=processing_time
                    )
                else:
                    error_text = await response.text()
                    logger.error(f"GNews search failed: {response.status} - {error_text}")
                    return NewsResponse(
                        articles=[],
                        total_results=0,
                        page=page,
                        page_size=page_size,
                        has_more=False,
                        query_params=params,
                        source=NewsSource.GNEWS,
                        processing_time_ms=processing_time
                    )
                    
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Error searching articles on GNews: {str(e)}")
            return NewsResponse(
                articles=[],
                total_results=0,
                page=page,
                page_size=page_size,
                has_more=False,
                query_params={},
                source=NewsSource.GNEWS,
                processing_time_ms=processing_time
            )
    
    def _parse_articles(self, articles_data: List[Dict[str, Any]]) -> List[NewsArticle]:
        """Parse raw article data from GNews."""
        articles = []
        
        for article_data in articles_data:
            try:
                # Generate unique ID
                article_id = self._generate_article_id(article_data)
                
                # Parse source
                source_data = article_data.get("source", {})
                source_name = source_data.get("name", "Unknown")
                source_url = source_data.get("url")
                
                # Parse dates
                published_at_str = article_data.get("publishedAt")
                published_at = dateutil.parser.parse(published_at_str) if published_at_str else datetime.now()
                
                # Parse URL
                url = article_data.get("url", "")
                if not url.startswith("http"):
                    continue  # Skip articles without valid URLs
                
                # Parse image
                image_data = article_data.get("image", "")
                image_url = image_data if image_data.startswith("http") else None
                
                # Estimate read time
                content = article_data.get("content", "") or article_data.get("description", "")
                word_count = len(content.split())
                read_time = max(1, word_count // 200)
                
                # Parse language from metadata
                language = NewsLanguage.EN
                lang_code = article_data.get("language", "en").lower()
                try:
                    language = NewsLanguage(lang_code)
                except ValueError:
                    pass
                
                # Create article
                article = NewsArticle(
                    id=article_id,
                    title=article_data.get("title", "").strip() or "No title",
                    description=article_data.get("description", "").strip(),
                    content=article_data.get("content", "").strip(),
                    url=url,
                    source_name=source_name,
                    source_url=source_url,
                    author=None,  # GNews doesn't provide author
                    published_at=published_at,
                    image_url=image_url,
                    language=language,
                    word_count=word_count,
                    read_time_minutes=read_time,
                    metadata={
                        "raw_source": "gnews",
                        "source_url": source_url,
                    }
                )
                
                articles.append(article)
                
            except Exception as e:
                logger.error(f"Error parsing GNews article: {str(e)}")
                continue
        
        return articles
    
    async def get_sources(self, **kwargs) -> List[Dict[str, Any]]:
        """GNews doesn't provide a sources endpoint."""
        return []
    
    async def validate_connection(self) -> bool:
        """Validate connection to GNews."""
        if not self.is_available():
            return False
        
        try:
            # Test with a simple search
            params = {
                "token": self.api_key,
                "q": "test",
                "max": 1,
            }
            
            url = f"{self.base_url}/search"
            
            async with self.http_client.session.get(url, params=params, timeout=10) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"GNews connection validation failed: {str(e)}")
            return False


class NewsClient:
    """
    Main news client for WorldBrief 360.
    Supports multiple news providers with fallback logic.
    """
    
    def __init__(
        self,
        default_provider: Optional[NewsSource] = None,
        providers_config: Optional[Dict[NewsSource, Dict[str, Any]]] = None
    ):
        self.default_provider = default_provider or NewsSource.NEWSAPI
        self.providers: Dict[NewsSource, BaseNewsProvider] = {}
        self._initialize_providers(providers_config or {})
        
        # Caches
        self.headlines_cache = TTLCache(maxsize=1000, ttl=300)  # 5 minutes
        self.search_cache = TTLCache(maxsize=2000, ttl=600)  # 10 minutes
        self.sources_cache = TTLCache(maxsize=100, ttl=86400)  # 24 hours
        
        logger.info(f"News client initialized with default provider: {self.default_provider.value}")
    
    def _initialize_providers(self, providers_config: Dict[NewsSource, Dict[str, Any]]):
        """Initialize news providers."""
        provider_classes = {
            NewsSource.NEWSAPI: NewsAPIProvider,
            NewsSource.GNEWS: GNewsProvider,
        }
        
        for source, provider_class in provider_classes.items():
            try:
                config = providers_config.get(source, {})
                provider = provider_class(config)
                
                # Validate provider if it has API key requirements
                if hasattr(provider, 'is_available'):
                    if provider.is_available():
                        self.providers[source] = provider
                        logger.info(f"Initialized news provider: {source.value}")
                    else:
                        logger.warning(f"Skipping unavailable news provider: {source.value}")
                else:
                    # Provider doesn't have availability check
                    self.providers[source] = provider
                    logger.info(f"Initialized news provider: {source.value}")
                    
            except Exception as e:
                logger.error(f"Failed to initialize {source.value}: {str(e)}")
    
    def get_provider(self, source: Optional[NewsSource] = None) -> BaseNewsProvider:
        """
        Get news provider.
        
        Args:
            source: News source (uses default if None)
            
        Returns:
            News provider instance
            
        Raises:
            ValueError: If provider not available
        """
        source = source or self.default_provider
        
        if source not in self.providers:
            # Try to get any available provider
            if self.providers:
                return next(iter(self.providers.values()))
            raise ValueError(f"No news providers available. Requested: {source.value}")
        
        return self.providers[source]
    
    def get_available_providers(self) -> List[Dict[str, Any]]:
        """Get list of available news providers."""
        available = []
        
        for source, provider in self.providers.items():
            available.append({
                "source": source.value,
                "name": source.name,
                "enabled": True,
                "description": str(provider)
            })
        
        return available
    
    def _generate_cache_key(self, prefix: str, params: Dict[str, Any]) -> str:
        """Generate cache key."""
        # Sort params to ensure consistent keys
        sorted_params = json.dumps(params, sort_keys=True, default=str)
        key_hash = hashlib.md5(sorted_params.encode()).hexdigest()
        return f"{prefix}:{key_hash}"
    
    async def get_top_headlines(
        self,
        query: Optional[str] = None,
        category: Optional[NewsCategory] = None,
        country: Optional[NewsCountry] = None,
        language: Optional[NewsLanguage] = None,
        page: int = 1,
        page_size: int = 20,
        source: Optional[NewsSource] = None,
        use_cache: bool = True,
        **kwargs
    ) -> NewsResponse:
        """
        Get top headlines.
        
        Args:
            query: Search query
            category: News category
            country: Country code
            language: Language code
            page: Page number
            page_size: Page size
            source: News source to use
            use_cache: Whether to use cache
            **kwargs: Additional parameters
            
        Returns:
            News response
        """
        params = {
            "query": query,
            "category": category.value if category else None,
            "country": country.value if country else None,
            "language": language.value if language else None,
            "page": page,
            "page_size": page_size,
            **kwargs
        }
        
        cache_key = self._generate_cache_key("headlines", params)
        
        if use_cache and cache_key in self.headlines_cache:
            logger.debug(f"Cache hit for headlines: {cache_key}")
            return self.headlines_cache[cache_key]
        
        try:
            provider = self.get_provider(source)
            response = await provider.get_top_headlines(
                query=query,
                category=category,
                country=country,
                language=language,
                page=page,
                page_size=page_size,
                **kwargs
            )
            
            # Cache the response
            if use_cache:
                self.headlines_cache[cache_key] = response
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to get top headlines: {str(e)}")
            
            # Try fallback provider
            if source and source != self.default_provider:
                try:
                    logger.info(f"Trying fallback provider for headlines: {self.default_provider.value}")
                    provider = self.get_provider(self.default_provider)
                    return await provider.get_top_headlines(
                        query=query,
                        category=category,
                        country=country,
                        language=language,
                        page=page,
                        page_size=page_size,
                        **kwargs
                    )
                except Exception as fallback_error:
                    logger.error(f"Fallback provider also failed: {str(fallback_error)}")
            
            return NewsResponse(
                articles=[],
                total_results=0,
                page=page,
                page_size=page_size,
                has_more=False,
                query_params=params,
                source=source or self.default_provider,
            )
    
    async def search_articles(
        self,
        query: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        language: Optional[NewsLanguage] = None,
        sort_by: str = "relevancy",
        page: int = 1,
        page_size: int = 20,
        source: Optional[NewsSource] = None,
        use_cache: bool = True,
        **kwargs
    ) -> NewsResponse:
        """
        Search for articles.
        
        Args:
            query: Search query
            from_date: Start date
            to_date: End date
            language: Language code
            sort_by: Sort order
            page: Page number
            page_size: Page size
            source: News source to use
            use_cache: Whether to use cache
            **kwargs: Additional parameters
            
        Returns:
            News response
        """
        params = {
            "query": query,
            "from_date": from_date.isoformat() if from_date else None,
            "to_date": to_date.isoformat() if to_date else None,
            "language": language.value if language else None,
            "sort_by": sort_by,
            "page": page,
            "page_size": page_size,
            **kwargs
        }
        
        cache_key = self._generate_cache_key("search", params)
        
        if use_cache and cache_key in self.search_cache:
            logger.debug(f"Cache hit for search: {cache_key}")
            return self.search_cache[cache_key]
        
        try:
            provider = self.get_provider(source)
            response = await provider.search_articles(
                query=query,
                from_date=from_date,
                to_date=to_date,
                language=language,
                sort_by=sort_by,
                page=page,
                page_size=page_size,
                **kwargs
            )
            
            # Cache the response
            if use_cache:
                self.search_cache[cache_key] = response
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to search articles: {str(e)}")
            
            # Try fallback provider
            if source and source != self.default_provider:
                try:
                    logger.info(f"Trying fallback provider for search: {self.default_provider.value}")
                    provider = self.get_provider(self.default_provider)
                    return await provider.search_articles(
                        query=query,
                        from_date=from_date,
                        to_date=to_date,
                        language=language,
                        sort_by=sort_by,
                        page=page,
                        page_size=page_size,
                        **kwargs
                    )
                except Exception as fallback_error:
                    logger.error(f"Fallback provider also failed: {str(fallback_error)}")
            
            return NewsResponse(
                articles=[],
                total_results=0,
                page=page,
                page_size=page_size,
                has_more=False,
                query_params=params,
                source=source or self.default_provider,
            )
    
    async def get_sources(
        self,
        category: Optional[NewsCategory] = None,
        country: Optional[NewsCountry] = None,
        language: Optional[NewsLanguage] = None,
        source: Optional[NewsSource] = None,
        use_cache: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Get available news sources.
        
        Args:
            category: Filter by category
            country: Filter by country
            language: Filter by language
            source: News source to use
            use_cache: Whether to use cache
            **kwargs: Additional parameters
            
        Returns:
            List of news sources
        """
        params = {
            "category": category.value if category else None,
            "country": country.value if country else None,
            "language": language.value if language else None,
            **kwargs
        }
        
        cache_key = self._generate_cache_key("sources", params)
        
        if use_cache and cache_key in self.sources_cache:
            logger.debug(f"Cache hit for sources: {cache_key}")
            return self.sources_cache[cache_key]
        
        try:
            provider = self.get_provider(source)
            sources = await provider.get_sources(
                category=category,
                country=country,
                language=language,
                **kwargs
            )
            
            # Cache the results
            if use_cache:
                self.sources_cache[cache_key] = sources
            
            return sources
            
        except Exception as e:
            logger.error(f"Failed to get sources: {str(e)}")
            return []
    
    async def get_articles_by_topic(
        self,
        topic: str,
        days_back: int = 7,
        max_articles: int = 50,
        sources: Optional[List[NewsSource]] = None,
        **kwargs
    ) -> List[NewsArticle]:
        """
        Get articles about a specific topic from multiple sources.
        
        Args:
            topic: Topic to search for
            days_back: Number of days to look back
            max_articles: Maximum number of articles to return
            sources: List of sources to use
            **kwargs: Additional parameters
            
        Returns:
            List of articles
        """
        from_date = datetime.now() - timedelta(days=days_back)
        
        # Use specified sources or all available
        if not sources:
            sources = list(self.providers.keys())
        
        all_articles = []
        
        # Search each source
        for source in sources:
            if source not in self.providers:
                continue
            
            try:
                response = await self.search_articles(
                    query=topic,
                    from_date=from_date,
                    page_size=min(50, max_articles // len(sources)),
                    source=source,
                    use_cache=True,
                    **kwargs
                )
                
                all_articles.extend(response.articles)
                
                if len(all_articles) >= max_articles:
                    break
                    
            except Exception as e:
                logger.error(f"Failed to get articles from {source.value}: {str(e)}")
                continue
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_articles = []
        
        for article in all_articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)
            
            if len(unique_articles) >= max_articles:
                break
        
        # Sort by date (newest first)
        unique_articles.sort(key=lambda x: x.published_at, reverse=True)
        
        return unique_articles
    
    async def get_trending_topics(
        self,
        country: Optional[NewsCountry] = None,
        category: Optional[NewsCategory] = None,
        limit: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Get trending topics based on current headlines.
        
        Args:
            country: Country code
            category: News category
            limit: Maximum number of topics
            **kwargs: Additional parameters
            
        Returns:
            List of trending topics
        """
        try:
            # Get current headlines
            response = await self.get_top_headlines(
                country=country,
                category=category,
                page_size=50,
                **kwargs
            )
            
            # Extract topics from article titles
            from collections import Counter
            import re
            
            # Common stop words to exclude
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
                'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
                'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                'will', 'would', 'shall', 'should', 'may', 'might', 'must',
                'can', 'could', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
                'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its',
                'our', 'their', 'this', 'that', 'these', 'those', 'what',
                'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how',
                'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
                'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
                'don', 'should', 'now'
            }
            
            # Extract words from titles
            words = []
            for article in response.articles:
                title = article.title.lower()
                # Remove punctuation and split
                title_words = re.findall(r'\b[a-z]{3,}\b', title)
                words.extend([w for w in title_words if w not in stop_words])
            
            # Count word frequencies
            word_counts = Counter(words)
            
            # Get most common topics
            topics = []
            for word, count in word_counts.most_common(limit):
                topics.append({
                    'topic': word.title(),
                    'count': count,
                    'articles': [
                        {
                            'title': a.title,
                            'url': str(a.url),
                            'source': a.source_name
                        }
                        for a in response.articles
                        if word in a.title.lower()
                    ][:3]  # Limit to 3 articles per topic
                })
            
            return topics
            
        except Exception as e:
            logger.error(f"Failed to get trending topics: {str(e)}")
            return []
    
    async def batch_search(
        self,
        queries: List[str],
        max_concurrent: int = 5,
        **kwargs
    ) -> Dict[str, NewsResponse]:
        """
        Perform multiple searches concurrently.
        
        Args:
            queries: List of search queries
            max_concurrent: Maximum concurrent searches
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping queries to news responses
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def search_with_semaphore(query):
            async with semaphore:
                return query, await self.search_articles(query, **kwargs)
        
        tasks = [search_with_semaphore(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        search_results = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch search failed: {str(result)}")
            elif isinstance(result, tuple) and len(result) == 2:
                query, response = result
                search_results[query] = response
        
        return search_results
    
    async def validate_all_connections(self) -> Dict[str, bool]:
        """Validate connections for all providers."""
        results = {}
        
        for source, provider in self.providers.items():
            try:
                results[source.value] = await provider.validate_connection()
            except Exception as e:
                results[source.value] = False
                logger.error(f"Connection validation failed for {source.value}: {str(e)}")
        
        return results


# Factory function for dependency injection
def get_news_client() -> NewsClient:
    """
    Factory function to create news client.
    
    Returns:
        Configured NewsClient instance
    """
    return NewsClient()