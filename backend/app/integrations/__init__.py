# backend/app/integrations/__init__.py
"""
External service integrations for WorldBrief 360.

This module contains clients for various external APIs and services.
All clients follow a consistent pattern with error handling, retry logic,
and proper configuration management.
"""

from .news_api_client import NewsAPIClient, GNewsClient
from .wikipedia_client import WikipediaClient
from .worldbank_client import WorldBankClient
from .imf_client import IMFClient
from .un_client import UNClient
from .openweather_client import OpenWeatherClient
from .maps_client import MapsClient
from .huggingface_client import HuggingFaceClient
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient
from .auth_providers import OAuthProvider, GoogleOAuth, GitHubOAuth
from .email_client import EmailClient
from .sms_client import SMSClient
from .push_notification_client import PushNotificationClient
from .payment_client import PaymentClient

__all__ = [
    'NewsAPIClient',
    'GNewsClient',
    'WikipediaClient',
    'WorldBankClient',
    'IMFClient',
    'UNClient',
    'OpenWeatherClient',
    'MapsClient',
    'HuggingFaceClient',
    'OpenAIClient',
    'AnthropicClient',
    'OAuthProvider',
    'GoogleOAuth',
    'GitHubOAuth',
    'EmailClient',
    'SMSClient',
    'PushNotificationClient',
    'PaymentClient',
]