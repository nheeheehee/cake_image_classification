'''config file for API settings'''

from pydantic import BaseSettings
from functools import lru_cache

class Settings (BaseSettings):
    '''BaseSettings'''
    environment: str = 'dev'

@lru_cache
def get_settings() -> BaseSettings:
    """Get base settings. Cache maximum 128 previous calls."""
    return Settings()
