"""
Configuration management for the AI Watermark Detection Tool.

This module handles application configuration including model paths,
API settings, detection thresholds, and deployment parameters.
All configuration values are documented with their purpose and valid ranges.
"""

import os
from typing import Dict, Any, List, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from pathlib import Path


class ModelConfig(BaseSettings):
    """
    Configuration for machine learning models.
    
    This class manages model-specific settings including paths,
    loading parameters, and performance optimization options.
    
    Attributes:
        gemma_model_name (str): Hugging Face model identifier for Gemma
        phi_model_name (str): Hugging Face model identifier for Phi
        model_cache_dir (str): Directory for caching downloaded models
        max_model_memory_mb (int): Maximum memory usage per model in MB
        cpu_threads (int): Number of CPU threads for model inference
        model_timeout_seconds (int): Timeout for model loading operations
    """
    
    gemma_model_name: str = Field(
        default="google/gemma-2-2b",
        description="Hugging Face model identifier for SynthID watermarking"
    )
    
    phi_model_name: str = Field(
        default="microsoft/Phi-3-mini-4k-instruct",
        description="Hugging Face model identifier for non-watermarked generation"
    )
    
    model_cache_dir: str = Field(
        default="./models_cache",
        description="Directory path for caching downloaded models"
    )
    
    max_model_memory_mb: int = Field(
        default=2048,
        ge=512,
        le=8192,
        description="Maximum memory usage per model in megabytes"
    )
    
    cpu_threads: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Number of CPU threads for model inference"
    )
    
    model_timeout_seconds: int = Field(
        default=300,
        ge=30,
        le=1800,
        description="Timeout for model loading operations in seconds"
    )
    
    @validator('model_cache_dir')
    def validate_cache_dir(cls, v):
        """Ensure model cache directory exists or can be created."""
        cache_path = Path(v)
        cache_path.mkdir(parents=True, exist_ok=True)
        return str(cache_path.absolute())


class DetectionConfig(BaseSettings):
    """
    Configuration for watermark detection algorithms.
    
    This class manages detection-specific settings including thresholds,
    confidence scoring parameters, and fallback detection options.
    
    Attributes:
        default_threshold (float): Default confidence threshold for classification
        synthid_weight (float): Weight for SynthID detector in combined results
        custom_weight (float): Weight for custom detector in combined results
        min_text_length (int): Minimum text length for reliable detection
        max_text_length (int): Maximum text length to process
        enable_fallback (bool): Whether to use fallback detection methods
    """
    
    default_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Default confidence threshold for watermark classification"
    )
    
    synthid_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for SynthID detector in combined scoring"
    )
    
    custom_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for custom detector in combined scoring"
    )
    
    min_text_length: int = Field(
        default=50,
        ge=10,
        le=500,
        description="Minimum text length for reliable detection"
    )
    
    max_text_length: int = Field(
        default=50000,
        ge=1000,
        le=100000,
        description="Maximum text length to process"
    )
    
    enable_fallback: bool = Field(
        default=True,
        description="Enable fallback detection when primary methods fail"
    )
    
    @validator('synthid_weight', 'custom_weight')
    def validate_weights(cls, v, values):
        """Ensure detection weights are properly balanced."""
        if 'synthid_weight' in values:
            total_weight = v + values.get('synthid_weight', 0)
            if abs(total_weight - 1.0) > 0.01:  # Allow small floating point errors
                raise ValueError("Detection weights must sum to 1.0")
        return v


class APIConfig(BaseSettings):
    """
    Configuration for FastAPI server and HTTP settings.
    
    This class manages API server configuration including ports,
    CORS settings, rate limiting, and security parameters.
    
    Attributes:
        host (str): Server host address
        port (int): Server port number
        debug (bool): Enable debug mode
        cors_origins (List[str]): Allowed CORS origins
        max_request_size_mb (int): Maximum request size in megabytes
        rate_limit_requests_per_minute (int): Rate limiting threshold
        api_key_required (bool): Whether API key authentication is required
    """
    
    host: str = Field(
        default="0.0.0.0",
        description="Server host address"
    )
    
    port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="Server port number"
    )
    
    debug: bool = Field(
        default=False,
        description="Enable debug mode with detailed error messages"
    )
    
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="List of allowed CORS origins"
    )
    
    max_request_size_mb: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum request size in megabytes"
    )
    
    rate_limit_requests_per_minute: int = Field(
        default=60,
        ge=10,
        le=1000,
        description="Maximum requests per minute per client"
    )
    
    api_key_required: bool = Field(
        default=False,
        description="Whether API key authentication is required"
    )


class DatabaseConfig(BaseSettings):
    """
    Configuration for database connections and storage.
    
    This class manages database settings including connection parameters,
    storage paths, and data retention policies.
    
    Attributes:
        database_url (str): SQLite database file path
        max_history_records (int): Maximum number of analysis records to keep
        cleanup_interval_hours (int): Hours between automatic cleanup operations
        backup_enabled (bool): Whether to create periodic database backups
        backup_interval_hours (int): Hours between database backups
    """
    
    database_url: str = Field(
        default="sqlite:///./watermark_detection.db",
        description="SQLite database connection URL"
    )
    
    max_history_records: int = Field(
        default=10000,
        ge=100,
        le=100000,
        description="Maximum number of analysis records to retain"
    )
    
    cleanup_interval_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Hours between automatic cleanup operations"
    )
    
    backup_enabled: bool = Field(
        default=True,
        description="Enable periodic database backups"
    )
    
    backup_interval_hours: int = Field(
        default=168,  # Weekly
        ge=24,
        le=720,
        description="Hours between database backup operations"
    )


class AppConfig(BaseSettings):
    """
    Main application configuration combining all subsystem configs.
    
    This class aggregates configuration from all subsystems and provides
    a unified interface for accessing application settings.
    
    Attributes:
        model (ModelConfig): Model-related configuration
        detection (DetectionConfig): Detection algorithm configuration
        api (APIConfig): API server configuration
        database (DatabaseConfig): Database and storage configuration
        environment (str): Deployment environment (dev, staging, prod)
        log_level (str): Logging level
    """
    
    model: ModelConfig = Field(default_factory=ModelConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    
    environment: str = Field(
        default="development",
        description="Deployment environment"
    )
    
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    
    class Config:
        """Pydantic configuration for environment variable loading."""
        env_file = ".env"
        env_nested_delimiter = "__"
        case_sensitive = False
    
    @validator('environment')
    def validate_environment(cls, v):
        """Validate environment setting."""
        valid_environments = ["development", "staging", "production"]
        if v.lower() not in valid_environments:
            raise ValueError(f"Environment must be one of: {valid_environments}")
        return v.lower()
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate logging level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model configuration as dictionary.
        
        Returns:
            Dict[str, Any]: Model configuration parameters
        """
        return self.model.dict()
    
    def get_detection_config(self) -> Dict[str, Any]:
        """
        Get detection configuration as dictionary.
        
        Returns:
            Dict[str, Any]: Detection algorithm parameters
        """
        return self.detection.dict()
    
    def is_production(self) -> bool:
        """
        Check if running in production environment.
        
        Returns:
            bool: True if in production mode
        """
        return self.environment == "production"
    
    def get_cors_settings(self) -> Dict[str, Any]:
        """
        Get CORS configuration for FastAPI.
        
        Returns:
            Dict[str, Any]: CORS middleware settings
        """
        return {
            "allow_origins": self.api.cors_origins,
            "allow_credentials": True,
            "allow_methods": ["GET", "POST"],
            "allow_headers": ["*"],
        }


# Global configuration instance
config = AppConfig()


def get_config() -> AppConfig:
    """
    Get the global application configuration instance.
    
    This function provides access to the singleton configuration object
    that contains all application settings loaded from environment variables
    and configuration files.
    
    Returns:
        AppConfig: Global configuration instance with all subsystem settings
    """
    return config


def reload_config() -> AppConfig:
    """
    Reload configuration from environment and files.
    
    This function creates a new configuration instance, useful for
    picking up configuration changes without restarting the application.
    
    Returns:
        AppConfig: Newly loaded configuration instance
    """
    global config
    config = AppConfig()
    return config

    # Watermarking settings
    WATERMARKING_ENABLED = True
    DEFAULT_WATERMARK_VISIBILITY = "hidden"
    MAX_TEXT_LENGTH = 10000  # Limit for watermarking operations