"""
Database models for the AI Watermark Detection Tool.

This module defines the database models and data structures for
storing analysis results, test datasets, and application data.
"""

import json
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict
from uuid import uuid4

from models.schemas import DetectionResponse, TestSample


@dataclass
class AnalysisResultModel:
    """
    Database model for analysis results.
    
    This class represents an analysis result stored in the database,
    including all detection information and metadata.
    
    Attributes:
        id (str): Unique identifier for the analysis
        timestamp (datetime): When the analysis was performed
        input_text (str): Original text that was analyzed
        input_text_hash (str): Hash of the input text for deduplication
        confidence_score (float): Detection confidence score
        is_watermarked (bool): Binary classification result
        model_identified (Optional[str]): Identified source model
        detection_methods (List[str]): Methods used for detection
        analysis_metadata (Dict[str, Any]): Detailed analysis metadata
        processing_time_ms (int): Processing time in milliseconds
        user_feedback (Optional[str]): Optional user feedback
        created_at (Optional[datetime]): Record creation timestamp
        updated_at (Optional[datetime]): Record update timestamp
    """
    id: str
    timestamp: datetime
    input_text: str
    input_text_hash: str
    confidence_score: float
    is_watermarked: bool
    model_identified: Optional[str]
    detection_methods: List[str]
    analysis_metadata: Dict[str, Any]
    processing_time_ms: int
    user_feedback: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    @classmethod
    def from_detection_response(
        cls,
        input_text: str,
        response: DetectionResponse,
        user_feedback: Optional[str] = None
    ) -> 'AnalysisResultModel':
        """
        Create an AnalysisResultModel from a DetectionResponse.
        
        Args:
            input_text (str): Original input text
            response (DetectionResponse): Detection response data
            user_feedback (Optional[str]): Optional user feedback
            
        Returns:
            AnalysisResultModel: Database model instance
        """
        # Generate hash of input text for deduplication
        text_hash = hashlib.sha256(input_text.encode('utf-8')).hexdigest()
        
        return cls(
            id=str(uuid4()),
            timestamp=response.timestamp,
            input_text=input_text,
            input_text_hash=text_hash,
            confidence_score=response.confidence_score,
            is_watermarked=response.is_watermarked,
            model_identified=response.model_identified,
            detection_methods=response.detection_methods_used,
            analysis_metadata=response.analysis_metadata,
            processing_time_ms=response.processing_time_ms,
            user_feedback=user_feedback
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert model to dictionary for database storage.
        
        Returns:
            Dict[str, Any]: Dictionary representation suitable for database
        """
        data = asdict(self)
        
        # Convert datetime objects to ISO strings
        if self.timestamp:
            data['timestamp'] = self.timestamp.isoformat()
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.updated_at:
            data['updated_at'] = self.updated_at.isoformat()
        
        # Convert lists and dicts to JSON strings
        data['detection_methods'] = json.dumps(self.detection_methods)
        data['analysis_metadata'] = json.dumps(self.analysis_metadata)
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisResultModel':
        """
        Create model from dictionary (database row).
        
        Args:
            data (Dict[str, Any]): Dictionary from database
            
        Returns:
            AnalysisResultModel: Model instance
        """
        # Parse datetime strings
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if isinstance(data.get('updated_at'), str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        # Parse JSON strings
        if isinstance(data.get('detection_methods'), str):
            data['detection_methods'] = json.loads(data['detection_methods'])
        if isinstance(data.get('analysis_metadata'), str):
            data['analysis_metadata'] = json.loads(data['analysis_metadata'])
        
        return cls(**data)
    
    def to_detection_response(self) -> DetectionResponse:
        """
        Convert model back to DetectionResponse.
        
        Returns:
            DetectionResponse: API response format
        """
        return DetectionResponse(
            confidence_score=self.confidence_score,
            is_watermarked=self.is_watermarked,
            model_identified=self.model_identified,
            detection_methods_used=self.detection_methods,
            analysis_metadata=self.analysis_metadata,
            processing_time_ms=self.processing_time_ms,
            timestamp=self.timestamp
        )


@dataclass
class TestDatasetModel:
    """
    Database model for test dataset samples.
    
    This class represents a test dataset sample stored in the database,
    including generation parameters and metadata.
    
    Attributes:
        id (str): Unique identifier for the dataset sample
        name (str): Human-readable name for the sample
        description (Optional[str]): Optional description
        text_content (str): Generated text content
        is_watermarked (bool): Whether the text is watermarked
        expected_score (float): Expected detection confidence score
        source_model (str): Model used to generate the text
        generation_params (Dict[str, Any]): Parameters used for generation
        dataset_metadata (Dict[str, Any]): Additional metadata
        created_at (Optional[datetime]): Record creation timestamp
        updated_at (Optional[datetime]): Record update timestamp
    """
    id: str
    name: str
    description: Optional[str]
    text_content: str
    is_watermarked: bool
    expected_score: float
    source_model: str
    generation_params: Dict[str, Any]
    dataset_metadata: Dict[str, Any]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    @classmethod
    def from_test_sample(
        cls,
        sample: TestSample,
        name: Optional[str] = None,
        description: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> 'TestDatasetModel':
        """
        Create a TestDatasetModel from a TestSample.
        
        Args:
            sample (TestSample): Test sample data
            name (Optional[str]): Name for the dataset sample
            description (Optional[str]): Description of the sample
            additional_metadata (Optional[Dict[str, Any]]): Additional metadata
            
        Returns:
            TestDatasetModel: Database model instance
        """
        sample_id = str(uuid4())
        
        # Generate name if not provided
        if name is None:
            watermark_type = "watermarked" if sample.is_watermarked else "clean"
            name = f"{watermark_type}_sample_{sample_id[:8]}"
        
        # Combine metadata
        metadata = {
            "sample_created_at": sample.created_at.isoformat(),
            "text_length": len(sample.text),
            "word_count": len(sample.text.split())
        }
        
        if additional_metadata:
            metadata.update(additional_metadata)
        
        return cls(
            id=sample_id,
            name=name,
            description=description,
            text_content=sample.text,
            is_watermarked=sample.is_watermarked,
            expected_score=sample.expected_score,
            source_model=sample.source_model,
            generation_params=sample.generation_params,
            dataset_metadata=metadata
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert model to dictionary for database storage.
        
        Returns:
            Dict[str, Any]: Dictionary representation suitable for database
        """
        data = asdict(self)
        
        # Convert datetime objects to ISO strings
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.updated_at:
            data['updated_at'] = self.updated_at.isoformat()
        
        # Convert dicts to JSON strings
        data['generation_params'] = json.dumps(self.generation_params)
        data['dataset_metadata'] = json.dumps(self.dataset_metadata)
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestDatasetModel':
        """
        Create model from dictionary (database row).
        
        Args:
            data (Dict[str, Any]): Dictionary from database
            
        Returns:
            TestDatasetModel: Model instance
        """
        # Parse datetime strings
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if isinstance(data.get('updated_at'), str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        # Parse JSON strings
        if isinstance(data.get('generation_params'), str):
            data['generation_params'] = json.loads(data['generation_params'])
        if isinstance(data.get('dataset_metadata'), str):
            data['dataset_metadata'] = json.loads(data['dataset_metadata'])
        
        return cls(**data)
    
    def to_test_sample(self) -> TestSample:
        """
        Convert model back to TestSample.
        
        Returns:
            TestSample: API format test sample
        """
        # Extract original creation time from metadata
        created_at = self.created_at
        if 'sample_created_at' in self.dataset_metadata:
            try:
                created_at = datetime.fromisoformat(self.dataset_metadata['sample_created_at'])
            except (ValueError, KeyError):
                pass
        
        return TestSample(
            text=self.text_content,
            is_watermarked=self.is_watermarked,
            expected_score=self.expected_score,
            source_model=self.source_model,
            generation_params=self.generation_params,
            created_at=created_at or datetime.utcnow()
        )


@dataclass
class DatasetCollectionModel:
    """
    Database model for dataset collections.
    
    This class represents a collection of test dataset samples,
    allowing for organized grouping and management of test data.
    
    Attributes:
        id (str): Unique identifier for the collection
        name (str): Human-readable name for the collection
        description (Optional[str]): Optional description
        total_samples (int): Total number of samples in collection
        watermarked_count (int): Number of watermarked samples
        clean_count (int): Number of clean samples
        collection_metadata (Dict[str, Any]): Collection metadata
        created_at (Optional[datetime]): Record creation timestamp
        updated_at (Optional[datetime]): Record update timestamp
    """
    id: str
    name: str
    description: Optional[str]
    total_samples: int
    watermarked_count: int
    clean_count: int
    collection_metadata: Dict[str, Any]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    @classmethod
    def create_new(
        cls,
        name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'DatasetCollectionModel':
        """
        Create a new dataset collection.
        
        Args:
            name (str): Collection name
            description (Optional[str]): Collection description
            metadata (Optional[Dict[str, Any]]): Additional metadata
            
        Returns:
            DatasetCollectionModel: New collection instance
        """
        return cls(
            id=str(uuid4()),
            name=name,
            description=description,
            total_samples=0,
            watermarked_count=0,
            clean_count=0,
            collection_metadata=metadata or {}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert model to dictionary for database storage.
        
        Returns:
            Dict[str, Any]: Dictionary representation suitable for database
        """
        data = asdict(self)
        
        # Convert datetime objects to ISO strings
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.updated_at:
            data['updated_at'] = self.updated_at.isoformat()
        
        # Convert dict to JSON string
        data['collection_metadata'] = json.dumps(self.collection_metadata)
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetCollectionModel':
        """
        Create model from dictionary (database row).
        
        Args:
            data (Dict[str, Any]): Dictionary from database
            
        Returns:
            DatasetCollectionModel: Model instance
        """
        # Parse datetime strings
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if isinstance(data.get('updated_at'), str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        # Parse JSON string
        if isinstance(data.get('collection_metadata'), str):
            data['collection_metadata'] = json.loads(data['collection_metadata'])
        
        return cls(**data)
    
    def update_counts(self, watermarked_count: int, clean_count: int) -> None:
        """
        Update sample counts for the collection.
        
        Args:
            watermarked_count (int): Number of watermarked samples
            clean_count (int): Number of clean samples
        """
        self.watermarked_count = watermarked_count
        self.clean_count = clean_count
        self.total_samples = watermarked_count + clean_count


@dataclass
class AppSettingModel:
    """
    Database model for application settings.
    
    This class represents application configuration settings
    stored in the database for persistence across restarts.
    
    Attributes:
        key (str): Setting key/name
        value (str): Setting value (JSON serialized)
        description (Optional[str]): Optional description
        updated_at (Optional[datetime]): Last update timestamp
    """
    key: str
    value: str
    description: Optional[str] = None
    updated_at: Optional[datetime] = None
    
    @classmethod
    def create_setting(
        cls,
        key: str,
        value: Any,
        description: Optional[str] = None
    ) -> 'AppSettingModel':
        """
        Create a new application setting.
        
        Args:
            key (str): Setting key
            value (Any): Setting value (will be JSON serialized)
            description (Optional[str]): Setting description
            
        Returns:
            AppSettingModel: New setting instance
        """
        return cls(
            key=key,
            value=json.dumps(value),
            description=description
        )
    
    def get_value(self) -> Any:
        """
        Get the deserialized setting value.
        
        Returns:
            Any: Deserialized setting value
        """
        try:
            return json.loads(self.value)
        except (json.JSONDecodeError, TypeError):
            return self.value
    
    def set_value(self, value: Any) -> None:
        """
        Set the setting value (will be JSON serialized).
        
        Args:
            value (Any): New setting value
        """
        self.value = json.dumps(value)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert model to dictionary for database storage.
        
        Returns:
            Dict[str, Any]: Dictionary representation suitable for database
        """
        data = asdict(self)
        
        # Convert datetime to ISO string
        if self.updated_at:
            data['updated_at'] = self.updated_at.isoformat()
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AppSettingModel':
        """
        Create model from dictionary (database row).
        
        Args:
            data (Dict[str, Any]): Dictionary from database
            
        Returns:
            AppSettingModel: Model instance
        """
        # Parse datetime string
        if isinstance(data.get('updated_at'), str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        return cls(**data)