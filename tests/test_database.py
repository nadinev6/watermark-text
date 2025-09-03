"""
Unit tests for database operations and models.

This module tests the database layer including connection management,
model serialization, and CRUD operations for analysis results and test datasets.
"""

import pytest
import tempfile
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from backend.database.connection import DatabaseManager
from backend.database.models import (
    AnalysisResultModel,
    TestDatasetModel,
    DatasetCollectionModel,
    AppSettingModel
)
from backend.database.operations import (
    AnalysisResultOperations,
    TestDatasetOperations
)
from backend.models.schemas import DetectionResponse, TestSample


class TestDatabaseModels:
    """Test cases for database model classes."""
    
    def test_analysis_result_model_creation(self):
        """Test AnalysisResultModel creation and serialization."""
        # Create a detection response
        response = DetectionResponse(
            confidence_score=0.85,
            is_watermarked=True,
            model_identified="google/gemma-2-2b",
            detection_methods_used=["synthid", "custom"],
            analysis_metadata={"test": "data"},
            processing_time_ms=150,
            timestamp=datetime.utcnow()
        )
        
        # Create model from response
        model = AnalysisResultModel.from_detection_response(
            "Test input text", response, "Good detection"
        )
        
        # Verify model attributes
        assert model.input_text == "Test input text"
        assert model.confidence_score == 0.85
        assert model.is_watermarked is True
        assert model.model_identified == "google/gemma-2-2b"
        assert model.detection_methods == ["synthid", "custom"]
        assert model.user_feedback == "Good detection"
        assert len(model.input_text_hash) == 64  # SHA256 hash length
    
    def test_analysis_result_model_serialization(self):
        """Test AnalysisResultModel to_dict and from_dict methods."""
        # Create model
        model = AnalysisResultModel(
            id="test-id",
            timestamp=datetime.utcnow(),
            input_text="Test text",
            input_text_hash="test-hash",
            confidence_score=0.75,
            is_watermarked=False,
            model_identified=None,
            detection_methods=["custom"],
            analysis_metadata={"key": "value"},
            processing_time_ms=100
        )
        
        # Convert to dict
        data = model.to_dict()
        
        # Verify serialization
        assert data["id"] == "test-id"
        assert data["confidence_score"] == 0.75
        assert data["is_watermarked"] is False
        assert isinstance(data["timestamp"], str)  # Should be ISO string
        assert isinstance(data["detection_methods"], str)  # Should be JSON string
        assert isinstance(data["analysis_metadata"], str)  # Should be JSON string
        
        # Convert back to model
        restored_model = AnalysisResultModel.from_dict(data)
        
        # Verify restoration
        assert restored_model.id == model.id
        assert restored_model.confidence_score == model.confidence_score
        assert restored_model.is_watermarked == model.is_watermarked
        assert restored_model.detection_methods == model.detection_methods
        assert restored_model.analysis_metadata == model.analysis_metadata
    
    def test_test_dataset_model_creation(self):
        """Test TestDatasetModel creation from TestSample."""
        # Create test sample
        sample = TestSample(
            text="Generated test text",
            is_watermarked=True,
            expected_score=0.9,
            source_model="google/gemma-2-2b",
            generation_params={"temperature": 0.8},
            created_at=datetime.utcnow()
        )
        
        # Create model from sample
        model = TestDatasetModel.from_test_sample(
            sample, "Test Sample", "A test sample for validation"
        )
        
        # Verify model attributes
        assert model.name == "Test Sample"
        assert model.description == "A test sample for validation"
        assert model.text_content == "Generated test text"
        assert model.is_watermarked is True
        assert model.expected_score == 0.9
        assert model.source_model == "google/gemma-2-2b"
        assert model.generation_params == {"temperature": 0.8}
        assert "text_length" in model.dataset_metadata
        assert "word_count" in model.dataset_metadata
    
    def test_dataset_collection_model_creation(self):
        """Test DatasetCollectionModel creation and count updates."""
        # Create collection
        collection = DatasetCollectionModel.create_new(
            "Test Collection",
            "A collection for testing",
            {"purpose": "testing"}
        )
        
        # Verify initial state
        assert collection.name == "Test Collection"
        assert collection.description == "A collection for testing"
        assert collection.total_samples == 0
        assert collection.watermarked_count == 0
        assert collection.clean_count == 0
        assert collection.collection_metadata == {"purpose": "testing"}
        
        # Update counts
        collection.update_counts(5, 3)
        
        # Verify updated counts
        assert collection.watermarked_count == 5
        assert collection.clean_count == 3
        assert collection.total_samples == 8
    
    def test_app_setting_model_operations(self):
        """Test AppSettingModel value operations."""
        # Create setting with complex value
        setting = AppSettingModel.create_setting(
            "test_config",
            {"enabled": True, "threshold": 0.5, "methods": ["a", "b"]},
            "Test configuration setting"
        )
        
        # Verify creation
        assert setting.key == "test_config"
        assert setting.description == "Test configuration setting"
        
        # Verify value serialization/deserialization
        value = setting.get_value()
        assert value == {"enabled": True, "threshold": 0.5, "methods": ["a", "b"]}
        
        # Update value
        setting.set_value({"enabled": False, "threshold": 0.7})
        updated_value = setting.get_value()
        assert updated_value == {"enabled": False, "threshold": 0.7}


class TestDatabaseConnection:
    """Test cases for database connection management."""
    
    @pytest.fixture
    async def temp_db_manager(self):
        """Create temporary database manager for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        
        # Mock config to use temporary database
        mock_config = MagicMock()
        mock_config.database.database_url = f"sqlite:///{temp_path}"
        mock_config.database.max_history_records = 1000
        
        with patch('backend.database.connection.get_config', return_value=mock_config):
            manager = DatabaseManager()
            await manager.initialize()
            
            yield manager
            
            # Cleanup
            await manager.close()
            Path(temp_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_database_initialization(self, temp_db_manager):
        """Test database initialization and schema creation."""
        manager = temp_db_manager
        
        # Verify initialization
        assert manager._initialized is True
        assert len(manager._connection_pool) > 0
        
        # Verify tables exist
        tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
        results = await manager.execute_query(tables_query)
        
        table_names = [row['name'] for row in results]
        expected_tables = [
            'analysis_results', 'test_datasets', 'dataset_collections',
            'dataset_collection_items', 'app_settings'
        ]
        
        for table in expected_tables:
            assert table in table_names
    
    @pytest.mark.asyncio
    async def test_connection_pool_management(self, temp_db_manager):
        """Test connection pool get and return operations."""
        manager = temp_db_manager
        
        initial_pool_size = len(manager._connection_pool)
        
        # Get connection
        async with manager.get_connection() as conn:
            assert conn is not None
            # Pool should have one less connection
            assert len(manager._connection_pool) == initial_pool_size - 1
        
        # Connection should be returned to pool
        assert len(manager._connection_pool) == initial_pool_size
    
    @pytest.mark.asyncio
    async def test_transaction_management(self, temp_db_manager):
        """Test transaction commit and rollback."""
        manager = temp_db_manager
        
        # Test successful transaction
        async with manager.transaction() as conn:
            await conn.execute(
                "INSERT INTO app_settings (key, value) VALUES (?, ?)",
                ("test_key", "test_value")
            )
        
        # Verify data was committed
        results = await manager.execute_query(
            "SELECT * FROM app_settings WHERE key = ?",
            ("test_key",)
        )
        assert len(results) == 1
        assert results[0]['value'] == "test_value"
        
        # Test transaction rollback
        try:
            async with manager.transaction() as conn:
                await conn.execute(
                    "INSERT INTO app_settings (key, value) VALUES (?, ?)",
                    ("test_key2", "test_value2")
                )
                # Force an error to trigger rollback
                raise Exception("Test error")
        except Exception:
            pass
        
        # Verify data was not committed
        results = await manager.execute_query(
            "SELECT * FROM app_settings WHERE key = ?",
            ("test_key2",)
        )
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_database_stats(self, temp_db_manager):
        """Test database statistics retrieval."""
        manager = temp_db_manager
        
        stats = await manager.get_database_stats()
        
        # Verify stats structure
        assert "database_path" in stats
        assert "database_size_mb" in stats
        assert "tables" in stats
        assert "indexes" in stats
        assert "connection_pool_size" in stats
        assert "initialized" in stats
        
        # Verify table counts
        assert isinstance(stats["tables"], dict)
        assert "analysis_results" in stats["tables"]
        assert stats["tables"]["analysis_results"] == 0  # Should be empty initially


class TestAnalysisResultOperations:
    """Test cases for analysis result database operations."""
    
    @pytest.fixture
    async def temp_operations(self):
        """Create temporary analysis result operations for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        
        mock_config = MagicMock()
        mock_config.database.database_url = f"sqlite:///{temp_path}"
        mock_config.database.max_history_records = 1000
        
        with patch('backend.database.connection.get_config', return_value=mock_config):
            manager = DatabaseManager()
            await manager.initialize()
            
            operations = AnalysisResultOperations(manager)
            
            yield operations
            
            # Cleanup
            await manager.close()
            Path(temp_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_save_and_retrieve_analysis_result(self, temp_operations):
        """Test saving and retrieving analysis results."""
        operations = temp_operations
        
        # Create test data
        input_text = "Test text for analysis"
        response = DetectionResponse(
            confidence_score=0.8,
            is_watermarked=True,
            model_identified="google/gemma-2-2b",
            detection_methods_used=["synthid"],
            analysis_metadata={"test": "metadata"},
            processing_time_ms=200,
            timestamp=datetime.utcnow()
        )
        
        # Save analysis result
        result_id = await operations.save_analysis_result(
            input_text, response, "Test feedback"
        )
        
        assert result_id is not None
        assert len(result_id) > 0
        
        # Retrieve analysis result
        retrieved = await operations.get_analysis_result(result_id)
        
        assert retrieved is not None
        assert retrieved.input_text == input_text
        assert retrieved.confidence_score == 0.8
        assert retrieved.is_watermarked is True
        assert retrieved.model_identified == "google/gemma-2-2b"
        assert retrieved.user_feedback == "Test feedback"
    
    @pytest.mark.asyncio
    async def test_analysis_history_pagination(self, temp_operations):
        """Test analysis history retrieval with pagination."""
        operations = temp_operations
        
        # Create multiple analysis results
        for i in range(25):
            response = DetectionResponse(
                confidence_score=0.5 + (i * 0.01),
                is_watermarked=i % 2 == 0,
                model_identified="test-model" if i % 2 == 0 else None,
                detection_methods_used=["synthid"],
                analysis_metadata={},
                processing_time_ms=100 + i,
                timestamp=datetime.utcnow() - timedelta(minutes=i)
            )
            
            await operations.save_analysis_result(f"Test text {i}", response)
        
        # Test first page
        results, total_count = await operations.get_analysis_history(
            page=1, page_size=10
        )
        
        assert len(results) == 10
        assert total_count == 25
        
        # Test second page
        results, total_count = await operations.get_analysis_history(
            page=2, page_size=10
        )
        
        assert len(results) == 10
        assert total_count == 25
        
        # Test last page
        results, total_count = await operations.get_analysis_history(
            page=3, page_size=10
        )
        
        assert len(results) == 5  # Remaining results
        assert total_count == 25
    
    @pytest.mark.asyncio
    async def test_analysis_history_filtering(self, temp_operations):
        """Test analysis history filtering by watermarked status."""
        operations = temp_operations
        
        # Create mixed results
        for i in range(10):
            response = DetectionResponse(
                confidence_score=0.7,
                is_watermarked=i < 5,  # First 5 are watermarked
                model_identified="test-model" if i < 5 else None,
                detection_methods_used=["synthid"],
                analysis_metadata={},
                processing_time_ms=100,
                timestamp=datetime.utcnow()
            )
            
            await operations.save_analysis_result(f"Test text {i}", response)
        
        # Filter for watermarked only
        watermarked_results, watermarked_count = await operations.get_analysis_history(
            watermarked_filter=True
        )
        
        assert watermarked_count == 5
        assert all(result.is_watermarked for result in watermarked_results)
        
        # Filter for clean only
        clean_results, clean_count = await operations.get_analysis_history(
            watermarked_filter=False
        )
        
        assert clean_count == 5
        assert all(not result.is_watermarked for result in clean_results)
    
    @pytest.mark.asyncio
    async def test_update_user_feedback(self, temp_operations):
        """Test updating user feedback for analysis results."""
        operations = temp_operations
        
        # Create and save analysis result
        response = DetectionResponse(
            confidence_score=0.8,
            is_watermarked=True,
            model_identified="test-model",
            detection_methods_used=["synthid"],
            analysis_metadata={},
            processing_time_ms=100,
            timestamp=datetime.utcnow()
        )
        
        result_id = await operations.save_analysis_result("Test text", response)
        
        # Update feedback
        success = await operations.update_user_feedback(
            result_id, "Updated feedback"
        )
        
        assert success is True
        
        # Verify update
        retrieved = await operations.get_analysis_result(result_id)
        assert retrieved.user_feedback == "Updated feedback"
    
    @pytest.mark.asyncio
    async def test_analysis_statistics(self, temp_operations):
        """Test analysis statistics calculation."""
        operations = temp_operations
        
        # Create test data with known statistics
        base_time = datetime.utcnow()
        
        for i in range(20):
            response = DetectionResponse(
                confidence_score=0.5 + (i * 0.02),  # Increasing confidence
                is_watermarked=i < 10,  # Half watermarked
                model_identified="model-a" if i < 10 else "model-b",
                detection_methods_used=["synthid"],
                analysis_metadata={},
                processing_time_ms=100 + i,  # Increasing processing time
                timestamp=base_time - timedelta(hours=i)
            )
            
            await operations.save_analysis_result(f"Test text {i}", response)
        
        # Get statistics
        stats = await operations.get_analysis_statistics(days=7)
        
        # Verify statistics
        assert stats["total_analyses"] == 20
        assert stats["watermarked_count"] == 10
        assert stats["clean_count"] == 10
        assert stats["watermarked_ratio"] == 0.5
        assert stats["avg_confidence"] > 0.5
        assert stats["avg_processing_time_ms"] > 100
        assert "model_distribution" in stats
        assert "method_usage" in stats


class TestTestDatasetOperations:
    """Test cases for test dataset database operations."""
    
    @pytest.fixture
    async def temp_dataset_operations(self):
        """Create temporary test dataset operations for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        
        mock_config = MagicMock()
        mock_config.database.database_url = f"sqlite:///{temp_path}"
        
        with patch('backend.database.connection.get_config', return_value=mock_config):
            manager = DatabaseManager()
            await manager.initialize()
            
            operations = TestDatasetOperations(manager)
            
            yield operations
            
            # Cleanup
            await manager.close()
            Path(temp_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_save_test_samples_with_collection(self, temp_dataset_operations):
        """Test saving test samples with collection creation."""
        operations = temp_dataset_operations
        
        # Create test samples
        samples = [
            TestSample(
                text="Watermarked sample text",
                is_watermarked=True,
                expected_score=0.9,
                source_model="google/gemma-2-2b",
                generation_params={"temperature": 0.8},
                created_at=datetime.utcnow()
            ),
            TestSample(
                text="Clean sample text",
                is_watermarked=False,
                expected_score=0.1,
                source_model="microsoft/Phi-3-mini-4k-instruct",
                generation_params={"temperature": 0.8},
                created_at=datetime.utcnow()
            )
        ]
        
        # Save samples with collection
        collection_id, sample_ids = await operations.save_test_samples(
            samples,
            collection_name="Test Collection",
            collection_description="A test collection"
        )
        
        assert collection_id is not None
        assert len(sample_ids) == 2
        
        # Verify collection was created
        collections, total_count = await operations.get_dataset_collections()
        
        assert total_count == 1
        assert len(collections) == 1
        assert collections[0].name == "Test Collection"
        assert collections[0].total_samples == 2
        assert collections[0].watermarked_count == 1
        assert collections[0].clean_count == 1
    
    @pytest.mark.asyncio
    async def test_get_collection_samples(self, temp_dataset_operations):
        """Test retrieving samples from a collection."""
        operations = temp_dataset_operations
        
        # Create and save test samples
        samples = [
            TestSample(
                text=f"Sample text {i}",
                is_watermarked=i % 2 == 0,
                expected_score=0.8 if i % 2 == 0 else 0.2,
                source_model="test-model",
                generation_params={},
                created_at=datetime.utcnow()
            )
            for i in range(10)
        ]
        
        collection_id, _ = await operations.save_test_samples(
            samples, "Test Collection"
        )
        
        # Retrieve samples from collection
        retrieved_samples, total_count = await operations.get_collection_samples(
            collection_id, page=1, page_size=5
        )
        
        assert len(retrieved_samples) == 5
        assert total_count == 10
        
        # Verify sample order and content
        for i, sample in enumerate(retrieved_samples):
            assert sample.text_content == f"Sample text {i}"
            assert sample.is_watermarked == (i % 2 == 0)


if __name__ == "__main__":
    pytest.main([__file__])