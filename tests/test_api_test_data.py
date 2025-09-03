"""
Unit tests for test data generation API endpoints.

This module tests the FastAPI test data generation endpoints including
dataset creation, validation, and service management functionality.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

from backend.api.test_data import router, TestDataService, get_test_data_service
from backend.models.schemas import TestDataRequest, TestDataResponse, TestSample


class TestTestDataService:
    """Test cases for TestDataService functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        config = MagicMock()
        config.model.gemma_model_name = "google/gemma-2-2b"
        config.model.phi_model_name = "microsoft/Phi-3-mini-4k-instruct"
        return config
    
    @pytest.fixture
    def test_data_service(self, mock_config):
        """Create TestDataService instance for testing."""
        with patch('backend.api.test_data.get_config', return_value=mock_config):
            return TestDataService()
    
    @pytest.fixture
    def sample_test_samples(self):
        """Create sample test samples."""
        return [
            TestSample(
                text="Watermarked sample text",
                is_watermarked=True,
                expected_score=0.8,
                source_model="google/gemma-2-2b",
                generation_params={}
            ),
            TestSample(
                text="Clean sample text",
                is_watermarked=False,
                expected_score=0.2,
                source_model="microsoft/Phi-3-mini-4k-instruct",
                generation_params={}
            )
        ]
    
    def test_service_initialization(self, test_data_service):
        """Test TestDataService initialization."""
        assert test_data_service.total_generations == 0
        assert test_data_service.successful_generations == 0
        assert test_data_service.failed_generations == 0
        assert test_data_service._generator is None
    
    @patch('backend.api.test_data.create_test_generator')
    def test_get_generator(self, mock_create_generator, test_data_service):
        """Test test data generator initialization."""
        mock_generator = MagicMock()
        mock_create_generator.return_value = mock_generator
        
        generator = test_data_service._get_generator()
        
        assert generator == mock_generator
        assert test_data_service._generator == mock_generator
        mock_create_generator.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_test_data_success(self, test_data_service, sample_test_samples):
        """Test successful test data generation."""
        # Mock generator
        mock_generator = MagicMock()
        mock_generator.create_balanced_dataset.return_value = sample_test_samples
        test_data_service._generator = mock_generator
        
        # Create request
        request = TestDataRequest(
            sample_count=2,
            watermarked_ratio=0.5,
            prompts=["Test prompt 1", "Test prompt 2"],
            max_length=200
        )
        
        background_tasks = MagicMock()
        
        # Generate test data
        response = await test_data_service.generate_test_data(request, background_tasks)
        
        # Verify response
        assert isinstance(response, TestDataResponse)
        assert len(response.samples) == 2
        assert response.total_samples == 2
        assert response.watermarked_count == 1  # One watermarked sample
        assert response.processing_time_ms > 0
        
        # Verify statistics
        assert "total_samples" in response.generation_stats
        assert "watermarked_samples" in response.generation_stats
        assert "clean_samples" in response.generation_stats
        
        # Verify service statistics updated
        assert test_data_service.total_generations == 1
        assert test_data_service.successful_generations == 1
    
    @pytest.mark.asyncio
    async def test_generate_test_data_invalid_count(self, test_data_service):
        """Test test data generation with invalid sample count."""
        request = TestDataRequest(
            sample_count=0,  # Invalid count
            watermarked_ratio=0.5,
            prompts=["Test prompt"],
            max_length=200
        )
        
        background_tasks = MagicMock()
        
        with pytest.raises(Exception):  # Should raise HTTPException
            await test_data_service.generate_test_data(request, background_tasks)
    
    @pytest.mark.asyncio
    async def test_generate_test_data_invalid_ratio(self, test_data_service):
        """Test test data generation with invalid watermarked ratio."""
        request = TestDataRequest(
            sample_count=10,
            watermarked_ratio=1.5,  # Invalid ratio
            prompts=["Test prompt"],
            max_length=200
        )
        
        background_tasks = MagicMock()
        
        with pytest.raises(Exception):  # Should raise HTTPException
            await test_data_service.generate_test_data(request, background_tasks)
    
    @pytest.mark.asyncio
    async def test_generate_test_data_empty_prompts(self, test_data_service):
        """Test test data generation with empty prompts."""
        request = TestDataRequest(
            sample_count=10,
            watermarked_ratio=0.5,
            prompts=[],  # Empty prompts
            max_length=200
        )
        
        background_tasks = MagicMock()
        
        with pytest.raises(Exception):  # Should raise HTTPException
            await test_data_service.generate_test_data(request, background_tasks)
    
    @pytest.mark.asyncio
    async def test_validate_dataset_success(self, test_data_service, sample_test_samples):
        """Test successful dataset validation."""
        # Mock generator
        mock_generator = MagicMock()
        mock_generator.validate_dataset.return_value = {
            "valid": True,
            "total_samples": 2,
            "watermarked_count": 1,
            "clean_count": 1,
            "validation_errors": []
        }
        test_data_service._generator = mock_generator
        
        # Validate dataset
        result = await test_data_service.validate_dataset(sample_test_samples)
        
        # Verify result
        assert result["valid"] is True
        assert result["total_samples"] == 2
        assert result["watermarked_count"] == 1
        assert result["clean_count"] == 1
        assert len(result["validation_errors"]) == 0
    
    def test_get_service_stats(self, test_data_service):
        """Test service statistics retrieval."""
        # Set some test values
        test_data_service.total_generations = 5
        test_data_service.successful_generations = 4
        test_data_service.failed_generations = 1
        
        # Mock generator stats
        mock_generator = MagicMock()
        mock_generator.get_generation_stats.return_value = {
            "total_generations": 10,
            "total_tokens_generated": 1000
        }
        test_data_service._generator = mock_generator
        
        stats = test_data_service.get_service_stats()
        
        assert stats["total_generations"] == 5
        assert stats["successful_generations"] == 4
        assert stats["failed_generations"] == 1
        assert stats["success_rate"] == 0.8
        assert stats["generator_initialized"] is True
        assert "generator_stats" in stats


class TestTestDataAPI:
    """Test cases for test data API endpoints."""
    
    @pytest.fixture
    def app(self):
        """Create FastAPI test application."""
        app = FastAPI()
        app.include_router(router)
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_service(self):
        """Create mock test data service."""
        service = MagicMock()
        service.generate_test_data = AsyncMock()
        service.validate_dataset = AsyncMock()
        service.get_service_stats.return_value = {
            "total_generations": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "success_rate": 0.0,
            "generator_initialized": False,
            "generator_stats": {}
        }
        return service
    
    def test_generate_test_data_endpoint_success(self, client, mock_service):
        """Test successful test data generation endpoint."""
        # Mock response
        mock_response = TestDataResponse(
            samples=[
                TestSample(
                    text="Generated text",
                    is_watermarked=True,
                    expected_score=0.8,
                    source_model="google/gemma-2-2b",
                    generation_params={}
                )
            ],
            generation_stats={"total_samples": 1},
            total_samples=1,
            watermarked_count=1,
            processing_time_ms=1000
        )
        mock_service.generate_test_data.return_value = mock_response
        
        # Patch service dependency
        with patch('backend.api.test_data.get_test_data_service', return_value=mock_service):
            response = client.post(
                "/api/generate-test-data",
                json={
                    "sample_count": 1,
                    "watermarked_ratio": 1.0,
                    "prompts": ["Test prompt"],
                    "max_length": 200
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_samples"] == 1
        assert data["watermarked_count"] == 1
        assert data["processing_time_ms"] == 1000
        assert len(data["samples"]) == 1
    
    def test_generate_test_data_endpoint_invalid_request(self, client, mock_service):
        """Test test data generation endpoint with invalid request."""
        with patch('backend.api.test_data.get_test_data_service', return_value=mock_service):
            response = client.post(
                "/api/generate-test-data",
                json={
                    "sample_count": -1,  # Invalid count
                    "watermarked_ratio": 0.5,
                    "prompts": ["Test prompt"],
                    "max_length": 200
                }
            )
        
        # Should return validation error
        assert response.status_code >= 400
    
    def test_validate_dataset_endpoint_success(self, client, mock_service):
        """Test successful dataset validation endpoint."""
        # Mock validation response
        mock_validation = {
            "valid": True,
            "total_samples": 2,
            "watermarked_count": 1,
            "clean_count": 1,
            "validation_errors": []
        }
        mock_service.validate_dataset.return_value = mock_validation
        
        # Sample data for validation
        sample_data = [
            {
                "text": "Sample text 1",
                "is_watermarked": True,
                "expected_score": 0.8,
                "source_model": "google/gemma-2-2b",
                "generation_params": {},
                "created_at": "2024-01-01T00:00:00"
            },
            {
                "text": "Sample text 2",
                "is_watermarked": False,
                "expected_score": 0.2,
                "source_model": "microsoft/Phi-3-mini-4k-instruct",
                "generation_params": {},
                "created_at": "2024-01-01T00:00:00"
            }
        ]
        
        with patch('backend.api.test_data.get_test_data_service', return_value=mock_service):
            response = client.post(
                "/api/validate-dataset",
                json=sample_data
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert data["total_samples"] == 2
        assert data["watermarked_count"] == 1
        assert data["clean_count"] == 1
    
    def test_generation_stats_endpoint(self, client, mock_service):
        """Test generation statistics endpoint."""
        with patch('backend.api.test_data.get_test_data_service', return_value=mock_service):
            response = client.get("/api/generation-stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_generations" in data
        assert "successful_generations" in data
        assert "generator_initialized" in data
    
    def test_default_prompts_endpoint(self, client):
        """Test default prompts endpoint."""
        # Mock the generator and prompts
        mock_generator = MagicMock()
        mock_generator.default_prompts = ["Prompt 1", "Prompt 2"]
        
        mock_evaluation_prompts = ["Eval prompt 1", "Eval prompt 2"]
        
        with patch('backend.api.test_data.create_test_generator', return_value=mock_generator):
            with patch('backend.utils.test_data_generator.create_evaluation_prompts', return_value=mock_evaluation_prompts):
                response = client.get("/api/default-prompts")
        
        assert response.status_code == 200
        data = response.json()
        assert "default_prompts" in data
        assert "evaluation_prompts" in data
        assert "total_default" in data
        assert "total_evaluation" in data
        assert len(data["default_prompts"]) == 2
        assert len(data["evaluation_prompts"]) == 2


class TestTestDataIntegration:
    """Integration tests for test data API with actual components."""
    
    @pytest.fixture
    def app(self):
        """Create FastAPI test application."""
        app = FastAPI()
        app.include_router(router)
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    @patch('backend.api.test_data.create_test_generator')
    def test_end_to_end_generation_flow(self, mock_create_generator, client):
        """Test end-to-end test data generation flow."""
        # Mock generator with realistic behavior
        mock_generator = MagicMock()
        
        # Mock successful dataset creation
        mock_samples = [
            TestSample(
                text="Generated watermarked text for testing purposes",
                is_watermarked=True,
                expected_score=0.85,
                source_model="google/gemma-2-2b",
                generation_params={"temperature": 0.8}
            ),
            TestSample(
                text="Generated clean text for testing purposes",
                is_watermarked=False,
                expected_score=0.15,
                source_model="microsoft/Phi-3-mini-4k-instruct",
                generation_params={"temperature": 0.8}
            )
        ]
        
        mock_generator.create_balanced_dataset.return_value = mock_samples
        mock_generator.get_generation_stats.return_value = {
            "total_generations": 2,
            "total_tokens_generated": 100
        }
        
        mock_create_generator.return_value = mock_generator
        
        # Test generation request
        response = client.post(
            "/api/generate-test-data",
            json={
                "sample_count": 2,
                "watermarked_ratio": 0.5,
                "prompts": ["Write about AI technology"],
                "max_length": 150
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "samples" in data
        assert "generation_stats" in data
        assert "total_samples" in data
        assert "watermarked_count" in data
        assert "processing_time_ms" in data
        
        # Verify sample data
        assert len(data["samples"]) == 2
        assert data["total_samples"] == 2
        assert data["watermarked_count"] == 1
        
        # Verify individual samples
        samples = data["samples"]
        watermarked_sample = next(s for s in samples if s["is_watermarked"])
        clean_sample = next(s for s in samples if not s["is_watermarked"])
        
        assert watermarked_sample["expected_score"] > 0.5
        assert clean_sample["expected_score"] < 0.5
        assert watermarked_sample["source_model"] == "google/gemma-2-2b"
        assert clean_sample["source_model"] == "microsoft/Phi-3-mini-4k-instruct"


if __name__ == "__main__":
    pytest.main([__file__])