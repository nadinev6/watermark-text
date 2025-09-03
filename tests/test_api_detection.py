"""
Unit tests for detection API endpoints.

This module tests the FastAPI detection endpoints including single text
analysis, batch processing, and service management functionality.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

from backend.api.detection import router, DetectionService, get_detection_service
from backend.models.schemas import DetectionRequest, DetectionResponse
from backend.detectors.base import DetectionResult, DetectionMethod


class TestDetectionService:
    """Test cases for DetectionService functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        config = MagicMock()
        config.api.rate_limit_requests_per_minute = 60
        config.api.max_request_size_mb = 10
        config.api.debug = True
        return config
    
    @pytest.fixture
    def detection_service(self, mock_config):
        """Create DetectionService instance for testing."""
        with patch('backend.api.detection.get_config', return_value=mock_config):
            return DetectionService()
    
    @pytest.fixture
    def mock_detection_result(self):
        """Create mock detection result."""
        return DetectionResult(
            confidence_score=0.8,
            is_watermarked=True,
            model_identified="google/gemma-2-2b",
            detection_method=DetectionMethod.SYNTHID.value,
            metadata={"test": "data"},
            processing_time_ms=100
        )
    
    def test_service_initialization(self, detection_service):
        """Test DetectionService initialization."""
        assert detection_service.total_detections == 0
        assert detection_service.successful_detections == 0
        assert detection_service.failed_detections == 0
        assert detection_service._synthid_detector is None
        assert detection_service._custom_detector is None
    
    @patch('backend.api.detection.SynthIDDetector')
    def test_get_synthid_detector(self, mock_synthid_class, detection_service):
        """Test SynthID detector initialization."""
        mock_detector = MagicMock()
        mock_synthid_class.return_value = mock_detector
        
        detector = detection_service._get_synthid_detector()
        
        assert detector == mock_detector
        assert detection_service._synthid_detector == mock_detector
        mock_synthid_class.assert_called_once()
    
    @patch('backend.api.detection.CustomDetector')
    def test_get_custom_detector(self, mock_custom_class, detection_service):
        """Test custom detector initialization."""
        mock_detector = MagicMock()
        mock_custom_class.return_value = mock_detector
        
        detector = detection_service._get_custom_detector()
        
        assert detector == mock_detector
        assert detection_service._custom_detector == mock_detector
        mock_custom_class.assert_called_once()
    
    @patch('backend.api.detection.create_default_combiner')
    def test_get_result_combiner(self, mock_combiner_func, detection_service):
        """Test result combiner initialization."""
        mock_combiner = MagicMock()
        mock_combiner_func.return_value = mock_combiner
        
        combiner = detection_service._get_result_combiner()
        
        assert combiner == mock_combiner
        assert detection_service._result_combiner == mock_combiner
        mock_combiner_func.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_detect_watermark_success(self, detection_service, mock_detection_result):
        """Test successful watermark detection."""
        # Mock detectors
        mock_synthid = MagicMock()
        mock_synthid.is_available.return_value = True
        mock_synthid.detect.return_value = mock_detection_result
        detection_service._synthid_detector = mock_synthid
        
        # Create request
        request = DetectionRequest(
            text="Test text for watermark detection",
            detection_methods=["synthid"],
            include_metadata=True
        )
        
        # Mock background tasks
        background_tasks = MagicMock()
        
        # Perform detection
        response = await detection_service.detect_watermark(request, background_tasks)
        
        # Verify response
        assert isinstance(response, DetectionResponse)
        assert response.confidence_score == 0.8
        assert response.is_watermarked is True
        assert response.model_identified == "google/gemma-2-2b"
        assert "synthid" in response.detection_methods_used
        assert response.processing_time_ms > 0
        
        # Verify statistics updated
        assert detection_service.total_detections == 1
        assert detection_service.successful_detections == 1
    
    @pytest.mark.asyncio
    async def test_detect_watermark_empty_text(self, detection_service):
        """Test detection with empty text."""
        request = DetectionRequest(
            text="",
            detection_methods=["synthid"]
        )
        
        background_tasks = MagicMock()
        
        with pytest.raises(Exception):  # Should raise HTTPException
            await detection_service.detect_watermark(request, background_tasks)
    
    @pytest.mark.asyncio
    async def test_detect_watermark_multiple_detectors(self, detection_service, mock_detection_result):
        """Test detection with multiple detectors."""
        # Mock SynthID detector
        mock_synthid = MagicMock()
        mock_synthid.is_available.return_value = True
        mock_synthid.detect.return_value = mock_detection_result
        detection_service._synthid_detector = mock_synthid
        
        # Mock custom detector
        custom_result = DetectionResult(
            confidence_score=0.6,
            is_watermarked=True,
            model_identified=None,
            detection_method=DetectionMethod.CUSTOM.value,
            metadata={"custom": "data"},
            processing_time_ms=50
        )
        
        mock_custom = MagicMock()
        mock_custom.is_available.return_value = True
        mock_custom.detect.return_value = custom_result
        detection_service._custom_detector = mock_custom
        
        # Mock combiner
        combined_result = DetectionResult(
            confidence_score=0.75,
            is_watermarked=True,
            model_identified="google/gemma-2-2b",
            detection_method=DetectionMethod.COMBINED.value,
            metadata={"combined": "data"},
            processing_time_ms=150
        )
        
        mock_combiner = MagicMock()
        mock_combiner.combine_results.return_value = combined_result
        detection_service._result_combiner = mock_combiner
        
        # Create request
        request = DetectionRequest(
            text="Test text for detection",
            detection_methods=["synthid", "custom"],
            include_metadata=True
        )
        
        background_tasks = MagicMock()
        
        # Perform detection
        response = await detection_service.detect_watermark(request, background_tasks)
        
        # Verify response
        assert response.confidence_score == 0.75
        assert response.is_watermarked is True
        assert len(response.detection_methods_used) == 2
        
        # Verify combiner was called
        mock_combiner.combine_results.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_detect_batch_success(self, detection_service, mock_detection_result):
        """Test successful batch detection."""
        # Mock detector
        mock_synthid = MagicMock()
        mock_synthid.is_available.return_value = True
        mock_synthid.detect.return_value = mock_detection_result
        detection_service._synthid_detector = mock_synthid
        
        # Create batch request
        from backend.models.schemas import BatchRequest
        request = BatchRequest(
            texts=["Text 1", "Text 2", "Text 3"],
            detection_methods=["synthid"],
            include_metadata=False
        )
        
        background_tasks = MagicMock()
        
        # Perform batch detection
        response = await detection_service.detect_batch(request, background_tasks)
        
        # Verify response
        assert len(response.results) == 3
        assert response.total_processed == 3
        assert response.batch_stats["total_items"] == 3
        assert response.batch_stats["successful_items"] >= 0
        assert response.processing_time_ms > 0
    
    def test_get_service_stats(self, detection_service):
        """Test service statistics retrieval."""
        # Set some test values
        detection_service.total_detections = 10
        detection_service.successful_detections = 8
        detection_service.failed_detections = 2
        
        stats = detection_service.get_service_stats()
        
        assert stats["total_detections"] == 10
        assert stats["successful_detections"] == 8
        assert stats["failed_detections"] == 2
        assert stats["success_rate"] == 0.8
        assert "synthid_available" in stats
        assert "custom_available" in stats


class TestDetectionAPI:
    """Test cases for detection API endpoints."""
    
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
        """Create mock detection service."""
        service = MagicMock()
        service.detect_watermark = AsyncMock()
        service.detect_batch = AsyncMock()
        service.get_service_stats.return_value = {
            "total_detections": 0,
            "successful_detections": 0,
            "failed_detections": 0,
            "success_rate": 0.0,
            "synthid_available": True,
            "custom_available": True
        }
        return service
    
    def test_detect_endpoint_success(self, client, mock_service):
        """Test successful detection endpoint."""
        # Mock response
        mock_response = DetectionResponse(
            confidence_score=0.8,
            is_watermarked=True,
            model_identified="google/gemma-2-2b",
            detection_methods_used=["synthid"],
            analysis_metadata={},
            processing_time_ms=100
        )
        mock_service.detect_watermark.return_value = mock_response
        
        # Patch service dependency
        with patch('backend.api.detection.get_detection_service', return_value=mock_service):
            response = client.post(
                "/api/detect",
                json={
                    "text": "Test text for detection",
                    "detection_methods": ["synthid"],
                    "include_metadata": True
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["confidence_score"] == 0.8
        assert data["is_watermarked"] is True
        assert data["model_identified"] == "google/gemma-2-2b"
    
    def test_detect_endpoint_invalid_request(self, client, mock_service):
        """Test detection endpoint with invalid request."""
        with patch('backend.api.detection.get_detection_service', return_value=mock_service):
            response = client.post(
                "/api/detect",
                json={
                    "text": "",  # Empty text
                    "detection_methods": ["synthid"]
                }
            )
        
        # Should return error (exact status code depends on validation)
        assert response.status_code >= 400
    
    def test_batch_endpoint_success(self, client, mock_service):
        """Test successful batch detection endpoint."""
        from backend.models.schemas import BatchResponse
        
        # Mock response
        mock_response = BatchResponse(
            results=[],
            batch_stats={"total_items": 2, "successful_items": 2},
            total_processed=2,
            processing_time_ms=200
        )
        mock_service.detect_batch.return_value = mock_response
        
        with patch('backend.api.detection.get_detection_service', return_value=mock_service):
            response = client.post(
                "/api/analyze-batch",
                json={
                    "texts": ["Text 1", "Text 2"],
                    "detection_methods": ["synthid"],
                    "include_metadata": False
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_processed"] == 2
        assert data["processing_time_ms"] == 200
    
    def test_history_endpoint(self, client, mock_service):
        """Test history retrieval endpoint."""
        with patch('backend.api.detection.get_detection_service', return_value=mock_service):
            response = client.get("/api/history")
        
        assert response.status_code == 200
        data = response.json()
        assert "records" in data
        assert "total_count" in data
        assert "page" in data
        assert "page_size" in data
    
    def test_stats_endpoint(self, client, mock_service):
        """Test statistics endpoint."""
        with patch('backend.api.detection.get_detection_service', return_value=mock_service):
            response = client.get("/api/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_detections" in data
        assert "successful_detections" in data
        assert "synthid_available" in data
        assert "custom_available" in data


if __name__ == "__main__":
    pytest.main([__file__])