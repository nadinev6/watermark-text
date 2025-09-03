"""
Database operations for the AI Watermark Detection Tool.

This module provides high-level database operations for managing
analysis results, test datasets, and application data.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta

from .connection import DatabaseManager, get_database
from .models import (
    AnalysisResultModel,
    TestDatasetModel,
    DatasetCollectionModel,
    AppSettingModel
)
from models.schemas import DetectionResponse, TestSample

logger = logging.getLogger(__name__)


class AnalysisResultOperations:
    """
    Database operations for analysis results.
    
    This class provides methods for storing, retrieving, and managing
    watermark detection analysis results in the database.
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize analysis result operations.
        
        Args:
            db_manager (Optional[DatabaseManager]): Database manager instance
        """
        self.db_manager = db_manager
    
    async def _get_db(self) -> DatabaseManager:
        """Get database manager instance."""
        if self.db_manager is None:
            self.db_manager = await get_database()
        return self.db_manager
    
    async def save_analysis_result(
        self,
        input_text: str,
        response: DetectionResponse,
        user_feedback: Optional[str] = None
    ) -> str:
        """
        Save an analysis result to the database.
        
        Args:
            input_text (str): Original input text
            response (DetectionResponse): Detection response
            user_feedback (Optional[str]): Optional user feedback
            
        Returns:
            str: ID of the saved analysis result
            
        Raises:
            Exception: If save operation fails
        """
        try:
            db = await self._get_db()
            
            # Create model from response
            model = AnalysisResultModel.from_detection_response(
                input_text, response, user_feedback
            )
            
            # Convert to database format
            data = model.to_dict()
            
            # Insert into database
            query = """
            INSERT INTO analysis_results (
                id, timestamp, input_text, input_text_hash, confidence_score,
                is_watermarked, model_identified, detection_methods,
                analysis_metadata, processing_time_ms, user_feedback
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            params = (
                data['id'], data['timestamp'], data['input_text'],
                data['input_text_hash'], data['confidence_score'],
                data['is_watermarked'], data['model_identified'],
                data['detection_methods'], data['analysis_metadata'],
                data['processing_time_ms'], data['user_feedback']
            )
            
            await db.execute_update(query, params)
            
            logger.info(f"Saved analysis result: {model.id}")
            return model.id
            
        except Exception as e:
            logger.error(f"Failed to save analysis result: {e}")
            raise
    
    async def get_analysis_result(self, result_id: str) -> Optional[AnalysisResultModel]:
        """
        Retrieve an analysis result by ID.
        
        Args:
            result_id (str): Analysis result ID
            
        Returns:
            Optional[AnalysisResultModel]: Analysis result or None if not found
        """
        try:
            db = await self._get_db()
            
            query = "SELECT * FROM analysis_results WHERE id = ?"
            results = await db.execute_query(query, (result_id,))
            
            if results:
                return AnalysisResultModel.from_dict(results[0])
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get analysis result {result_id}: {e}")
            return None
    
    async def get_analysis_history(
        self,
        page: int = 1,
        page_size: int = 20,
        watermarked_filter: Optional[bool] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Tuple[List[AnalysisResultModel], int]:
        """
        Retrieve analysis history with pagination and filtering.
        
        Args:
            page (int): Page number (1-based)
            page_size (int): Number of results per page
            watermarked_filter (Optional[bool]): Filter by watermarked status
            start_date (Optional[datetime]): Filter by start date
            end_date (Optional[datetime]): Filter by end date
            
        Returns:
            Tuple[List[AnalysisResultModel], int]: Results and total count
        """
        try:
            db = await self._get_db()
            
            # Build WHERE clause
            where_conditions = []
            params = []
            
            if watermarked_filter is not None:
                where_conditions.append("is_watermarked = ?")
                params.append(watermarked_filter)
            
            if start_date:
                where_conditions.append("timestamp >= ?")
                params.append(start_date.isoformat())
            
            if end_date:
                where_conditions.append("timestamp <= ?")
                params.append(end_date.isoformat())
            
            where_clause = ""
            if where_conditions:
                where_clause = "WHERE " + " AND ".join(where_conditions)
            
            # Get total count
            count_query = f"SELECT COUNT(*) as count FROM analysis_results {where_clause}"
            count_result = await db.execute_query(count_query, tuple(params))
            total_count = count_result[0]['count'] if count_result else 0
            
            # Get paginated results
            offset = (page - 1) * page_size
            query = f"""
            SELECT * FROM analysis_results {where_clause}
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
            """
            
            params.extend([page_size, offset])
            results = await db.execute_query(query, tuple(params))
            
            # Convert to models
            models = [AnalysisResultModel.from_dict(row) for row in results]
            
            return models, total_count
            
        except Exception as e:
            logger.error(f"Failed to get analysis history: {e}")
            return [], 0
    
    async def update_user_feedback(
        self,
        result_id: str,
        feedback: str
    ) -> bool:
        """
        Update user feedback for an analysis result.
        
        Args:
            result_id (str): Analysis result ID
            feedback (str): User feedback text
            
        Returns:
            bool: True if update was successful
        """
        try:
            db = await self._get_db()
            
            query = "UPDATE analysis_results SET user_feedback = ? WHERE id = ?"
            affected_rows = await db.execute_update(query, (feedback, result_id))
            
            return affected_rows > 0
            
        except Exception as e:
            logger.error(f"Failed to update user feedback for {result_id}: {e}")
            return False
    
    async def delete_analysis_result(self, result_id: str) -> bool:
        """
        Delete an analysis result.
        
        Args:
            result_id (str): Analysis result ID
            
        Returns:
            bool: True if deletion was successful
        """
        try:
            db = await self._get_db()
            
            query = "DELETE FROM analysis_results WHERE id = ?"
            affected_rows = await db.execute_update(query, (result_id,))
            
            return affected_rows > 0
            
        except Exception as e:
            logger.error(f"Failed to delete analysis result {result_id}: {e}")
            return False
    
    async def get_analysis_statistics(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get analysis statistics for the specified period.
        
        Args:
            days (int): Number of days to include in statistics
            
        Returns:
            Dict[str, Any]: Analysis statistics
        """
        try:
            db = await self._get_db()
            
            # Calculate date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # Get basic statistics
            stats_query = """
            SELECT 
                COUNT(*) as total_analyses,
                COUNT(CASE WHEN is_watermarked = 1 THEN 1 END) as watermarked_count,
                COUNT(CASE WHEN is_watermarked = 0 THEN 1 END) as clean_count,
                AVG(confidence_score) as avg_confidence,
                AVG(processing_time_ms) as avg_processing_time,
                MIN(timestamp) as earliest_analysis,
                MAX(timestamp) as latest_analysis
            FROM analysis_results 
            WHERE timestamp >= ? AND timestamp <= ?
            """
            
            stats_result = await db.execute_query(
                stats_query, 
                (start_date.isoformat(), end_date.isoformat())
            )
            
            stats = stats_result[0] if stats_result else {}
            
            # Get model distribution
            model_query = """
            SELECT model_identified, COUNT(*) as count
            FROM analysis_results 
            WHERE timestamp >= ? AND timestamp <= ? AND model_identified IS NOT NULL
            GROUP BY model_identified
            ORDER BY count DESC
            """
            
            model_results = await db.execute_query(
                model_query,
                (start_date.isoformat(), end_date.isoformat())
            )
            
            model_distribution = {row['model_identified']: row['count'] for row in model_results}
            
            # Get detection method usage
            method_query = """
            SELECT detection_methods, COUNT(*) as count
            FROM analysis_results 
            WHERE timestamp >= ? AND timestamp <= ?
            GROUP BY detection_methods
            ORDER BY count DESC
            """
            
            method_results = await db.execute_query(
                method_query,
                (start_date.isoformat(), end_date.isoformat())
            )
            
            return {
                "period_days": days,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "total_analyses": stats.get('total_analyses', 0),
                "watermarked_count": stats.get('watermarked_count', 0),
                "clean_count": stats.get('clean_count', 0),
                "watermarked_ratio": (
                    stats.get('watermarked_count', 0) / max(1, stats.get('total_analyses', 1))
                ),
                "avg_confidence": stats.get('avg_confidence', 0.0),
                "avg_processing_time_ms": stats.get('avg_processing_time', 0.0),
                "earliest_analysis": stats.get('earliest_analysis'),
                "latest_analysis": stats.get('latest_analysis'),
                "model_distribution": model_distribution,
                "method_usage": {row['detection_methods']: row['count'] for row in method_results}
            }
            
        except Exception as e:
            logger.error(f"Failed to get analysis statistics: {e}")
            return {}


class TestDatasetOperations:
    """
    Database operations for test datasets.
    
    This class provides methods for storing, retrieving, and managing
    test dataset samples and collections in the database.
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize test dataset operations.
        
        Args:
            db_manager (Optional[DatabaseManager]): Database manager instance
        """
        self.db_manager = db_manager
    
    async def _get_db(self) -> DatabaseManager:
        """Get database manager instance."""
        if self.db_manager is None:
            self.db_manager = await get_database()
        return self.db_manager
    
    async def save_test_samples(
        self,
        samples: List[TestSample],
        collection_name: Optional[str] = None,
        collection_description: Optional[str] = None
    ) -> Tuple[str, List[str]]:
        """
        Save test samples to the database.
        
        Args:
            samples (List[TestSample]): Test samples to save
            collection_name (Optional[str]): Name for the collection
            collection_description (Optional[str]): Collection description
            
        Returns:
            Tuple[str, List[str]]: Collection ID and list of sample IDs
            
        Raises:
            Exception: If save operation fails
        """
        try:
            db = await self._get_db()
            
            # Create collection if name provided
            collection_id = None
            if collection_name:
                collection = DatasetCollectionModel.create_new(
                    name=collection_name,
                    description=collection_description,
                    metadata={
                        "total_samples": len(samples),
                        "created_from": "api_generation"
                    }
                )
                
                collection_id = await self._save_collection(collection)
            
            # Save individual samples
            sample_ids = []
            
            async with db.transaction() as conn:
                for i, sample in enumerate(samples):
                    # Create model from sample
                    model = TestDatasetModel.from_test_sample(
                        sample,
                        name=f"sample_{i+1}" if collection_name else None
                    )
                    
                    # Save sample
                    sample_id = await self._save_dataset_sample(model, conn)
                    sample_ids.append(sample_id)
                    
                    # Link to collection if exists
                    if collection_id:
                        await self._add_sample_to_collection(
                            collection_id, sample_id, i, conn
                        )
                
                # Update collection counts
                if collection_id:
                    watermarked_count = sum(1 for s in samples if s.is_watermarked)
                    clean_count = len(samples) - watermarked_count
                    
                    await self._update_collection_counts(
                        collection_id, watermarked_count, clean_count, conn
                    )
            
            logger.info(f"Saved {len(samples)} test samples to collection {collection_id}")
            return collection_id or "", sample_ids
            
        except Exception as e:
            logger.error(f"Failed to save test samples: {e}")
            raise
    
    async def _save_collection(self, collection: DatasetCollectionModel) -> str:
        """Save a dataset collection."""
        db = await self._get_db()
        
        data = collection.to_dict()
        
        query = """
        INSERT INTO dataset_collections (
            id, name, description, total_samples, watermarked_count,
            clean_count, collection_metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            data['id'], data['name'], data['description'],
            data['total_samples'], data['watermarked_count'],
            data['clean_count'], data['collection_metadata']
        )
        
        await db.execute_update(query, params)
        return collection.id
    
    async def _save_dataset_sample(
        self,
        model: TestDatasetModel,
        conn=None
    ) -> str:
        """Save a dataset sample."""
        if conn is None:
            db = await self._get_db()
            async with db.get_connection() as conn:
                return await self._save_dataset_sample(model, conn)
        
        data = model.to_dict()
        
        query = """
        INSERT INTO test_datasets (
            id, name, description, text_content, is_watermarked,
            expected_score, source_model, generation_params, dataset_metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            data['id'], data['name'], data['description'],
            data['text_content'], data['is_watermarked'],
            data['expected_score'], data['source_model'],
            data['generation_params'], data['dataset_metadata']
        )
        
        await conn.execute(query, params)
        return model.id
    
    async def _add_sample_to_collection(
        self,
        collection_id: str,
        sample_id: str,
        order: int,
        conn=None
    ) -> None:
        """Add a sample to a collection."""
        if conn is None:
            db = await self._get_db()
            async with db.get_connection() as conn:
                return await self._add_sample_to_collection(
                    collection_id, sample_id, order, conn
                )
        
        query = """
        INSERT INTO dataset_collection_items (collection_id, dataset_id, item_order)
        VALUES (?, ?, ?)
        """
        
        await conn.execute(query, (collection_id, sample_id, order))
    
    async def _update_collection_counts(
        self,
        collection_id: str,
        watermarked_count: int,
        clean_count: int,
        conn=None
    ) -> None:
        """Update collection sample counts."""
        if conn is None:
            db = await self._get_db()
            async with db.get_connection() as conn:
                return await self._update_collection_counts(
                    collection_id, watermarked_count, clean_count, conn
                )
        
        query = """
        UPDATE dataset_collections 
        SET total_samples = ?, watermarked_count = ?, clean_count = ?
        WHERE id = ?
        """
        
        total_samples = watermarked_count + clean_count
        await conn.execute(query, (total_samples, watermarked_count, clean_count, collection_id))
    
    async def get_dataset_collections(
        self,
        page: int = 1,
        page_size: int = 20
    ) -> Tuple[List[DatasetCollectionModel], int]:
        """
        Get dataset collections with pagination.
        
        Args:
            page (int): Page number (1-based)
            page_size (int): Number of results per page
            
        Returns:
            Tuple[List[DatasetCollectionModel], int]: Collections and total count
        """
        try:
            db = await self._get_db()
            
            # Get total count
            count_query = "SELECT COUNT(*) as count FROM dataset_collections"
            count_result = await db.execute_query(count_query)
            total_count = count_result[0]['count'] if count_result else 0
            
            # Get paginated results
            offset = (page - 1) * page_size
            query = """
            SELECT * FROM dataset_collections
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """
            
            results = await db.execute_query(query, (page_size, offset))
            
            # Convert to models
            models = [DatasetCollectionModel.from_dict(row) for row in results]
            
            return models, total_count
            
        except Exception as e:
            logger.error(f"Failed to get dataset collections: {e}")
            return [], 0
    
    async def get_collection_samples(
        self,
        collection_id: str,
        page: int = 1,
        page_size: int = 50
    ) -> Tuple[List[TestDatasetModel], int]:
        """
        Get samples from a specific collection.
        
        Args:
            collection_id (str): Collection ID
            page (int): Page number (1-based)
            page_size (int): Number of results per page
            
        Returns:
            Tuple[List[TestDatasetModel], int]: Samples and total count
        """
        try:
            db = await self._get_db()
            
            # Get total count
            count_query = """
            SELECT COUNT(*) as count 
            FROM dataset_collection_items 
            WHERE collection_id = ?
            """
            count_result = await db.execute_query(count_query, (collection_id,))
            total_count = count_result[0]['count'] if count_result else 0
            
            # Get paginated results
            offset = (page - 1) * page_size
            query = """
            SELECT td.* FROM test_datasets td
            JOIN dataset_collection_items dci ON td.id = dci.dataset_id
            WHERE dci.collection_id = ?
            ORDER BY dci.item_order ASC
            LIMIT ? OFFSET ?
            """
            
            results = await db.execute_query(query, (collection_id, page_size, offset))
            
            # Convert to models
            models = [TestDatasetModel.from_dict(row) for row in results]
            
            return models, total_count
            
        except Exception as e:
            logger.error(f"Failed to get collection samples for {collection_id}: {e}")
            return [], 0