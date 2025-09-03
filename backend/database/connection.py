"""
Database connection management and configuration.

This module provides SQLite database connection management with
connection pooling, transaction handling, and migration support.
"""

import sqlite3
import logging
import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path
from contextlib import asynccontextmanager
import aiosqlite
import json
from datetime import datetime, timedelta

from utils.config import get_config

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages SQLite database connections and operations.
    
    This class provides connection pooling, transaction management,
    and database schema initialization for the application.
    """
    
    def __init__(self):
        """Initialize the database manager."""
        self.config = get_config()
        self.database_url = self.config.database.database_url
        
        # Extract database path from URL
        if self.database_url.startswith("sqlite:///"):
            self.db_path = self.database_url[10:]  # Remove "sqlite:///"
        else:
            self.db_path = self.database_url
        
        # Ensure database directory exists
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        # Connection pool (simple implementation)
        self._connection_pool: List[aiosqlite.Connection] = []
        self._pool_size = 5
        self._initialized = False
        
        logger.info(f"Database manager initialized with path: {self.db_path}")
    
    async def initialize(self) -> None:
        """
        Initialize database schema and connection pool.
        
        This method creates the database tables if they don't exist
        and sets up the connection pool for efficient operations.
        """
        if self._initialized:
            return
        
        try:
            # Create database file if it doesn't exist
            Path(self.db_path).touch(exist_ok=True)
            
            # Initialize schema
            await self._create_schema()
            
            # Initialize connection pool
            await self._initialize_pool()
            
            self._initialized = True
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    async def _create_schema(self) -> None:
        """
        Create database schema with all required tables.
        
        This method creates the database tables for storing analysis
        results, test datasets, and other application data.
        """
        schema_sql = """
        -- Analysis results table
        CREATE TABLE IF NOT EXISTS analysis_results (
            id TEXT PRIMARY KEY,
            timestamp DATETIME NOT NULL,
            input_text TEXT NOT NULL,
            input_text_hash TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            is_watermarked BOOLEAN NOT NULL,
            model_identified TEXT,
            detection_methods TEXT NOT NULL,  -- JSON array
            analysis_metadata TEXT,           -- JSON object
            processing_time_ms INTEGER NOT NULL,
            user_feedback TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Test datasets table
        CREATE TABLE IF NOT EXISTS test_datasets (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            text_content TEXT NOT NULL,
            is_watermarked BOOLEAN NOT NULL,
            expected_score REAL NOT NULL,
            source_model TEXT NOT NULL,
            generation_params TEXT,           -- JSON object
            dataset_metadata TEXT,            -- JSON object
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Dataset collections table (for grouping test samples)
        CREATE TABLE IF NOT EXISTS dataset_collections (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            description TEXT,
            total_samples INTEGER NOT NULL DEFAULT 0,
            watermarked_count INTEGER NOT NULL DEFAULT 0,
            clean_count INTEGER NOT NULL DEFAULT 0,
            collection_metadata TEXT,         -- JSON object
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Link table for datasets and collections
        CREATE TABLE IF NOT EXISTS dataset_collection_items (
            collection_id TEXT NOT NULL,
            dataset_id TEXT NOT NULL,
            item_order INTEGER,
            PRIMARY KEY (collection_id, dataset_id),
            FOREIGN KEY (collection_id) REFERENCES dataset_collections(id) ON DELETE CASCADE,
            FOREIGN KEY (dataset_id) REFERENCES test_datasets(id) ON DELETE CASCADE
        );
        
        -- Application settings table
        CREATE TABLE IF NOT EXISTS app_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            description TEXT,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Create indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_analysis_results_timestamp ON analysis_results(timestamp);
        CREATE INDEX IF NOT EXISTS idx_analysis_results_hash ON analysis_results(input_text_hash);
        CREATE INDEX IF NOT EXISTS idx_analysis_results_watermarked ON analysis_results(is_watermarked);
        CREATE INDEX IF NOT EXISTS idx_test_datasets_watermarked ON test_datasets(is_watermarked);
        CREATE INDEX IF NOT EXISTS idx_test_datasets_model ON test_datasets(source_model);
        CREATE INDEX IF NOT EXISTS idx_dataset_collections_name ON dataset_collections(name);
        
        -- Create triggers for updating timestamps
        CREATE TRIGGER IF NOT EXISTS update_analysis_results_timestamp 
        AFTER UPDATE ON analysis_results
        BEGIN
            UPDATE analysis_results SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
        END;
        
        CREATE TRIGGER IF NOT EXISTS update_test_datasets_timestamp 
        AFTER UPDATE ON test_datasets
        BEGIN
            UPDATE test_datasets SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
        END;
        
        CREATE TRIGGER IF NOT EXISTS update_dataset_collections_timestamp 
        AFTER UPDATE ON dataset_collections
        BEGIN
            UPDATE dataset_collections SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
        END;
        """
        
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.executescript(schema_sql)
            await conn.commit()
            
        logger.info("Database schema created successfully")
    
    async def _initialize_pool(self) -> None:
        """Initialize connection pool with configured size."""
        for _ in range(self._pool_size):
            conn = await aiosqlite.connect(self.db_path)
            # Enable foreign key constraints
            await conn.execute("PRAGMA foreign_keys = ON")
            # Set WAL mode for better concurrency
            await conn.execute("PRAGMA journal_mode = WAL")
            self._connection_pool.append(conn)
        
        logger.info(f"Connection pool initialized with {self._pool_size} connections")
    
    @asynccontextmanager
    async def get_connection(self):
        """
        Get a database connection from the pool.
        
        This context manager provides a database connection and ensures
        proper cleanup and return to the pool.
        
        Yields:
            aiosqlite.Connection: Database connection
        """
        if not self._initialized:
            await self.initialize()
        
        if not self._connection_pool:
            # Create new connection if pool is empty
            conn = await aiosqlite.connect(self.db_path)
            await conn.execute("PRAGMA foreign_keys = ON")
            await conn.execute("PRAGMA journal_mode = WAL")
        else:
            conn = self._connection_pool.pop()
        
        try:
            yield conn
        finally:
            # Return connection to pool if there's space
            if len(self._connection_pool) < self._pool_size:
                self._connection_pool.append(conn)
            else:
                await conn.close()
    
    @asynccontextmanager
    async def transaction(self):
        """
        Execute operations within a database transaction.
        
        This context manager provides transaction support with
        automatic commit/rollback handling.
        
        Yields:
            aiosqlite.Connection: Database connection with transaction
        """
        async with self.get_connection() as conn:
            try:
                await conn.execute("BEGIN")
                yield conn
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
    
    async def execute_query(
        self,
        query: str,
        params: Optional[tuple] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a SELECT query and return results as dictionaries.
        
        Args:
            query (str): SQL query to execute
            params (Optional[tuple]): Query parameters
            
        Returns:
            List[Dict[str, Any]]: Query results as list of dictionaries
        """
        async with self.get_connection() as conn:
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(query, params or ())
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
    
    async def execute_update(
        self,
        query: str,
        params: Optional[tuple] = None
    ) -> int:
        """
        Execute an INSERT, UPDATE, or DELETE query.
        
        Args:
            query (str): SQL query to execute
            params (Optional[tuple]): Query parameters
            
        Returns:
            int: Number of affected rows
        """
        async with self.get_connection() as conn:
            cursor = await conn.execute(query, params or ())
            await conn.commit()
            return cursor.rowcount
    
    async def execute_many(
        self,
        query: str,
        params_list: List[tuple]
    ) -> int:
        """
        Execute a query multiple times with different parameters.
        
        Args:
            query (str): SQL query to execute
            params_list (List[tuple]): List of parameter tuples
            
        Returns:
            int: Total number of affected rows
        """
        async with self.transaction() as conn:
            cursor = await conn.executemany(query, params_list)
            return cursor.rowcount
    
    async def cleanup_old_records(self) -> Dict[str, int]:
        """
        Clean up old records based on configuration settings.
        
        This method removes old analysis results and test datasets
        based on the configured retention policies.
        
        Returns:
            Dict[str, int]: Number of records cleaned up by table
        """
        cleanup_stats = {}
        
        try:
            # Clean up old analysis results
            max_records = self.config.database.max_history_records
            
            # Get count of records to delete
            count_query = """
            SELECT COUNT(*) as count FROM analysis_results 
            WHERE id NOT IN (
                SELECT id FROM analysis_results 
                ORDER BY timestamp DESC 
                LIMIT ?
            )
            """
            
            result = await self.execute_query(count_query, (max_records,))
            records_to_delete = result[0]['count'] if result else 0
            
            if records_to_delete > 0:
                # Delete old records
                delete_query = """
                DELETE FROM analysis_results 
                WHERE id NOT IN (
                    SELECT id FROM analysis_results 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                )
                """
                
                deleted_count = await self.execute_update(delete_query, (max_records,))
                cleanup_stats['analysis_results'] = deleted_count
                
                logger.info(f"Cleaned up {deleted_count} old analysis results")
            
            # Clean up orphaned test datasets (not in any collection)
            orphan_query = """
            DELETE FROM test_datasets 
            WHERE id NOT IN (
                SELECT DISTINCT dataset_id FROM dataset_collection_items
            ) AND created_at < datetime('now', '-30 days')
            """
            
            orphaned_count = await self.execute_update(orphan_query)
            if orphaned_count > 0:
                cleanup_stats['orphaned_datasets'] = orphaned_count
                logger.info(f"Cleaned up {orphaned_count} orphaned test datasets")
            
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Cleanup operation failed: {e}")
            return cleanup_stats
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics and health information.
        
        Returns:
            Dict[str, Any]: Database statistics including table sizes and performance metrics
        """
        try:
            stats = {
                "database_path": self.db_path,
                "database_size_mb": 0,
                "tables": {},
                "indexes": {},
                "connection_pool_size": len(self._connection_pool),
                "initialized": self._initialized
            }
            
            # Get database file size
            if Path(self.db_path).exists():
                stats["database_size_mb"] = Path(self.db_path).stat().st_size / (1024 * 1024)
            
            # Get table statistics
            tables = ["analysis_results", "test_datasets", "dataset_collections", "app_settings"]
            
            for table in tables:
                count_query = f"SELECT COUNT(*) as count FROM {table}"
                result = await self.execute_query(count_query)
                stats["tables"][table] = result[0]['count'] if result else 0
            
            # Get index information
            index_query = "SELECT name, tbl_name FROM sqlite_master WHERE type = 'index'"
            indexes = await self.execute_query(index_query)
            stats["indexes"] = {idx['name']: idx['tbl_name'] for idx in indexes}
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {"error": str(e)}
    
    async def backup_database(self, backup_path: str) -> bool:
        """
        Create a backup of the database.
        
        Args:
            backup_path (str): Path for the backup file
            
        Returns:
            bool: True if backup was successful
        """
        try:
            # Ensure backup directory exists
            Path(backup_path).parent.mkdir(parents=True, exist_ok=True)
            
            async with self.get_connection() as conn:
                # Use SQLite backup API
                backup_conn = await aiosqlite.connect(backup_path)
                try:
                    await conn.backup(backup_conn)
                    logger.info(f"Database backup created: {backup_path}")
                    return True
                finally:
                    await backup_conn.close()
                    
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return False
    
    async def close(self) -> None:
        """Close all database connections and cleanup resources."""
        try:
            # Close all connections in pool
            for conn in self._connection_pool:
                await conn.close()
            
            self._connection_pool.clear()
            self._initialized = False
            
            logger.info("Database connections closed")
            
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")


# Global database manager instance
_database_manager: Optional[DatabaseManager] = None


async def get_database() -> DatabaseManager:
    """
    Get or create the global database manager instance.
    
    Returns:
        DatabaseManager: Global database manager
    """
    global _database_manager
    if _database_manager is None:
        _database_manager = DatabaseManager()
        await _database_manager.initialize()
    return _database_manager


async def initialize_database() -> None:
    """Initialize the database system."""
    await get_database()


async def cleanup_database() -> None:
    """Cleanup database resources."""
    global _database_manager
    if _database_manager is not None:
        await _database_manager.close()
        _database_manager = None