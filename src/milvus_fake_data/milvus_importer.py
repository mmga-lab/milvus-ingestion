"""Import generated data directly to Milvus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from .logging_config import get_logger
from .rich_display import display_error


class MilvusImporter:
    """Handle importing data to Milvus."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        user: str = "",
        password: str = "",
        db_name: str = "default",
        secure: bool = False,
    ):
        """Initialize Milvus connection.
        
        Args:
            host: Milvus server host
            port: Milvus server port
            user: Username for authentication
            password: Password for authentication
            db_name: Database name
            secure: Use secure connection (TLS)
        """
        self.logger = get_logger(__name__)
        self.host = host
        self.port = port
        self.db_name = db_name

        # Create connection alias
        self.alias = f"milvus_{host}_{port}"

        try:
            # Connect to Milvus
            connections.connect(
                alias=self.alias,
                host=host,
                port=port,
                user=user,
                password=password,
                db_name=db_name,
                secure=secure,
            )
            self.logger.info(
                f"Connected to Milvus at {host}:{port}",
                extra={"db_name": db_name}
            )
        except Exception as e:
            self.logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def import_data(
        self,
        data_path: Path,
        collection_name: str | None = None,
        drop_if_exists: bool = False,
        create_index: bool = True,
        batch_size: int = 10000,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """Import data from generated files to Milvus.
        
        Args:
            data_path: Path to the data directory containing parquet files and meta.json
            collection_name: Override collection name from meta.json
            drop_if_exists: Drop collection if it already exists
            create_index: Create index on vector fields after import
            batch_size: Batch size for inserting data
            show_progress: Show progress bar
            
        Returns:
            Dictionary with import statistics
        """
        if not data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")

        # Load metadata
        meta_path = data_path / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json not found in {data_path}")

        with open(meta_path) as f:
            metadata = json.load(f)

        # Get collection name
        final_collection_name = collection_name or metadata["collection_name"]

        # Check if collection exists
        if utility.has_collection(final_collection_name, using=self.alias):
            if drop_if_exists:
                utility.drop_collection(final_collection_name, using=self.alias)
                self.logger.info(f"Dropped existing collection: {final_collection_name}")
            else:
                raise ValueError(
                    f"Collection '{final_collection_name}' already exists. "
                    "Use --drop-if-exists to recreate it."
                )

        # Create collection schema
        schema = self._create_schema_from_metadata(metadata)

        # Create collection
        collection = Collection(
            name=final_collection_name,
            schema=schema,
            using=self.alias,
        )
        self.logger.info(f"Created collection: {final_collection_name}")

        # Find all parquet files
        parquet_files = sorted(data_path.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {data_path}")

        # Import data from all parquet files
        total_inserted = 0
        failed_batches = []

        for parquet_file in parquet_files:
            self.logger.info(f"Processing {parquet_file.name}")

            # Read parquet file
            df = pd.read_parquet(parquet_file)
            total_rows = len(df)

            if show_progress:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                ) as progress:
                    task = progress.add_task(
                        f"Importing {parquet_file.name}",
                        total=total_rows
                    )

                    # Insert in batches
                    for i in range(0, total_rows, batch_size):
                        batch_df = df.iloc[i:i+batch_size]
                        try:
                            # Convert DataFrame to list of lists for Milvus
                            data = []
                            for column in batch_df.columns:
                                column_data = batch_df[column].tolist()
                                data.append(column_data)

                            collection.insert(data)
                            total_inserted += len(batch_df)
                            progress.update(task, advance=len(batch_df))
                        except Exception as e:
                            self.logger.error(f"Failed to insert batch {i//batch_size}: {e}")
                            failed_batches.append({
                                "file": parquet_file.name,
                                "batch": i//batch_size,
                                "error": str(e)
                            })
                            progress.update(task, advance=len(batch_df))
            else:
                # Insert without progress bar
                for i in range(0, total_rows, batch_size):
                    batch_df = df.iloc[i:i+batch_size]
                    try:
                        data = []
                        for column in batch_df.columns:
                            column_data = batch_df[column].tolist()
                            data.append(column_data)

                        collection.insert(data)
                        total_inserted += len(batch_df)
                    except Exception as e:
                        self.logger.error(f"Failed to insert batch {i//batch_size}: {e}")
                        failed_batches.append({
                            "file": parquet_file.name,
                            "batch": i//batch_size,
                            "error": str(e)
                        })

        # Flush data
        collection.flush()
        self.logger.info("Data flushed to disk")

        # Create indexes if requested
        index_info = []
        if create_index:
            index_info = self._create_indexes(collection, metadata)

        # Load collection
        collection.load()
        self.logger.info(f"Collection '{final_collection_name}' loaded")

        return {
            "collection_name": final_collection_name,
            "total_inserted": total_inserted,
            "failed_batches": failed_batches,
            "indexes_created": index_info,
            "collection_loaded": True,
        }

    def _create_schema_from_metadata(self, metadata: dict[str, Any]) -> CollectionSchema:
        """Create Milvus collection schema from metadata."""
        fields = []

        for field_info in metadata["fields"]:
            field_name = field_info["name"]
            field_type = field_info["type"]

            # Map field type to Milvus DataType
            milvus_type = self._get_milvus_datatype(field_type)

            # Create field schema
            field_schema = FieldSchema(
                name=field_name,
                dtype=milvus_type,
                is_primary=field_info.get("is_primary", False),
                auto_id=field_info.get("auto_id", False),
                max_length=field_info.get("max_length", 65535) if field_type in ["VarChar", "String"] else None,
                dim=field_info.get("dim") if "Vector" in field_type else None,
                max_capacity=field_info.get("max_capacity") if field_type == "Array" else None,
                element_type=DataType[field_info.get("element_type")] if field_type == "Array" else None,
            )

            # Remove None values from kwargs
            kwargs = {k: v for k, v in field_schema.__dict__.items() if v is not None}
            fields.append(FieldSchema(**kwargs))

        return CollectionSchema(
            fields=fields,
            description=metadata.get("description", "")
        )

    def _get_milvus_datatype(self, field_type: str) -> DataType:
        """Map field type string to Milvus DataType."""
        type_mapping = {
            "Bool": DataType.BOOL,
            "Int8": DataType.INT8,
            "Int16": DataType.INT16,
            "Int32": DataType.INT32,
            "Int64": DataType.INT64,
            "Float": DataType.FLOAT,
            "Double": DataType.DOUBLE,
            "String": DataType.VARCHAR,
            "VarChar": DataType.VARCHAR,
            "JSON": DataType.JSON,
            "Array": DataType.ARRAY,
            "FloatVector": DataType.FLOAT_VECTOR,
            "BinaryVector": DataType.BINARY_VECTOR,
            "Float16Vector": DataType.FLOAT16_VECTOR,
            "BFloat16Vector": DataType.BFLOAT16_VECTOR,
            "SparseFloatVector": DataType.SPARSE_FLOAT_VECTOR,
        }

        if field_type not in type_mapping:
            raise ValueError(f"Unknown field type: {field_type}")

        return type_mapping[field_type]

    def _create_indexes(self, collection: Collection, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        """Create indexes on vector fields."""
        index_info = []

        for field_info in metadata["fields"]:
            if "Vector" in field_info["type"] and field_info["type"] != "SparseFloatVector":
                field_name = field_info["name"]

                # Default index parameters
                index_params = {
                    "metric_type": "L2",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 1024}
                }

                # For small dimensions, use FLAT index
                if field_info.get("dim", 0) <= 32:
                    index_params = {
                        "metric_type": "L2",
                        "index_type": "FLAT",
                        "params": {}
                    }

                try:
                    collection.create_index(
                        field_name=field_name,
                        index_params=index_params
                    )
                    self.logger.info(f"Created index on field: {field_name}")
                    index_info.append({
                        "field": field_name,
                        "index_type": index_params["index_type"],
                        "metric_type": index_params["metric_type"]
                    })
                except Exception as e:
                    self.logger.error(f"Failed to create index on {field_name}: {e}")

        return index_info

    def close(self):
        """Close Milvus connection."""
        try:
            connections.disconnect(self.alias)
            self.logger.info("Disconnected from Milvus")
        except Exception as e:
            self.logger.error(f"Error disconnecting from Milvus: {e}")

    def test_connection(self) -> bool:
        """Test Milvus connection."""
        try:
            # Try to list collections
            collections = utility.list_collections(using=self.alias)
            self.logger.info(
                f"Successfully connected to Milvus. Found {len(collections)} collections."
            )
            return True
        except Exception as e:
            display_error(f"Failed to connect to Milvus: {e}")
            return False
