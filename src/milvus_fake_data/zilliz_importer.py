"""Import generated data directly to Zilliz Cloud."""

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
from .rich_display import display_error, display_info


class ZillizImporter:
    """Handle importing data to Zilliz Cloud."""

    def __init__(
        self,
        uri: str,
        token: str,
        db_name: str = "default",
    ):
        """Initialize Zilliz Cloud connection.
        
        Args:
            uri: Zilliz Cloud cluster endpoint (e.g., https://in03-xxx.api.gcp-us-west1.zillizcloud.com)
            token: API token for authentication
            db_name: Database name (default: "default")
        """
        self.logger = get_logger(__name__)
        self.uri = uri
        self.db_name = db_name

        # Create connection alias
        self.alias = "zilliz_cloud"

        try:
            # Connect to Zilliz Cloud
            connections.connect(
                alias=self.alias,
                uri=uri,
                token=token,
                db_name=db_name,
            )
            self.logger.info(
                "Connected to Zilliz Cloud",
                extra={"uri": uri, "db_name": db_name}
            )
        except Exception as e:
            self.logger.error(f"Failed to connect to Zilliz Cloud: {e}")
            raise

    def import_data(
        self,
        data_path: Path,
        collection_name: str | None = None,
        drop_if_exists: bool = False,
        create_index: bool = True,
        batch_size: int = 5000,  # Smaller batch size for cloud
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """Import data from generated files to Zilliz Cloud.
        
        Args:
            data_path: Path to the data directory containing parquet files and meta.json
            collection_name: Override collection name from meta.json
            drop_if_exists: Drop collection if it already exists
            create_index: Create index on vector fields after import
            batch_size: Batch size for inserting data (smaller for cloud)
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

        # Create collection with optimized settings for Zilliz Cloud
        collection = Collection(
            name=final_collection_name,
            schema=schema,
            using=self.alias,
            consistency_level="Eventually",  # Better performance for bulk import
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
                        f"Importing {parquet_file.name} to Zilliz Cloud",
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

                            # Flush more frequently for cloud
                            if (i + batch_size) % (batch_size * 10) == 0:
                                collection.flush()

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

                        # Flush more frequently for cloud
                        if (i + batch_size) % (batch_size * 10) == 0:
                            collection.flush()

                    except Exception as e:
                        self.logger.error(f"Failed to insert batch {i//batch_size}: {e}")
                        failed_batches.append({
                            "file": parquet_file.name,
                            "batch": i//batch_size,
                            "error": str(e)
                        })

        # Final flush
        collection.flush()
        self.logger.info("Data flushed to Zilliz Cloud")

        # Create indexes if requested
        index_info = []
        if create_index:
            display_info("Creating indexes (this may take a few minutes on Zilliz Cloud)...")
            index_info = self._create_indexes(collection, metadata)

        # Load collection
        display_info("Loading collection (this may take a few minutes on Zilliz Cloud)...")
        collection.load()
        self.logger.info(f"Collection '{final_collection_name}' loaded")

        return {
            "collection_name": final_collection_name,
            "total_inserted": total_inserted,
            "failed_batches": failed_batches,
            "indexes_created": index_info,
            "collection_loaded": True,
            "cluster_endpoint": self.uri,
        }

    def _create_schema_from_metadata(self, metadata: dict[str, Any]) -> CollectionSchema:
        """Create Milvus collection schema from metadata."""
        fields = []

        for field_info in metadata["fields"]:
            field_name = field_info["name"]
            field_type = field_info["type"]

            # Map field type to Milvus DataType
            milvus_type = self._get_milvus_datatype(field_type)

            # Build field schema parameters
            field_params = {
                "name": field_name,
                "dtype": milvus_type,
                "is_primary": field_info.get("is_primary", False),
                "auto_id": field_info.get("auto_id", False),
            }

            # Add type-specific parameters
            if field_type in ["VarChar", "String"]:
                field_params["max_length"] = field_info.get("max_length", 65535)
            elif "Vector" in field_type:
                field_params["dim"] = field_info["dim"]
            elif field_type == "Array":
                field_params["max_capacity"] = field_info["max_capacity"]
                field_params["element_type"] = DataType[field_info["element_type"]]

            fields.append(FieldSchema(**field_params))

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
        """Create indexes on vector fields optimized for Zilliz Cloud."""
        index_info = []

        for field_info in metadata["fields"]:
            if "Vector" in field_info["type"] and field_info["type"] != "SparseFloatVector":
                field_name = field_info["name"]
                dim = field_info.get("dim", 0)

                # Optimized index parameters for Zilliz Cloud
                if dim <= 32:
                    # Small dimensions - use FLAT
                    index_params = {
                        "metric_type": "L2",
                        "index_type": "FLAT",
                        "params": {}
                    }
                elif dim <= 256:
                    # Medium dimensions - use IVF_FLAT
                    index_params = {
                        "metric_type": "L2",
                        "index_type": "IVF_FLAT",
                        "params": {"nlist": 1024}
                    }
                else:
                    # Large dimensions - use HNSW for better performance
                    index_params = {
                        "metric_type": "L2",
                        "index_type": "HNSW",
                        "params": {"M": 16, "efConstruction": 200}
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
        """Close Zilliz Cloud connection."""
        try:
            connections.disconnect(self.alias)
            self.logger.info("Disconnected from Zilliz Cloud")
        except Exception as e:
            self.logger.error(f"Error disconnecting from Zilliz Cloud: {e}")

    def test_connection(self) -> bool:
        """Test Zilliz Cloud connection."""
        try:
            # Try to list collections
            collections = utility.list_collections(using=self.alias)
            self.logger.info(
                f"Successfully connected to Zilliz Cloud. Found {len(collections)} collections."
            )
            return True
        except Exception as e:
            display_error(f"Failed to connect to Zilliz Cloud: {e}")
            return False
