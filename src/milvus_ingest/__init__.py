"""Milvus Ingest package.
High-performance data ingestion tool for Milvus vector database.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

from .exceptions import (
    GenerationError,
    MilvusIngestError,
    SchemaError,
    UnsupportedFieldTypeError,
)
from .generator import generate_mock_data

__all__ = [
    "__version__",
    "generate_mock_data",
    "MilvusIngestError",
    "SchemaError",
    "UnsupportedFieldTypeError",
    "GenerationError",
]
