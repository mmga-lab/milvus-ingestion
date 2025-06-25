"""Milvus Fake Data package.
Generate mock data based on Milvus collection schema.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

from .exceptions import (
    GenerationError,
    MilvusFakeDataError,
    SchemaError,
    UnsupportedFieldTypeError,
)
from .generator import generate_mock_data

__all__ = [
    "__version__",
    "generate_mock_data",
    "MilvusFakeDataError",
    "SchemaError",
    "UnsupportedFieldTypeError",
    "GenerationError",
]
