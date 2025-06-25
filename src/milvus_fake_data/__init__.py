"""Milvus Fake Data package.
Generate mock data based on Milvus collection schema.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__: str = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]
