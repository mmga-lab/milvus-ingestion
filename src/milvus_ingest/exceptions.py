"""Custom exceptions for milvus-ingest."""


class MilvusIngestError(Exception):
    """Base exception for milvus-ingest."""

    pass


class SchemaError(MilvusIngestError):
    """Raised when there's an issue with the schema format or content."""

    pass


class UnsupportedFieldTypeError(MilvusIngestError):
    """Raised when an unsupported field type is encountered."""

    pass


class GenerationError(MilvusIngestError):
    """Raised when there's an error during data generation."""

    pass


class PrimaryKeyException(MilvusIngestError):
    """Raised when there's an issue with primary key configuration."""

    pass
