"""Custom exceptions for milvus-fake-data."""


class MilvusFakeDataError(Exception):
    """Base exception for milvus-fake-data."""

    pass


class SchemaError(MilvusFakeDataError):
    """Raised when there's an issue with the schema format or content."""

    pass


class UnsupportedFieldTypeError(MilvusFakeDataError):
    """Raised when an unsupported field type is encountered."""

    pass


class GenerationError(MilvusFakeDataError):
    """Raised when there's an error during data generation."""

    pass


class PrimaryKeyException(MilvusFakeDataError):
    """Raised when there's an issue with primary key configuration."""

    pass
