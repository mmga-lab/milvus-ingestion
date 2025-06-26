"""Pydantic models for schema validation with helpful error messages."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator, model_validator

if TYPE_CHECKING:
    from pydantic import ValidationInfo


class FieldType(str, Enum):
    """Supported Milvus field types."""

    BOOL = "Bool"
    INT8 = "Int8"
    INT16 = "Int16"
    INT32 = "Int32"
    INT64 = "Int64"
    FLOAT = "Float"
    DOUBLE = "Double"
    VARCHAR = "VarChar"
    STRING = "String"
    JSON = "JSON"
    ARRAY = "Array"
    FLOAT_VECTOR = "FloatVector"
    BINARY_VECTOR = "BinaryVector"
    FLOAT16_VECTOR = "Float16Vector"
    BFLOAT16_VECTOR = "BFloat16Vector"
    INT8_VECTOR = "Int8Vector"
    SPARSE_FLOAT_VECTOR = "SparseFloatVector"


class ArrayElementType(str, Enum):
    """Supported array element types."""

    BOOL = "Bool"
    INT8 = "Int8"
    INT16 = "Int16"
    INT32 = "Int32"
    INT64 = "Int64"
    FLOAT = "Float"
    DOUBLE = "Double"
    VARCHAR = "VarChar"
    STRING = "String"


class FieldSchema(BaseModel):
    """Pydantic model for field schema validation."""

    name: str = Field(..., description="Field name", min_length=1, max_length=255)
    type: FieldType = Field(..., description="Field data type")
    is_primary: bool = Field(
        default=False, description="Whether this field is the primary key"
    )
    auto_id: bool = Field(
        default=False, description="Whether to auto-generate ID values"
    )
    nullable: bool = Field(default=False, description="Whether this field can be null")

    # Numeric field constraints
    min: int | float | None = Field(
        default=None, description="Minimum value for numeric fields"
    )
    max: int | float | None = Field(
        default=None, description="Maximum value for numeric fields"
    )

    # String field constraints
    max_length: int | None = Field(
        default=None, description="Maximum length for string fields", ge=1, le=65535
    )

    # Vector field constraints
    dim: int | None = Field(
        default=None, description="Dimension for vector fields", ge=1, le=32768
    )

    # Array field constraints
    element_type: ArrayElementType | None = Field(
        default=None, description="Element type for array fields"
    )
    max_capacity: int | None = Field(
        default=None, description="Maximum capacity for array fields", ge=1, le=4096
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate field name."""
        if not v.replace("_", "").isalnum():
            raise ValueError(
                f"Field name '{v}' is invalid. Field names must contain only letters, numbers, and underscores."
            )
        return v

    @field_validator("auto_id")
    @classmethod
    def validate_auto_id(cls, v: bool, info: ValidationInfo) -> bool:
        """Validate auto_id can only be True for primary key fields."""
        if v and not info.data.get("is_primary", False):
            raise ValueError(
                "auto_id can only be True for primary key fields. Set is_primary=true first."
            )
        return v

    @model_validator(mode="after")
    def validate_field_constraints(self) -> FieldSchema:
        """Validate field-specific constraints."""
        field_type = self.type.value.upper()

        # String fields must have max_length
        if field_type in {"VARCHAR", "STRING"} and self.max_length is None:
            raise ValueError(
                f"Field '{self.name}' with type '{self.type}' must specify max_length.\n"
                f'Example: {{"name": "{self.name}", "type": "{self.type}", "max_length": 128}}'
            )

        # Vector fields must have dim
        if "VECTOR" in field_type and self.dim is None:
            raise ValueError(
                f"Field '{self.name}' with type '{self.type}' must specify dim (dimension).\n"
                f'Example: {{"name": "{self.name}", "type": "{self.type}", "dim": 128}}'
            )

        # Array fields must have element_type and max_capacity
        if field_type == "ARRAY":
            if self.element_type is None:
                raise ValueError(
                    f"Field '{self.name}' with type 'Array' must specify element_type.\n"
                    f'Example: {{"name": "{self.name}", "type": "Array", "element_type": "VarChar", "max_capacity": 10}}'
                )
            if self.max_capacity is None:
                raise ValueError(
                    f"Field '{self.name}' with type 'Array' must specify max_capacity.\n"
                    f'Example: {{"name": "{self.name}", "type": "Array", "element_type": "VarChar", "max_capacity": 10}}'
                )
            # Array of VarChar needs max_length
            if (
                self.element_type in {ArrayElementType.VARCHAR, ArrayElementType.STRING}
                and self.max_length is None
            ):
                raise ValueError(
                    f"Field '{self.name}' with Array of VarChar/String must specify max_length for elements.\n"
                    f'Example: {{"name": "{self.name}", "type": "Array", "element_type": "VarChar", "max_capacity": 10, "max_length": 50}}'
                )

        # Validate min/max constraints
        if self.min is not None and self.max is not None and self.min >= self.max:
            raise ValueError(
                f"Field '{self.name}': min value ({self.min}) must be less than max value ({self.max})"
            )

        # Check incompatible field attributes
        if not field_type.startswith(("INT", "FLOAT", "DOUBLE")) and (
            self.min is not None or self.max is not None
        ):
            raise ValueError(
                f"Field '{self.name}' with type '{self.type}' cannot have min/max constraints. "
                f"These are only valid for numeric types (Int8, Int16, Int32, Int64, Float, Double)."
            )

        if (
            field_type not in {"VARCHAR", "STRING"}
            and field_type != "ARRAY"
            and self.max_length is not None
        ):
            raise ValueError(
                f"Field '{self.name}' with type '{self.type}' cannot have max_length. "
                f"This is only valid for VarChar, String, and Array fields."
            )

        if "VECTOR" not in field_type and self.dim is not None:
            raise ValueError(
                f"Field '{self.name}' with type '{self.type}' cannot have dim. "
                f"This is only valid for vector fields (FloatVector, BinaryVector, etc.)."
            )

        return self


class CollectionSchema(BaseModel):
    """Pydantic model for collection schema validation."""

    collection_name: str | None = Field(
        default=None, description="Collection name", min_length=1, max_length=255
    )
    fields: list[FieldSchema] = Field(
        ..., description="List of field schemas", min_length=1
    )

    @field_validator("collection_name")
    @classmethod
    def validate_collection_name(cls, v: str | None) -> str | None:
        """Validate collection name."""
        if v is not None and not v.replace("_", "").isalnum():
            raise ValueError(
                f"Collection name '{v}' is invalid. Collection names must contain only letters, numbers, and underscores."
            )
        return v

    @model_validator(mode="after")
    def validate_schema(self) -> CollectionSchema:
        """Validate the overall schema."""
        # Check for at least one primary key
        primary_fields = [f for f in self.fields if f.is_primary]
        if not primary_fields:
            raise ValueError(
                "Schema must have exactly one primary key field.\n"
                "Add is_primary=true to one of your fields.\n"
                'Example: {"name": "id", "type": "Int64", "is_primary": true}'
            )

        if len(primary_fields) > 1:
            primary_names = [f.name for f in primary_fields]
            raise ValueError(
                f"Schema can have only one primary key field, but found {len(primary_fields)}: {', '.join(primary_names)}\n"
                "Set is_primary=true for only one field."
            )

        # Check for duplicate field names
        field_names = [f.name for f in self.fields]
        duplicates = [name for name in set(field_names) if field_names.count(name) > 1]
        if duplicates:
            raise ValueError(
                f"Duplicate field names found: {', '.join(duplicates)}\n"
                "Each field must have a unique name."
            )

        # Validate auto_id constraint
        primary_field = primary_fields[0]
        if primary_field.auto_id and primary_field.type not in {FieldType.INT64}:
            raise ValueError(
                f"Primary key field '{primary_field.name}' with auto_id=true must be of type Int64, not {primary_field.type}.\n"
                "Change type to Int64 or set auto_id=false."
            )

        return self


# Support for list-only schemas (backward compatibility)
SchemaModel = CollectionSchema | list[FieldSchema]


def validate_schema_data(data: dict[str, Any] | list[dict[str, Any]]) -> SchemaModel:
    """Validate schema data and return appropriate model."""
    if isinstance(data, list):
        # List of fields only
        return [FieldSchema.model_validate(field) for field in data]
    else:
        # Full collection schema
        return CollectionSchema.model_validate(data)


def get_schema_help() -> str:
    """Get comprehensive help text for writing schema files."""
    return """
# Milvus Schema File Format

## Basic Structure

### Full Schema Format:
```json
{
  "collection_name": "my_collection",
  "fields": [
    {"name": "id", "type": "Int64", "is_primary": true},
    {"name": "title", "type": "VarChar", "max_length": 128},
    {"name": "embedding", "type": "FloatVector", "dim": 768}
  ]
}
```

### Simplified Format (fields only):
```json
[
  {"name": "id", "type": "Int64", "is_primary": true},
  {"name": "title", "type": "VarChar", "max_length": 128}
]
```

## Field Types and Requirements

### Numeric Types
- **Int8, Int16, Int32, Int64**: Integer types
- **Float, Double**: Floating-point types
- Optional: `min`, `max` for value ranges

### String Types
- **VarChar, String**: Variable-length strings
- Required: `max_length` (1-65535)

### Other Types
- **Bool**: Boolean values
- **JSON**: JSON objects

### Vector Types
- **FloatVector**: 32-bit float vectors
- **BinaryVector**: Binary vectors
- **Float16Vector**: 16-bit float vectors
- **BFloat16Vector**: Brain float vectors
- **Int8Vector**: 8-bit integer vectors
- **SparseFloatVector**: Sparse float vectors
- Required: `dim` (dimension, 1-32768)

### Array Type
- **Array**: Array of elements
- Required: `element_type`, `max_capacity`
- If element_type is VarChar/String: also need `max_length`

## Field Properties

### Required Properties
- `name`: Field name (letters, numbers, underscores only)
- `type`: Field type (see above)

### Optional Properties
- `is_primary`: Primary key flag (exactly one field must be true)
- `auto_id`: Auto-generate ID (only for Int64 primary keys)
- `nullable`: Allow null values
- `min`, `max`: Value constraints (numeric types only)
- `max_length`: String length limit (string/array types)
- `dim`: Vector dimension (vector types only)
- `element_type`: Array element type (array only)
- `max_capacity`: Array capacity (array only)

## Examples

### E-commerce Collection:
```json
{
  "collection_name": "products",
  "fields": [
    {"name": "id", "type": "Int64", "is_primary": true, "auto_id": true},
    {"name": "title", "type": "VarChar", "max_length": 200},
    {"name": "price", "type": "Float", "min": 0.0, "max": 10000.0},
    {"name": "tags", "type": "Array", "element_type": "VarChar", "max_capacity": 10, "max_length": 50},
    {"name": "in_stock", "type": "Bool", "nullable": true},
    {"name": "metadata", "type": "JSON"},
    {"name": "embedding", "type": "FloatVector", "dim": 768}
  ]
}
```

### Simple Document Collection:
```json
[
  {"name": "doc_id", "type": "VarChar", "max_length": 100, "is_primary": true},
  {"name": "content", "type": "VarChar", "max_length": 5000},
  {"name": "vector", "type": "FloatVector", "dim": 384}
]
```
""".strip()
