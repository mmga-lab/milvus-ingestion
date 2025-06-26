"""Core logic for generating mock data from a Milvus collection schema (offline file).

The module exposes a single public function `generate_mock_data`, which returns a
pandas `DataFrame` containing random data following the provided schema.

Supported field types (case-insensitive):
    • Int8 / Int16 / Int32 / Int64
    • Float / Double
    • Bool
    • VarChar (string)
    • FloatVector (requires `dim` in field definition)
    • BinaryVector (requires `dim` in field definition, dim is number of bits)

Schema file format (JSON / YAML):
{
  "collection_name": "my_collection",
  "fields": [
    {"name": "id", "type": "Int64", "is_primary": true},
    {"name": "age", "type": "Int32"},
    {"name": "embedding", "type": "FloatVector", "dim": 128}
  ]
}

If the top-level key `fields` is missing we assume the whole dict is already a
list of fields.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml
from faker import Faker
from pydantic import ValidationError

from .models import get_schema_help, validate_schema_data

faker = Faker()
NULL_PROB = 0.1  # probability to generate null for nullable fields

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_mock_data(
    schema_path: str | Path, rows: int = 1000, seed: int | None = None
) -> pd.DataFrame:
    """Generate mock data according to the schema described by *schema_path*.

    Args:
        schema_path: Path to JSON or YAML schema file.
        rows: Number of rows to generate.
        seed: Optional random seed for reproducibility.

    Returns:
        A *pandas.DataFrame* containing mock data with *rows* rows.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        Faker.seed(seed)

    schema_data = _load_schema(schema_path)
    try:
        validated_schema = validate_schema_data(schema_data)
        if isinstance(validated_schema, list):
            # List of field schemas
            fields = [
                field.model_dump(exclude_none=True, exclude_unset=True)
                for field in validated_schema
            ]
        else:
            # Collection schema
            fields = [
                field.model_dump(exclude_none=True, exclude_unset=True)
                for field in validated_schema.fields
            ]
    except ValidationError as e:
        # Format validation errors with helpful messages
        error_msg = "Schema validation failed:\n\n"
        for error in e.errors():
            loc = " -> ".join(str(x) for x in error["loc"])
            error_msg += f"• {loc}: {error['msg']}\n"

        error_msg += f"\n{get_schema_help()}"
        raise ValueError(error_msg) from e

    # Generate rows one by one so we can respect nullable & auto_id handling
    if seed is not None:
        # Use deterministic base for reproducible tests
        base_ts = (seed * 1000) << 18
    else:
        import time

        base_ts = int(time.time() * 1000) << 18

    rows_data: list[dict[str, Any]] = []
    pk_field = next((f for f in fields if f.get("is_primary")), None)
    for idx in range(rows):
        row: dict[str, Any] = {}
        for f in fields:
            name = f["name"]
            f_type = f["type"].upper()
            # Skip auto_id fields
            if f.get("auto_id"):
                continue
            # Primary key handling (monotonic unique)
            if pk_field and name == pk_field["name"]:
                row[name] = _gen_pk_value(f_type, base_ts, idx)
                continue
            # Nullable handling
            if f.get("nullable") and random.random() < NULL_PROB:
                row[name] = None
                continue
            # Generate value by type
            row[name] = _gen_value_by_field(f)
        rows_data.append(row)

    return pd.DataFrame(rows_data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gen_pk_value(f_type: str, base_ts: int, idx: int) -> int | str:
    """Generate unique & monotonic primary key value."""
    if f_type == "INT64":
        return base_ts + idx
    return str(base_ts + idx)


def _gen_value_by_field(field: dict[str, Any]) -> Any:
    """Generate a random value matching field definition."""
    f_type = field["type"].upper()
    # Canonicalize vector type names like FLOATVECTOR -> FLOAT_VECTOR
    if "VECTOR" in f_type and "_VECTOR" not in f_type:
        f_type = f_type.replace("VECTOR", "_VECTOR")
    dim = int(field.get("dim", 8))
    max_length = int(field.get("max_length", 128))
    if f_type == "BOOL":
        return random.choice([True, False])
    if f_type in {"INT8", "INT16", "INT32", "INT64"}:
        low, high = field.get("min", 0), field.get("max", 1_000_000)
        return random.randint(low, high)
    if f_type in {"FLOAT", "DOUBLE"}:
        low, high = field.get("min", 0.0), field.get("max", 1_000.0)
        return random.uniform(low, high)
    if f_type in {"VARCHAR", "STRING"}:
        # Use 80% of max_length to avoid hitting the limit
        safe_max_length = int(max_length * 0.8)
        return faker.text(max_nb_chars=safe_max_length)
    if f_type == "JSON":
        return {"key": faker.text(max_nb_chars=16)}
    if f_type == "ARRAY":
        element_type = field.get("element_type", "INT32").upper()
        max_capacity = int(field.get("max_capacity", 5))
        arr_len = random.randint(1, max_capacity)
        # Create element field with proper constraints
        element_field = {"type": element_type}
        # Pass max_length to VARCHAR elements in arrays
        if element_type in {"VARCHAR", "STRING"} and "max_length" in field:
            element_field["max_length"] = field["max_length"]
        return [_gen_value_by_field(element_field) for _ in range(arr_len)]
    # Vector types
    if f_type == "BINARY_VECTOR":
        return np.random.randint(0, 2, dim).astype(np.int8).tolist()
    if f_type == "INT8_VECTOR":
        return np.random.randint(0, 256, dim).astype(np.int8).tolist()
    if f_type in {"FLOAT_VECTOR", "FLOAT16_VECTOR", "BFLOAT16_VECTOR"}:
        dtype = (
            np.float16
            if f_type in {"FLOAT16_VECTOR", "BFLOAT16_VECTOR"}
            else np.float32
        )
        return np.random.random(dim).astype(dtype).tolist()
    if f_type == "SPARSE_FLOAT_VECTOR":
        return np.random.randint(0, 2, dim).astype(np.int8).tolist()
    raise ValueError(f"Unsupported field type: {f_type}")


def _load_schema(path: str | Path) -> dict[str, Any] | list[Any]:
    """Load JSON or YAML schema file and return as dictionary."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    content = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return cast("dict[str, Any]", yaml.safe_load(content))
    elif path.suffix.lower() == ".json":
        return cast("dict[str, Any]", json.loads(content))
    else:
        # fallback: try json then yaml
        try:
            return cast("dict[str, Any]", json.loads(content))
        except json.JSONDecodeError:
            return cast("dict[str, Any]", yaml.safe_load(content))
