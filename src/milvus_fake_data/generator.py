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
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml  # type: ignore
from faker import Faker

faker = Faker()
NULL_PROB = 0.1  # probability to generate null for nullable fields

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_mock_data(schema_path: str | Path, rows: int = 1000, seed: int | None = None) -> pd.DataFrame:
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

    schema = _load_schema(schema_path)
    fields = schema.get("fields", schema)  # accept list-only schema

    # Generate rows one by one so we can respect nullable & auto_id handling
    import time
    base_ts = int(time.time() * 1000) << 18

    rows_data: list[Dict[str, Any]] = []
    pk_field = next((f for f in fields if f.get("is_primary")), None)
    for idx in range(rows):
        row: Dict[str, Any] = {}
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

def _gen_pk_value(f_type: str, base_ts: int, idx: int):
    """Generate unique & monotonic primary key value."""
    if f_type == "INT64":
        return base_ts + idx
    return str(base_ts + idx)


def _gen_value_by_field(field: Dict[str, Any]):
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
        return faker.text(max_nb_chars=max_length)
    if f_type == "JSON":
        return {"key": faker.text(max_nb_chars=16)}
    if f_type == "ARRAY":
        element_type = field.get("element_type", "INT32").upper()
        max_capacity = int(field.get("max_capacity", 5))
        arr_len = random.randint(1, max_capacity)
        return [_gen_value_by_field({"type": element_type}) for _ in range(arr_len)]
    # Vector types
    if f_type == "BINARY_VECTOR":
        return np.random.randint(0, 2, dim).astype(np.int8).tolist()
    if f_type == "INT8_VECTOR":
        return np.random.randint(0, 256, dim).astype(np.int8).tolist()
    if f_type in {"FLOAT_VECTOR", "FLOAT16_VECTOR", "BFLOAT16_VECTOR"}:
        dtype = np.float16 if f_type in {"FLOAT16_VECTOR", "BFLOAT16_VECTOR"} else np.float32
        return np.random.random(dim).astype(dtype).tolist()
    if f_type == "SPARSE_FLOAT_VECTOR":
        return np.random.randint(0, 2, dim).astype(np.int8).tolist()
    raise ValueError(f"Unsupported field type: {f_type}")


def _load_schema(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    content = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(content)
    elif path.suffix.lower() == ".json":
        return json.loads(content)
    else:
        # fallback: try json then yaml
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return yaml.safe_load(content)

