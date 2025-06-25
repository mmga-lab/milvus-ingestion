"""Tests for the generator module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from milvus_fake_data.generator import generate_mock_data


@pytest.fixture
def basic_schema():
    """Basic schema for testing."""
    return {
        "collection_name": "test_collection",
        "fields": [
            {"name": "id", "type": "Int64", "is_primary": True},
            {"name": "title", "type": "VarChar", "max_length": 100},
            {"name": "score", "type": "Float"},
            {"name": "active", "type": "Bool"},
            {"name": "embedding", "type": "FloatVector", "dim": 8},
        ],
    }


@pytest.fixture
def complex_schema():
    """Complex schema with various field types."""
    return {
        "collection_name": "complex_collection",
        "fields": [
            {"name": "id", "type": "Int64", "is_primary": True, "auto_id": True},
            {"name": "name", "type": "VarChar", "max_length": 50, "nullable": True},
            {"name": "age", "type": "Int32", "min": 0, "max": 100},
            {"name": "price", "type": "Double", "min": 0.0, "max": 1000.0},
            {
                "name": "tags",
                "type": "Array",
                "element_type": "VarChar",
                "max_capacity": 5,
                "max_length": 20,
            },
            {"name": "metadata", "type": "JSON"},
            {"name": "features", "type": "FloatVector", "dim": 128},
            {"name": "binary_data", "type": "BinaryVector", "dim": 64},
        ],
    }


def test_generate_mock_data_basic(basic_schema):
    """Test basic mock data generation."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(basic_schema, f)
        schema_path = Path(f.name)

    try:
        df = generate_mock_data(schema_path, rows=10, seed=42)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert list(df.columns) == ["id", "title", "score", "active", "embedding"]

        # Test data types
        assert df["id"].dtype == "int64"
        assert df["title"].dtype == "object"
        assert df["score"].dtype == "float64"
        assert df["active"].dtype == "bool"
        assert df["embedding"].dtype == "object"

        # Test vector dimensions
        assert all(len(vec) == 8 for vec in df["embedding"])

        # Test primary key uniqueness
        assert df["id"].nunique() == len(df)

    finally:
        schema_path.unlink()


def test_generate_mock_data_with_seed(basic_schema):
    """Test reproducible data generation with seed."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(basic_schema, f)
        schema_path = Path(f.name)

    try:
        df1 = generate_mock_data(schema_path, rows=5, seed=123)
        df2 = generate_mock_data(schema_path, rows=5, seed=123)

        pd.testing.assert_frame_equal(df1, df2)

    finally:
        schema_path.unlink()


def test_generate_mock_data_complex(complex_schema):
    """Test complex schema with various field types."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(complex_schema, f)
        schema_path = Path(f.name)

    try:
        df = generate_mock_data(schema_path, rows=20, seed=42)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 20

        # Test auto_id field is skipped
        assert "id" not in df.columns

        # Test nullable field has some null values
        assert df["name"].isnull().any()

        # Test constraints
        assert df["age"].min() >= 0
        assert df["age"].max() <= 100
        assert df["price"].min() >= 0.0
        assert df["price"].max() <= 1000.0

        # Test array field
        assert all(isinstance(arr, list) for arr in df["tags"])
        assert all(len(arr) <= 5 for arr in df["tags"])

        # Test JSON field
        assert all(isinstance(obj, dict) for obj in df["metadata"])

        # Test vector dimensions
        assert all(len(vec) == 128 for vec in df["features"])
        assert all(len(vec) == 64 for vec in df["binary_data"])

    finally:
        schema_path.unlink()


def test_generate_mock_data_yaml_schema():
    """Test YAML schema loading."""
    schema_yaml = """
collection_name: "yaml_test"
fields:
  - name: "id"
    type: "Int64"
    is_primary: true
  - name: "description"
    type: "VarChar"
    max_length: 200
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(schema_yaml)
        schema_path = Path(f.name)

    try:
        df = generate_mock_data(schema_path, rows=5, seed=42)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert list(df.columns) == ["id", "description"]

    finally:
        schema_path.unlink()


def test_schema_without_collection_name():
    """Test schema that is just a list of fields."""
    schema = [
        {"name": "id", "type": "Int64", "is_primary": True},
        {"name": "value", "type": "Float"},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(schema, f)
        schema_path = Path(f.name)

    try:
        df = generate_mock_data(schema_path, rows=3, seed=42)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == ["id", "value"]

    finally:
        schema_path.unlink()


def test_vector_types():
    """Test different vector types."""
    schema = {
        "fields": [
            {"name": "id", "type": "Int64", "is_primary": True},
            {"name": "float_vec", "type": "FloatVector", "dim": 4},
            {"name": "binary_vec", "type": "BinaryVector", "dim": 8},
            {"name": "int8_vec", "type": "Int8Vector", "dim": 6},
        ]
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(schema, f)
        schema_path = Path(f.name)

    try:
        df = generate_mock_data(schema_path, rows=3, seed=42)

        # Test float vector
        assert all(len(vec) == 4 for vec in df["float_vec"])
        assert all(isinstance(vec[0], float | np.floating) for vec in df["float_vec"])

        # Test binary vector
        assert all(len(vec) == 8 for vec in df["binary_vec"])
        assert all(all(v in [0, 1] for v in vec) for vec in df["binary_vec"])

        # Test int8 vector
        assert all(len(vec) == 6 for vec in df["int8_vec"])
        assert all(
            all(isinstance(v, int | np.integer) for v in vec) for vec in df["int8_vec"]
        )

    finally:
        schema_path.unlink()


def test_file_not_found():
    """Test error handling for non-existent schema file."""
    with pytest.raises(FileNotFoundError):
        generate_mock_data("/non/existent/file.json", rows=1)


def test_unsupported_field_type():
    """Test error handling for unsupported field types."""
    schema = {
        "fields": [
            {"name": "id", "type": "Int64", "is_primary": True},
            {"name": "unsupported", "type": "UnsupportedType"},
        ]
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(schema, f)
        schema_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="Schema validation failed"):
            generate_mock_data(schema_path, rows=1, seed=42)

    finally:
        schema_path.unlink()
