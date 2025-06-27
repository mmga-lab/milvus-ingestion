"""Tests for built-in schemas functionality."""

import json
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from milvus_fake_data.builtin_schemas import (
    get_schema_info,
    list_builtin_schemas,
    load_builtin_schema,
    save_schema_to_file,
    validate_all_builtin_schemas,
)
from milvus_fake_data.cli import main
from milvus_fake_data.generator import generate_mock_data


def test_list_builtin_schemas():
    """Test listing built-in schemas."""
    schemas = list_builtin_schemas()

    # Check that we have expected schemas
    expected_schemas = {
        "simple",
        "ecommerce",
        "documents",
        "images",
        "users",
        "videos",
        "news",
    }
    assert set(schemas.keys()) == expected_schemas

    # Check schema metadata structure
    for _schema_id, info in schemas.items():
        assert "name" in info
        assert "description" in info
        assert "use_cases" in info
        assert "fields_count" in info
        assert "vector_dims" in info
        assert "file" in info
        assert isinstance(info["use_cases"], list)
        assert isinstance(info["fields_count"], int)
        assert isinstance(info["vector_dims"], list)


def test_get_schema_info():
    """Test getting specific schema info."""
    # Test existing schema
    info = get_schema_info("simple")
    assert info is not None
    assert info["name"] == "Simple Example"
    assert "Simple example schema" in info["description"]

    # Test non-existing schema
    info = get_schema_info("nonexistent")
    assert info is None


def test_load_builtin_schema():
    """Test loading built-in schemas."""
    # Test loading valid schema
    schema = load_builtin_schema("simple")
    assert isinstance(schema, dict)
    assert "collection_name" in schema
    assert "fields" in schema
    assert schema["collection_name"] == "simple_example"
    assert len(schema["fields"]) == 6

    # Test loading invalid schema
    with pytest.raises(ValueError, match="Unknown schema ID"):
        load_builtin_schema("nonexistent")


def test_save_schema_to_file():
    """Test saving schema to file."""
    schema = load_builtin_schema("simple")

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "test_schema.json"
        save_schema_to_file(schema, output_path)

        assert output_path.exists()

        # Verify the saved content
        with open(output_path, encoding="utf-8") as f:
            saved_schema = json.load(f)

        assert saved_schema == schema


def test_validate_all_builtin_schemas():
    """Test that all built-in schemas are valid."""
    errors = validate_all_builtin_schemas()
    assert len(errors) == 0, f"Schema validation errors: {errors}"


def test_generate_data_from_builtin_schemas():
    """Test generating data from all built-in schemas."""
    schemas = list_builtin_schemas()

    for schema_id in schemas:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            schema_data = load_builtin_schema(schema_id)
            json.dump(schema_data, f)
            schema_path = Path(f.name)

        try:
            # Test data generation
            df = generate_mock_data(schema_path, rows=3, seed=42)

            assert len(df) == 3
            assert len(df.columns) > 0

            # Check that vector fields have the right dimensions
            for field in schema_data["fields"]:
                if field["type"].endswith("Vector"):
                    field_name = field["name"]
                    if field_name in df.columns:  # Skip auto_id fields
                        expected_dim = field["dim"]
                        # Check first non-null value
                        sample_vector = df[field_name].dropna().iloc[0]
                        assert len(sample_vector) == expected_dim
        finally:
            schema_path.unlink()


def test_cli_list_schemas():
    """Test CLI command to list schemas."""
    runner = CliRunner()
    result = runner.invoke(main, ["schema", "list"])

    assert result.exit_code == 0
    assert "Built-in Schemas" in result.output
    assert "simple" in result.output
    assert "ecommerce" in result.output
    assert "Simple Example" in result.output


def test_cli_builtin_schema_generation():
    """Test CLI generation with built-in schema."""
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as temp_dir:
        result = runner.invoke(
            main,
            [
                "generate",
                "--builtin",
                "simple",
                "--rows",
                "3",
                "--out",
                f"{temp_dir}/test_output.parquet",
                "--seed",
                "42",
            ],
            input="y\n",
        )

        assert result.exit_code == 0
        assert "✓ Loaded built-in schema: simple" in result.output
        assert "Saved 3 rows" in result.output

        output_file = Path(temp_dir) / "test_output.parquet"
        assert output_file.exists()


def test_cli_invalid_builtin_schema():
    """Test CLI with invalid built-in schema."""
    runner = CliRunner()

    result = runner.invoke(
        main, ["generate", "--builtin", "nonexistent", "--rows", "1"]
    )

    assert result.exit_code == 1
    assert "✗ Error with schema" in result.output
    assert "Available schemas:" in result.output


def test_cli_conflicting_arguments():
    """Test CLI with conflicting arguments."""
    runner = CliRunner()

    # Test schema + builtin (create a dummy schema file first)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"fields": [{"name": "id", "type": "Int64", "is_primary": True}]}, f)
        schema_file = Path(f.name)

    try:
        result = runner.invoke(
            main, ["generate", "--schema", str(schema_file), "--builtin", "simple"]
        )
        assert result.exit_code == 1
        assert "Cannot use" in result.output
    finally:
        schema_file.unlink()


def test_cli_validate_builtin_schema():
    """Test CLI validation with built-in schema."""
    runner = CliRunner()

    result = runner.invoke(main, ["generate", "--builtin", "simple", "--validate-only"])

    assert result.exit_code == 0
    assert "✓ Loaded built-in schema: simple" in result.output
    assert "✓ Schema" in result.output and "is valid!" in result.output


@pytest.mark.parametrize(
    "schema_id",
    ["simple", "ecommerce", "documents", "images", "users", "videos", "news"],
)
def test_schema_completeness(schema_id):
    """Test that each schema has all required fields and proper structure."""
    schema = load_builtin_schema(schema_id)

    # Check required top-level fields
    assert "collection_name" in schema
    assert "fields" in schema
    assert isinstance(schema["fields"], list)
    assert len(schema["fields"]) > 0

    # Check that there's exactly one primary key
    primary_fields = [f for f in schema["fields"] if f.get("is_primary", False)]
    assert len(primary_fields) == 1, (
        f"Schema {schema_id} must have exactly one primary key"
    )

    # Check field structure
    for field in schema["fields"]:
        assert "name" in field
        assert "type" in field

        # Vector fields must have dim
        if field["type"].endswith("Vector"):
            assert "dim" in field, f"Vector field {field['name']} missing dimension"
            assert isinstance(field["dim"], int)
            assert field["dim"] > 0

        # String fields must have max_length
        if field["type"] in ["VarChar", "String"]:
            assert "max_length" in field, (
                f"String field {field['name']} missing max_length"
            )
            assert isinstance(field["max_length"], int)
            assert field["max_length"] > 0

        # Array fields must have element_type and max_capacity
        if field["type"] == "Array":
            assert "element_type" in field, (
                f"Array field {field['name']} missing element_type"
            )
            assert "max_capacity" in field, (
                f"Array field {field['name']} missing max_capacity"
            )

            # Array of strings must have max_length
            if field["element_type"] in ["VarChar", "String"]:
                assert "max_length" in field, (
                    f"Array of string field {field['name']} missing max_length"
                )
