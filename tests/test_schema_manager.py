"""Tests for schema management functionality."""

import json
import os
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from milvus_fake_data.cli import main
from milvus_fake_data.schema_manager import SchemaManager, reset_schema_manager


@pytest.fixture
def temp_schema_dir():
    """Create temporary directory for schemas."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_custom_schema():
    """Sample custom schema for testing."""
    return {
        "collection_name": "test_custom",
        "description": "Test custom schema",
        "fields": [
            {"name": "id", "type": "Int64", "is_primary": True},
            {"name": "name", "type": "VarChar", "max_length": 100},
            {"name": "embedding", "type": "FloatVector", "dim": 256},
        ],
    }


@pytest.fixture
def schema_manager(temp_schema_dir):
    """Schema manager with temporary directory."""
    return SchemaManager(temp_schema_dir)


class TestSchemaManager:
    """Test SchemaManager class."""

    def test_init_creates_directories(self, temp_schema_dir):
        """Test that initialization creates necessary directories and files."""
        manager = SchemaManager(temp_schema_dir)

        assert manager.schema_dir.exists()
        assert manager.metadata_file.exists()

        # Check metadata file is valid JSON
        with open(manager.metadata_file) as f:
            metadata = json.load(f)
        assert isinstance(metadata, dict)

    def test_add_schema(self, schema_manager, sample_custom_schema):
        """Test adding a custom schema."""
        schema_manager.add_schema(
            "test_schema",
            sample_custom_schema,
            "Test description",
            ["testing", "example"],
        )

        # Check schema file was created
        schema_file = schema_manager.schema_dir / "test_schema.json"
        assert schema_file.exists()

        # Check metadata was updated
        metadata = schema_manager._load_metadata()
        assert "test_schema" in metadata
        assert metadata["test_schema"]["name"] == "test_custom"
        assert metadata["test_schema"]["description"] == "Test description"
        assert metadata["test_schema"]["use_cases"] == ["testing", "example"]
        assert metadata["test_schema"]["fields_count"] == 3
        assert metadata["test_schema"]["vector_dims"] == [256]

    def test_add_schema_duplicate(self, schema_manager, sample_custom_schema):
        """Test adding duplicate schema raises error."""
        schema_manager.add_schema("test_schema", sample_custom_schema)

        with pytest.raises(ValueError, match="already exists"):
            schema_manager.add_schema("test_schema", sample_custom_schema)

    def test_add_schema_invalid(self, schema_manager):
        """Test adding invalid schema raises error."""
        invalid_schema = {"fields": [{"name": "field1"}]}  # Missing type

        with pytest.raises(ValueError, match="Invalid schema"):
            schema_manager.add_schema("invalid", invalid_schema)

    def test_load_schema_custom(self, schema_manager, sample_custom_schema):
        """Test loading a custom schema."""
        schema_manager.add_schema("test_schema", sample_custom_schema)

        loaded_schema = schema_manager.load_schema("test_schema")
        assert loaded_schema == sample_custom_schema

    def test_load_schema_builtin(self, schema_manager):
        """Test loading a built-in schema."""
        schema = schema_manager.load_schema("simple")
        assert isinstance(schema, dict)
        assert "collection_name" in schema
        assert "fields" in schema

    def test_load_schema_not_found(self, schema_manager):
        """Test loading non-existent schema raises error."""
        with pytest.raises(ValueError, match="does not exist"):
            schema_manager.load_schema("nonexistent")

    def test_update_schema(self, schema_manager, sample_custom_schema):
        """Test updating an existing custom schema."""
        schema_manager.add_schema("test_schema", sample_custom_schema)

        updated_schema = sample_custom_schema.copy()
        updated_schema["fields"].append({"name": "new_field", "type": "Float"})

        schema_manager.update_schema(
            "test_schema", updated_schema, "Updated description", ["updated"]
        )

        # Check schema was updated
        loaded_schema = schema_manager.load_schema("test_schema")
        assert len(loaded_schema["fields"]) == 4

        # Check metadata was updated
        metadata = schema_manager._load_metadata()
        assert metadata["test_schema"]["description"] == "Updated description"
        assert metadata["test_schema"]["use_cases"] == ["updated"]
        assert metadata["test_schema"]["fields_count"] == 4

    def test_update_schema_not_found(self, schema_manager, sample_custom_schema):
        """Test updating non-existent schema raises error."""
        with pytest.raises(ValueError, match="does not exist"):
            schema_manager.update_schema("nonexistent", sample_custom_schema)

    def test_update_builtin_schema(self, schema_manager, sample_custom_schema):
        """Test updating built-in schema raises error."""
        with pytest.raises(ValueError, match="Cannot update built-in"):
            schema_manager.update_schema("simple", sample_custom_schema)

    def test_remove_schema(self, schema_manager, sample_custom_schema):
        """Test removing a custom schema."""
        schema_manager.add_schema("test_schema", sample_custom_schema)

        schema_manager.remove_schema("test_schema")

        # Check schema file was removed
        schema_file = schema_manager.schema_dir / "test_schema.json"
        assert not schema_file.exists()

        # Check metadata was updated
        metadata = schema_manager._load_metadata()
        assert "test_schema" not in metadata

    def test_remove_schema_not_found(self, schema_manager):
        """Test removing non-existent schema raises error."""
        with pytest.raises(ValueError, match="does not exist"):
            schema_manager.remove_schema("nonexistent")

    def test_remove_builtin_schema(self, schema_manager):
        """Test removing built-in schema raises error."""
        with pytest.raises(ValueError, match="Cannot remove built-in"):
            schema_manager.remove_schema("simple")

    def test_list_all_schemas(self, schema_manager, sample_custom_schema):
        """Test listing all schemas (built-in + custom)."""
        schema_manager.add_schema("custom1", sample_custom_schema)

        all_schemas = schema_manager.list_all_schemas()

        # Should include built-in schemas
        assert "simple" in all_schemas
        assert "ecommerce" in all_schemas

        # Should include custom schema
        assert "custom1" in all_schemas
        assert all_schemas["custom1"]["type"] == "custom"

    def test_list_custom_schemas(self, schema_manager, sample_custom_schema):
        """Test listing only custom schemas."""
        schema_manager.add_schema("custom1", sample_custom_schema)

        custom_schemas = schema_manager.list_custom_schemas()

        # Should only include custom schemas
        assert "custom1" in custom_schemas
        assert "simple" not in custom_schemas

    def test_schema_exists(self, schema_manager, sample_custom_schema):
        """Test checking if schema exists."""
        assert schema_manager.schema_exists("simple")  # Built-in
        assert not schema_manager.schema_exists("nonexistent")

        schema_manager.add_schema("custom1", sample_custom_schema)
        assert schema_manager.schema_exists("custom1")  # Custom

    def test_is_builtin_schema(self, schema_manager, sample_custom_schema):
        """Test checking if schema is built-in."""
        assert schema_manager.is_builtin_schema("simple")
        assert not schema_manager.is_builtin_schema("nonexistent")

        schema_manager.add_schema("custom1", sample_custom_schema)
        assert not schema_manager.is_builtin_schema("custom1")

    def test_get_schema_info(self, schema_manager, sample_custom_schema):
        """Test getting schema information."""
        # Built-in schema
        info = schema_manager.get_schema_info("simple")
        assert info is not None
        assert info["name"] == "Simple Example"

        # Custom schema
        schema_manager.add_schema("custom1", sample_custom_schema, "Custom description")
        info = schema_manager.get_schema_info("custom1")
        assert info is not None
        assert info["name"] == "test_custom"
        assert info["description"] == "Custom description"

        # Non-existent schema
        info = schema_manager.get_schema_info("nonexistent")
        assert info is None

    def test_extract_vector_dims(self, schema_manager):
        """Test extracting vector dimensions from schema."""
        schema_data = {
            "fields": [
                {"name": "id", "type": "Int64"},
                {"name": "vec1", "type": "FloatVector", "dim": 128},
                {"name": "vec2", "type": "BinaryVector", "dim": 256},
                {"name": "vec3", "type": "FloatVector", "dim": 128},  # Duplicate
            ]
        }

        dims = schema_manager._extract_vector_dims(schema_data)
        assert dims == [128, 256]  # Sorted and unique


class TestSchemaManagerCLI:
    """Test schema management CLI commands."""

    def test_list_all_schemas_command(self):
        """Test --list-schemas command."""
        runner = CliRunner()
        result = runner.invoke(main, ["schema", "list"])

        assert result.exit_code == 0
        assert "Built-in Schemas" in result.output
        assert "simple" in result.output
        assert "ecommerce" in result.output

    def test_show_schema_builtin(self):
        """Test --show-schema with built-in schema."""
        runner = CliRunner()
        result = runner.invoke(main, ["schema", "show", "simple"])

        assert result.exit_code == 0
        assert "Schema: simple (Built-in)" in result.output
        assert "Simple Example" in result.output
        assert "Fields" in result.output

    def test_show_schema_not_found(self):
        """Test --show-schema with non-existent schema."""
        runner = CliRunner()
        result = runner.invoke(main, ["schema", "show", "nonexistent"])

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_add_schema_command(self, sample_custom_schema):
        """Test --add-schema command."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            # Set isolated schema directory
            schema_dir = Path.cwd() / "schemas"
            os.environ["MILVUS_FAKE_DATA_SCHEMA_DIR"] = str(schema_dir)
            reset_schema_manager()  # Reset global instance

            try:
                # Create schema file
                schema_file = Path("test_schema.json")
                with open(schema_file, "w") as f:
                    json.dump(sample_custom_schema, f)

                # Test add schema with interactive input
                inputs = [
                    "Test custom schema",  # description
                    "testing, example",  # use cases
                ]

                result = runner.invoke(
                    main,
                    ["schema", "add", "my_custom", str(schema_file)],
                    input="\n".join(inputs),
                )

                assert result.exit_code == 0
                assert "✓ Added custom schema: my_custom" in result.output
                assert "Test custom schema" in result.output
            finally:
                # Clean up environment
                if "MILVUS_FAKE_DATA_SCHEMA_DIR" in os.environ:
                    del os.environ["MILVUS_FAKE_DATA_SCHEMA_DIR"]
                reset_schema_manager()

    def test_add_schema_invalid_format(self):
        """Test schema add with invalid file."""
        runner = CliRunner()
        result = runner.invoke(
            main, ["schema", "add", "test_schema", "nonexistent.json"]
        )

        assert result.exit_code != 0

    def test_add_schema_file_not_found(self):
        """Test --add-schema with non-existent file."""
        runner = CliRunner()
        result = runner.invoke(
            main, ["schema", "add", "test", "/nonexistent/file.json"]
        )

        assert result.exit_code != 0
        assert "does not exist" in result.output

    def test_remove_schema_command(self, sample_custom_schema):
        """Test --remove-schema command."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            # Set isolated schema directory
            schema_dir = Path.cwd() / "schemas"
            os.environ["MILVUS_FAKE_DATA_SCHEMA_DIR"] = str(schema_dir)
            reset_schema_manager()  # Reset global instance

            try:
                # Create schema file
                schema_file = Path("test_schema.json")
                with open(schema_file, "w") as f:
                    json.dump(sample_custom_schema, f)

                # Add schema
                result = runner.invoke(
                    main,
                    ["schema", "add", "test_remove", str(schema_file)],
                    input="Test schema\ntesting",
                )
                assert result.exit_code == 0

                # Remove schema
                result = runner.invoke(
                    main, ["schema", "remove", "test_remove"], input="y"
                )  # Confirm removal

                assert result.exit_code == 0
                assert "✓ Removed custom schema: test_remove" in result.output
            finally:
                # Clean up environment
                if "MILVUS_FAKE_DATA_SCHEMA_DIR" in os.environ:
                    del os.environ["MILVUS_FAKE_DATA_SCHEMA_DIR"]
                reset_schema_manager()

    def test_remove_schema_not_found(self):
        """Test --remove-schema with non-existent schema."""
        runner = CliRunner()
        result = runner.invoke(main, ["schema", "remove", "nonexistent"])

        assert result.exit_code == 1
        assert "does not exist" in result.output

    def test_remove_builtin_schema(self):
        """Test --remove-schema with built-in schema."""
        runner = CliRunner()
        result = runner.invoke(main, ["schema", "remove", "simple"])

        assert result.exit_code == 1
        assert "Cannot remove built-in schema" in result.output

    def test_builtin_with_custom_schema(self, sample_custom_schema):
        """Test using --builtin flag with custom schema."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            # Set isolated schema directory
            schema_dir = Path.cwd() / "schemas"
            os.environ["MILVUS_FAKE_DATA_SCHEMA_DIR"] = str(schema_dir)
            reset_schema_manager()  # Reset global instance

            try:
                # Create schema file in isolated environment
                schema_file = Path("test_schema.json")
                with open(schema_file, "w") as f:
                    json.dump(sample_custom_schema, f)

                # Add custom schema
                result = runner.invoke(
                    main,
                    ["schema", "add", "my_custom_test", str(schema_file)],
                    input="Custom schema\ntesting",
                )
                assert result.exit_code == 0, (
                    f"Expected exit code 0 but got {result.exit_code}. Output: {result.output}"
                )

                # Use custom schema with --builtin flag
                result = runner.invoke(
                    main,
                    [
                        "generate",
                        "--builtin",
                        "my_custom_test",
                        "--rows",
                        "3",
                        "--out",
                        "output.parquet",
                        "--seed",
                        "42",
                    ],
                    input="y\n",
                )

                assert result.exit_code == 0
                assert "✓ Loaded custom schema: my_custom_test" in result.output
                assert "Saved 3 rows" in result.output
            finally:
                # Clean up environment
                if "MILVUS_FAKE_DATA_SCHEMA_DIR" in os.environ:
                    del os.environ["MILVUS_FAKE_DATA_SCHEMA_DIR"]
                reset_schema_manager()

    def test_show_custom_schema(self, sample_custom_schema):
        """Test --show-schema with custom schema."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            # Set isolated schema directory
            schema_dir = Path.cwd() / "schemas"
            os.environ["MILVUS_FAKE_DATA_SCHEMA_DIR"] = str(schema_dir)
            reset_schema_manager()  # Reset global instance

            try:
                # Create schema file in isolated environment
                schema_file = Path("test_schema.json")
                with open(schema_file, "w") as f:
                    json.dump(sample_custom_schema, f)

                # Add custom schema
                result = runner.invoke(
                    main,
                    ["schema", "add", "my_custom_show", str(schema_file)],
                    input="Custom test schema\ntesting, example",
                )
                assert result.exit_code == 0

                # Show custom schema
                result = runner.invoke(main, ["schema", "show", "my_custom_show"])

                assert result.exit_code == 0
                assert "Schema: my_custom_show (Custom)" in result.output
                assert "Custom test schema" in result.output
                assert "Fields" in result.output
                assert (
                    "id" in result.output
                    and "Int64" in result.output
                    and "PRIMARY" in result.output
                )
            finally:
                # Clean up environment
                if "MILVUS_FAKE_DATA_SCHEMA_DIR" in os.environ:
                    del os.environ["MILVUS_FAKE_DATA_SCHEMA_DIR"]
                reset_schema_manager()
