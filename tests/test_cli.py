"""Comprehensive End-to-End tests for milvus-fake-data CLI.

This test suite covers all major functionality including:
- Data generation with built-in and custom schemas
- Schema management operations
- File format support (Parquet/JSON)
- S3/MinIO upload functionality
- Milvus integration (insert/import)
- Error handling and validation
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from milvus_fake_data.cli import main


@pytest.fixture
def cli_runner():
    """CLI test runner with isolated filesystem."""
    return CliRunner()


@pytest.fixture
def sample_schema():
    """Sample schema for testing basic functionality."""
    return {
        "collection_name": "e2e_test",
        "fields": [
            {"name": "id", "type": "Int64", "is_primary": True, "auto_id": True},
            {"name": "title", "type": "VarChar", "max_length": 100},
            {
                "name": "description",
                "type": "VarChar",
                "max_length": 500,
                "nullable": True,
            },
            {"name": "price", "type": "Float", "min": 0.01, "max": 999.99},
            {"name": "rating", "type": "Int8", "min": 1, "max": 5},
            {
                "name": "tags",
                "type": "Array",
                "element_type": "VarChar",
                "max_capacity": 5,
                "max_length": 20,
            },
            {"name": "metadata", "type": "JSON", "nullable": True},
            {"name": "is_active", "type": "Bool"},
            {"name": "embedding", "type": "FloatVector", "dim": 128},
            {"name": "sparse_vector", "type": "SparseFloatVector"},
        ],
    }


@pytest.fixture
def vector_schema():
    """Schema with various vector types for testing vector generation."""
    return {
        "collection_name": "vector_test",
        "fields": [
            {"name": "id", "type": "Int64", "is_primary": True},
            {"name": "float_vec", "type": "FloatVector", "dim": 64},
            {"name": "binary_vec", "type": "BinaryVector", "dim": 128},
            {"name": "float16_vec", "type": "Float16Vector", "dim": 32},
            {"name": "bfloat16_vec", "type": "BFloat16Vector", "dim": 256},
            {"name": "sparse_vec", "type": "SparseFloatVector"},
        ],
    }


class TestDataGeneration:
    """Test data generation functionality."""

    def test_generate_with_builtin_schema(self, cli_runner):
        """Test data generation using built-in schemas."""
        with cli_runner.isolated_filesystem():
            # Test with simple built-in schema
            result = cli_runner.invoke(
                main,
                ["generate", "--builtin", "simple", "--rows", "50", "--seed", "42"],
            )

            assert result.exit_code == 0
            assert (
                "Generated 50 rows" in result.output or "Saved 50 rows" in result.output
            )

            # Check output directory exists
            output_dir = Path.home() / ".milvus-fake-data" / "data" / "simple_example"
            assert output_dir.exists()

            # Check meta.json exists
            meta_file = output_dir / "meta.json"
            assert meta_file.exists()

            # Verify meta.json content
            with open(meta_file) as f:
                meta = json.load(f)
            assert meta["schema"]["collection_name"] == "simple_example"
            assert meta["generation_info"]["total_rows"] == 50

    def test_generate_with_custom_schema(self, cli_runner, sample_schema):
        """Test data generation with custom schema."""
        with cli_runner.isolated_filesystem():
            # Create schema file
            schema_file = Path("test_schema.json")
            with open(schema_file, "w") as f:
                json.dump(sample_schema, f)

            result = cli_runner.invoke(
                main,
                [
                    "generate",
                    "--schema",
                    str(schema_file),
                    "--rows",
                    "25",
                    "--seed",
                    "123",
                    "--out",
                    "custom_output",
                ],
            )

            assert result.exit_code == 0

            # Check output directory
            output_dir = Path("custom_output")
            assert output_dir.exists()

            # Verify data files exist
            data_files = [f for f in output_dir.iterdir() if f.name != "meta.json"]
            assert len(data_files) > 0

    def test_generate_different_formats(self, cli_runner, sample_schema):
        """Test data generation in different formats."""
        with cli_runner.isolated_filesystem():
            schema_file = Path("schema.json")
            with open(schema_file, "w") as f:
                json.dump(sample_schema, f)

            # Test Parquet format (default)
            result = cli_runner.invoke(
                main,
                [
                    "generate",
                    "--schema",
                    str(schema_file),
                    "--rows",
                    "10",
                    "--format",
                    "parquet",
                    "--out",
                    "parquet_output",
                ],
            )
            assert result.exit_code == 0

            parquet_dir = Path("parquet_output")
            assert parquet_dir.exists()
            parquet_files = list(parquet_dir.glob("*.parquet"))
            assert len(parquet_files) > 0

            # Test JSON format
            result = cli_runner.invoke(
                main,
                [
                    "generate",
                    "--schema",
                    str(schema_file),
                    "--rows",
                    "10",
                    "--format",
                    "json",
                    "--out",
                    "json_output",
                ],
            )
            assert result.exit_code == 0

            json_dir = Path("json_output")
            assert json_dir.exists()
            json_files = list(json_dir.glob("*.json"))
            # Filter out meta.json
            data_json_files = [f for f in json_files if f.name != "meta.json"]
            assert len(data_json_files) > 0

    def test_generate_large_dataset_with_partitioning(self, cli_runner, sample_schema):
        """Test large dataset generation with automatic file partitioning."""
        with cli_runner.isolated_filesystem():
            schema_file = Path("schema.json")
            with open(schema_file, "w") as f:
                json.dump(sample_schema, f)

            result = cli_runner.invoke(
                main,
                [
                    "generate",
                    "--schema",
                    str(schema_file),
                    "--rows",
                    "15000",  # Should create multiple files
                    "--max-rows-per-file",
                    "5000",
                    "--batch-size",
                    "2500",
                    "--out",
                    "large_output",
                ],
            )

            assert result.exit_code == 0

            output_dir = Path("large_output")
            assert output_dir.exists()

            # Should have multiple data files due to partitioning
            data_files = [f for f in output_dir.iterdir() if f.name != "meta.json"]
            assert len(data_files) >= 3  # 15000 rows / 5000 max = 3 files

            # Verify metadata
            with open(output_dir / "meta.json") as f:
                meta = json.load(f)
            assert meta["generation_info"]["total_rows"] == 15000

    def test_generate_vector_types(self, cli_runner, vector_schema):
        """Test generation of various vector types."""
        with cli_runner.isolated_filesystem():
            schema_file = Path("vector_schema.json")
            with open(schema_file, "w") as f:
                json.dump(vector_schema, f)

            result = cli_runner.invoke(
                main,
                [
                    "generate",
                    "--schema",
                    str(schema_file),
                    "--rows",
                    "20",
                    "--seed",
                    "42",
                    "--out",
                    "vector_output",
                ],
            )

            assert result.exit_code == 0

            output_dir = Path("vector_output")
            assert output_dir.exists()

    def test_preview_mode(self, cli_runner, sample_schema):
        """Test preview mode functionality."""
        with cli_runner.isolated_filesystem():
            schema_file = Path("schema.json")
            with open(schema_file, "w") as f:
                json.dump(sample_schema, f)

            result = cli_runner.invoke(
                main,
                [
                    "generate",
                    "--schema",
                    str(schema_file),
                    "--rows",
                    "100",
                    "--preview",
                    "--seed",
                    "42",
                ],
            )

            assert result.exit_code == 0
            assert "Preview (top 5 rows):" in result.output
            # Should not create actual output files in preview mode

    def test_validate_only_mode(self, cli_runner, sample_schema):
        """Test schema validation without generation."""
        with cli_runner.isolated_filesystem():
            schema_file = Path("schema.json")
            with open(schema_file, "w") as f:
                json.dump(sample_schema, f)

            result = cli_runner.invoke(
                main, ["generate", "--schema", str(schema_file), "--validate-only"]
            )

            assert result.exit_code == 0
            # Just check that it shows the schema structure (validation passed)
            assert "sparse_vector: SparseFloatVector" in result.output


class TestSchemaManagement:
    """Test schema management operations."""

    def test_schema_list(self, cli_runner):
        """Test listing available schemas."""
        result = cli_runner.invoke(main, ["schema", "list"])

        assert result.exit_code == 0
        # Should list built-in schemas
        assert "simple" in result.output
        assert "ecommerce" in result.output

    def test_schema_show(self, cli_runner):
        """Test showing schema details."""
        result = cli_runner.invoke(main, ["schema", "show", "simple"])

        assert result.exit_code == 0
        assert "simple_example" in result.output or "Simple Example" in result.output

    def test_schema_add_and_remove(self, cli_runner, sample_schema):
        """Test adding and removing custom schemas."""
        with cli_runner.isolated_filesystem():
            # Create schema file
            schema_file = Path("my_schema.json")
            with open(schema_file, "w") as f:
                json.dump(sample_schema, f)

            # First try to remove schema if it exists from previous test runs
            cli_runner.invoke(main, ["schema", "remove", "my_test_custom"], input="y\n")

            # Add custom schema with interactive input
            result = cli_runner.invoke(
                main,
                ["schema", "add", "my_test_custom", str(schema_file)],
                input="Test schema description\ntesting, e2e\n",
            )

            if result.exit_code != 0:
                print(f"Schema add failed: {result.output}")

            assert result.exit_code == 0
            assert "my_test_custom" in result.output

            # Verify it appears in listing
            result = cli_runner.invoke(main, ["schema", "list"])
            assert "my_test_custom" in result.output

            # Use the custom schema
            result = cli_runner.invoke(
                main, ["generate", "--builtin", "my_test_custom", "--rows", "5"]
            )
            assert result.exit_code == 0

            # Remove custom schema
            result = cli_runner.invoke(
                main, ["schema", "remove", "my_test_custom"], input="y\n"
            )
            assert result.exit_code == 0

    def test_schema_help(self, cli_runner):
        """Test schema format help."""
        result = cli_runner.invoke(main, ["schema", "help"])

        assert result.exit_code == 0
        assert "Schema Format Guide" in result.output or "Field Types" in result.output


class TestS3MinIOIntegration:
    """Test S3/MinIO upload functionality."""

    def test_upload_to_minio(self, cli_runner, sample_schema):
        """Test uploading data to local MinIO."""
        with cli_runner.isolated_filesystem():
            # Generate data first
            schema_file = Path("schema.json")
            with open(schema_file, "w") as f:
                json.dump(sample_schema, f)

            result = cli_runner.invoke(
                main,
                [
                    "generate",
                    "--schema",
                    str(schema_file),
                    "--rows",
                    "10",
                    "--out",
                    "test_data",
                ],
            )
            assert result.exit_code == 0

            # Test upload to local MinIO
            result = cli_runner.invoke(
                main,
                [
                    "upload",
                    "test_data",
                    "s3://a-bucket/test-upload/",
                    "--access-key-id",
                    "minioadmin",
                    "--secret-access-key",
                    "minioadmin",
                    "--endpoint-url",
                    "http://127.0.0.1:9000",
                    "--no-verify-ssl",
                ],
            )

            if result.exit_code != 0:
                print(f"Upload failed: {result.output}")

            assert result.exit_code == 0
            # Upload succeeded if exit code is 0 and no error messages
            assert "error" not in result.output.lower()


class TestMilvusIntegration:
    """Test Milvus integration functionality."""

    def test_milvus_insert(self, cli_runner, sample_schema):
        """Test direct insert to local Milvus."""
        with cli_runner.isolated_filesystem():
            # Create a simple schema without sparse vector for Milvus compatibility
            simple_schema = {
                "collection_name": "test_milvus_insert",
                "fields": [
                    {
                        "name": "id",
                        "type": "Int64",
                        "is_primary": True,
                        "auto_id": True,
                    },
                    {"name": "title", "type": "VarChar", "max_length": 100},
                    {"name": "price", "type": "Float", "min": 0.01, "max": 999.99},
                    {"name": "is_active", "type": "Bool"},
                    {"name": "embedding", "type": "FloatVector", "dim": 128},
                ],
            }

            # Generate data first
            schema_file = Path("schema.json")
            with open(schema_file, "w") as f:
                json.dump(simple_schema, f)

            result = cli_runner.invoke(
                main,
                [
                    "generate",
                    "--schema",
                    str(schema_file),
                    "--rows",
                    "20",
                    "--out",
                    "milvus_data",
                ],
            )
            assert result.exit_code == 0

            # Test insert to local Milvus
            result = cli_runner.invoke(
                main,
                [
                    "to-milvus",
                    "insert",
                    "milvus_data",
                    "--uri",
                    "http://127.0.0.1:19530",
                    "--batch-size",
                    "10",
                    "--drop-if-exists",
                ],
            )

            if result.exit_code != 0:
                print(f"Milvus insert failed: {result.output}")

            assert result.exit_code == 0
            # Insert succeeded if exit code is 0 and no error messages
            assert "error" not in result.output.lower()

    @patch("milvus_fake_data.milvus_importer.MilvusBulkImporter")
    def test_milvus_bulk_import(self, mock_importer, cli_runner):
        """Test bulk import to Milvus."""
        with cli_runner.isolated_filesystem():
            # Create dummy data file
            data_dir = Path("import_data")
            data_dir.mkdir()
            (data_dir / "data.parquet").touch()
            (data_dir / "meta.json").write_text('{"collection_name": "test"}')

            # Mock bulk importer
            mock_instance = Mock()
            mock_importer.return_value = mock_instance
            mock_instance.bulk_import.return_value = "job-123"
            mock_instance.wait_for_completion.return_value = True

            # Test bulk import
            result = cli_runner.invoke(
                main,
                [
                    "to-milvus",
                    "import",
                    "test_collection",
                    str(data_dir),
                    "--wait",
                    "--uri",
                    "http://localhost:19530",
                ],
            )

            assert result.exit_code == 0
            mock_importer.assert_called_once()


class TestErrorHandling:
    """Test error handling and validation."""

    def test_invalid_schema_file(self, cli_runner):
        """Test handling of invalid schema files."""
        with cli_runner.isolated_filesystem():
            # Create invalid JSON file
            invalid_file = Path("invalid.json")
            invalid_file.write_text("{ invalid json")

            result = cli_runner.invoke(
                main, ["generate", "--schema", str(invalid_file), "--rows", "1"]
            )

            assert result.exit_code != 0
            assert "Error loading schema" in result.output

    def test_missing_required_parameters(self, cli_runner):
        """Test handling of missing required parameters."""
        result = cli_runner.invoke(main, ["generate", "--rows", "1"])

        assert result.exit_code != 0
        assert "One of --schema or --builtin is required" in result.output

    def test_invalid_builtin_schema(self, cli_runner):
        """Test handling of non-existent built-in schema."""
        result = cli_runner.invoke(
            main, ["generate", "--builtin", "nonexistent", "--rows", "1"]
        )

        assert result.exit_code != 0

    def test_schema_validation_errors(self, cli_runner):
        """Test schema validation error handling."""
        with cli_runner.isolated_filesystem():
            # Create schema with validation errors
            invalid_schema = {
                "collection_name": "test",
                "fields": [
                    {"name": "vector", "type": "FloatVector"},  # Missing required 'dim'
                ],
            }

            schema_file = Path("invalid_schema.json")
            with open(schema_file, "w") as f:
                json.dump(invalid_schema, f)

            result = cli_runner.invoke(
                main, ["generate", "--schema", str(schema_file), "--validate-only"]
            )

            assert result.exit_code != 0
            assert "validation" in result.output.lower()


class TestUtilityCommands:
    """Test utility commands."""

    def test_clean_command(self, cli_runner):
        """Test clean command."""
        with cli_runner.isolated_filesystem():
            # Create some test data
            test_dir = Path.home() / ".milvus-fake-data" / "data" / "test"
            test_dir.mkdir(parents=True, exist_ok=True)
            (test_dir / "data.parquet").touch()

            result = cli_runner.invoke(main, ["clean", "--yes"])

            assert result.exit_code == 0

    def test_help_commands(self, cli_runner):
        """Test help command functionality."""
        result = cli_runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Generate mock data for Milvus" in result.output

        result = cli_runner.invoke(main, ["generate", "--help"])
        assert result.exit_code == 0

        result = cli_runner.invoke(main, ["schema", "--help"])
        assert result.exit_code == 0


class TestPerformanceOptions:
    """Test performance-related options."""

    def test_batch_size_option(self, cli_runner, sample_schema):
        """Test different batch sizes."""
        with cli_runner.isolated_filesystem():
            schema_file = Path("schema.json")
            with open(schema_file, "w") as f:
                json.dump(sample_schema, f)

            # Test with custom batch size
            result = cli_runner.invoke(
                main,
                [
                    "generate",
                    "--schema",
                    str(schema_file),
                    "--rows",
                    "1000",
                    "--batch-size",
                    "250",
                    "--out",
                    "batch_test",
                ],
            )

            assert result.exit_code == 0

    def test_file_size_limits(self, cli_runner, sample_schema):
        """Test file size limit options."""
        with cli_runner.isolated_filesystem():
            schema_file = Path("schema.json")
            with open(schema_file, "w") as f:
                json.dump(sample_schema, f)

            result = cli_runner.invoke(
                main,
                [
                    "generate",
                    "--schema",
                    str(schema_file),
                    "--rows",
                    "5000",
                    "--max-file-size",
                    "1",  # 1MB limit
                    "--max-rows-per-file",
                    "1000",
                    "--out",
                    "size_test",
                ],
            )

            assert result.exit_code == 0

            output_dir = Path("size_test")
            data_files = [f for f in output_dir.iterdir() if f.name != "meta.json"]
            assert (
                len(data_files) >= 3
            )  # Should create multiple files due to size limits


# Integration test that exercises the full workflow
class TestFullWorkflow:
    """Test complete end-to-end workflows."""

    def test_complete_data_pipeline_workflow(self, cli_runner):
        """Test complete workflow from schema creation to data generation."""
        with cli_runner.isolated_filesystem():
            # Step 1: Create a custom schema
            custom_schema = {
                "collection_name": "complete_test",
                "fields": [
                    {"name": "id", "type": "Int64", "is_primary": True},
                    {"name": "product_name", "type": "VarChar", "max_length": 200},
                    {"name": "price", "type": "Double", "min": 1.0, "max": 1000.0},
                    {
                        "name": "category_tags",
                        "type": "Array",
                        "element_type": "VarChar",
                        "max_capacity": 3,
                        "max_length": 50,
                    },
                    {"name": "product_embedding", "type": "FloatVector", "dim": 256},
                    {"name": "is_featured", "type": "Bool"},
                    {"name": "metadata", "type": "JSON", "nullable": True},
                ],
            }

            schema_file = Path("complete_schema.json")
            with open(schema_file, "w") as f:
                json.dump(custom_schema, f, indent=2)

            # Step 2: Validate schema
            result = cli_runner.invoke(
                main, ["generate", "--schema", str(schema_file), "--validate-only"]
            )
            assert result.exit_code == 0

            # Step 3: Generate preview
            result = cli_runner.invoke(
                main,
                [
                    "generate",
                    "--schema",
                    str(schema_file),
                    "--rows",
                    "10",
                    "--preview",
                    "--seed",
                    "42",
                ],
            )
            assert result.exit_code == 0
            assert "Preview" in result.output

            # Step 4: Generate full dataset in multiple formats
            result = cli_runner.invoke(
                main,
                [
                    "generate",
                    "--schema",
                    str(schema_file),
                    "--rows",
                    "2500",
                    "--format",
                    "parquet",
                    "--out",
                    "complete_parquet",
                    "--batch-size",
                    "500",
                    "--max-rows-per-file",
                    "1000",
                    "--seed",
                    "42",
                ],
            )
            assert result.exit_code == 0

            # Verify output
            parquet_dir = Path("complete_parquet")
            assert parquet_dir.exists()

            # Check metadata
            with open(parquet_dir / "meta.json") as f:
                meta = json.load(f)
            assert meta["schema"]["collection_name"] == "complete_test"
            assert meta["generation_info"]["total_rows"] == 2500

            # Should have multiple files due to max_rows_per_file setting
            data_files = [f for f in parquet_dir.iterdir() if f.name != "meta.json"]
            assert len(data_files) >= 3

            # Step 5: Generate same data in JSON format
            result = cli_runner.invoke(
                main,
                [
                    "generate",
                    "--schema",
                    str(schema_file),
                    "--rows",
                    "100",
                    "--format",
                    "json",
                    "--out",
                    "complete_json",
                    "--seed",
                    "42",
                ],
            )
            assert result.exit_code == 0

            json_dir = Path("complete_json")
            assert json_dir.exists()

            # Verify JSON format files exist
            json_files = list(json_dir.glob("*.json"))
            data_json_files = [f for f in json_files if f.name != "meta.json"]
            assert len(data_json_files) > 0
