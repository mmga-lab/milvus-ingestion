"""Tests for the CLI module."""

import json
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from milvus_fake_data.cli import main


@pytest.fixture
def cli_runner():
    """CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_schema():
    """Sample schema for CLI tests."""
    return {
        "collection_name": "test_cli",
        "fields": [
            {"name": "id", "type": "Int64", "is_primary": True},
            {"name": "text", "type": "VarChar", "max_length": 50},
            {"name": "vector", "type": "FloatVector", "dim": 4},
        ],
    }


def test_cli_basic_usage(cli_runner, sample_schema):
    """Test basic CLI usage."""
    with tempfile.TemporaryDirectory() as temp_dir:
        schema_path = Path(temp_dir) / "schema.json"
        with open(schema_path, "w") as f:
            json.dump(sample_schema, f)

        result = cli_runner.invoke(
            main, ["--schema", str(schema_path), "--rows", "5", "--seed", "42"]
        )

        assert result.exit_code == 0
        assert "Saved 5 rows" in result.output

        # Check output file was created in default data directory
        from milvus_fake_data.cli import DEFAULT_DATA_DIR
        output_file = DEFAULT_DATA_DIR / "test_cli.parquet"
        assert output_file.exists()


def test_cli_csv_output(cli_runner, sample_schema):
    """Test CSV output format."""
    with tempfile.TemporaryDirectory() as temp_dir:
        schema_path = Path(temp_dir) / "schema.json"
        output_path = Path(temp_dir) / "output.csv"

        with open(schema_path, "w") as f:
            json.dump(sample_schema, f)

        result = cli_runner.invoke(
            main,
            [
                "--schema",
                str(schema_path),
                "--rows",
                "3",
                "-f",
                "csv",
                "--out",
                str(output_path),
                "--seed",
                "42",
            ],
        )

        assert result.exit_code == 0
        assert output_path.exists()


def test_cli_preview_option(cli_runner, sample_schema):
    """Test preview option."""
    with tempfile.TemporaryDirectory() as temp_dir:
        schema_path = Path(temp_dir) / "schema.json"

        with open(schema_path, "w") as f:
            json.dump(sample_schema, f)

        result = cli_runner.invoke(
            main,
            ["--schema", str(schema_path), "--rows", "2", "--preview", "--seed", "42"],
        )

        assert result.exit_code == 0
        assert "Preview (top 5 rows):" in result.output


def test_cli_missing_schema(cli_runner):
    """Test error when schema is missing."""
    result = cli_runner.invoke(main, ["--rows", "1"])

    assert result.exit_code == 1
    assert "One of --schema or --builtin is required" in result.output


def test_cli_nonexistent_schema(cli_runner):
    """Test error for non-existent schema file."""
    result = cli_runner.invoke(
        main, ["--schema", "/non/existent/file.json", "--rows", "1"]
    )

    assert result.exit_code != 0
