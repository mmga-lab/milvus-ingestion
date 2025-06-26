"""Command-line interface for milvus-fake-data.

Usage::

    python -m milvus_fake_data.cli --schema schema.json --rows 1000  # default parquet
    python -m milvus_fake_data.cli --schema schema.yaml -f csv --out /tmp/mock.csv

The script is also installed as ``milvus-fake-data`` when the package is
installed via PDM/pip.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
from pymilvus import CollectionSchema, DataType, FieldSchema
from pymilvus.bulk_writer import BulkFileType, LocalBulkWriter

if TYPE_CHECKING:
    import pandas as pd

from .builtin_schemas import (
    list_builtin_schemas,
    save_schema_to_file,
)
from .generator import generate_mock_data
from .models import get_schema_help, validate_schema_data
from .rich_display import (
    display_error,
    display_schema_details,
    display_schema_list,
    display_schema_validation,
    display_success,
)
from .schema_manager import get_schema_manager

_OUTPUT_FORMATS = {"parquet", "csv", "json", "npy"}

# Default directory for generated data files: ~/.milvus-fake-data/data
DEFAULT_DATA_DIR = Path.home() / ".milvus-fake-data" / "data"


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--schema",
    "schema_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to schema JSON/YAML file.",
)
@click.option(
    "--rows",
    "-r",
    default=1000,
    show_default=True,
    type=int,
    help="Number of rows to generate.",
)
@click.option(
    "-f",
    "--format",
    "output_format",
    default="parquet",
    show_default=True,
    type=click.Choice(sorted(_OUTPUT_FORMATS)),
    help="Output file format.",
)
@click.option(
    "-p",
    "--preview",
    is_flag=True,
    help="Print first 5 rows to terminal after generation.",
)
@click.option(
    "--out",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output file path. Default: <collection_name>.<ext>",
)
@click.option("--seed", type=int, help="Random seed for reproducibility.")
@click.option(
    "--validate-only",
    is_flag=True,
    help="Only validate schema without generating data.",
)
@click.option(
    "--schema-help", is_flag=True, help="Show schema format help and examples."
)
@click.option(
    "--list-schemas", is_flag=True, help="List all available built-in schemas."
)
@click.option(
    "--builtin",
    "builtin_schema",
    help="Use a built-in schema (e.g., 'ecommerce', 'documents').",
)
@click.option(
    "--save-schema",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Save the schema to a file (use with --builtin).",
)
@click.option(
    "--add-schema",
    "add_schema",
    help="Add a custom schema (format: 'schema_id:schema_file.json').",
)
@click.option(
    "--show-schema",
    "show_schema",
    help="Show details of a schema (built-in or custom).",
)
@click.option(
    "--list-all-schemas", is_flag=True, help="List all schemas (built-in and custom)."
)
@click.option("--remove-schema", "remove_schema", help="Remove a custom schema by ID.")
def main(
    schema_path: Path | None,
    rows: int,
    output_format: str,
    output_path: Path | None,
    seed: int | None,
    preview: bool = False,
    validate_only: bool = False,
    schema_help: bool = False,
    list_schemas: bool = False,
    builtin_schema: str | None = None,
    save_schema: Path | None = None,
    add_schema: str | None = None,
    show_schema: str | None = None,
    list_all_schemas: bool = False,
    remove_schema: str | None = None,
) -> None:
    """Generate mock data from *schema_path* and write to disk using pandas or LocalBulkWriter."""
    # ------------------------------------------------------------------
    # Handle special flags first
    # ------------------------------------------------------------------
    if schema_help:
        click.echo(get_schema_help())
        return

    if list_schemas:
        schemas = list_builtin_schemas()
        display_schema_list(schemas, "Available Built-in Schemas")
        click.echo(
            "\nFor detailed schema information: milvus-fake-data --show-schema <schema_id>"
        )
        return

    if list_all_schemas:
        manager = get_schema_manager()
        all_schemas = manager.list_all_schemas()

        # Separate built-in and custom schemas
        builtin_schemas = {
            k: v for k, v in all_schemas.items() if manager.is_builtin_schema(k)
        }
        custom_schemas = {
            k: v for k, v in all_schemas.items() if not manager.is_builtin_schema(k)
        }

        if builtin_schemas:
            display_schema_list(builtin_schemas, "Built-in Schemas")

        if custom_schemas:
            display_schema_list(custom_schemas, "Custom Schemas")

        if not builtin_schemas and not custom_schemas:
            click.echo("No schemas found.")

        return

    if show_schema:
        manager = get_schema_manager()
        try:
            info = manager.get_schema_info(show_schema)
            if not info:
                display_error(
                    f"Schema '{show_schema}' not found.",
                    "Use --list-all-schemas to see available schemas.",
                )
                raise SystemExit(1)

            schema_data = manager.load_schema(show_schema)
            is_builtin = manager.is_builtin_schema(show_schema)

            display_schema_details(show_schema, info, schema_data, is_builtin)

        except Exception as e:
            display_error(f"Error showing schema: {e}")
            raise SystemExit(1) from e
        return

    if add_schema:
        manager = get_schema_manager()
        try:
            # Parse schema_id:schema_file format
            if ":" not in add_schema:
                display_error(
                    "Format should be: schema_id:schema_file.json",
                    "Example: --add-schema my_schema:my_schema.json",
                )
                raise SystemExit(1)

            schema_id, schema_file = add_schema.split(":", 1)
            schema_path = Path(schema_file)

            if not schema_path.exists():
                display_error(f"Schema file not found: {schema_path}")
                raise SystemExit(1)

            # Load and validate schema
            try:
                import yaml

                content = schema_path.read_text("utf-8")
                if schema_path.suffix.lower() in {".yaml", ".yml"}:
                    schema_data = yaml.safe_load(content)
                else:
                    schema_data = json.loads(content)
            except Exception as e:
                display_error(f"Error reading schema file: {e}")
                raise SystemExit(1) from e

            # Get additional info from user
            description = click.prompt(
                "Schema description (optional)", default="", show_default=False
            )
            use_cases_input = click.prompt(
                "Use cases (comma-separated, optional)", default="", show_default=False
            )
            use_cases = (
                [uc.strip() for uc in use_cases_input.split(",") if uc.strip()]
                if use_cases_input
                else []
            )

            manager.add_schema(schema_id, schema_data, description, use_cases)

            details = f"Description: {description or 'N/A'}\n"
            details += f"Use cases: {', '.join(use_cases) if use_cases else 'N/A'}\n"
            details += f"Usage: milvus-fake-data --show-schema {schema_id}"

            display_success(f"Added custom schema: {schema_id}", details)

        except ValueError as e:
            display_error(f"Error adding schema: {e}")
            raise SystemExit(1) from e
        except Exception as e:
            display_error(f"Unexpected error: {e}")
            raise SystemExit(1) from e
        return

    if remove_schema:
        manager = get_schema_manager()
        try:
            if not manager.schema_exists(remove_schema):
                display_error(f"Schema '{remove_schema}' does not exist.")
                raise SystemExit(1)

            if manager.is_builtin_schema(remove_schema):
                display_error(f"Cannot remove built-in schema '{remove_schema}'.")
                raise SystemExit(1)

            if click.confirm(
                f"Are you sure you want to remove schema '{remove_schema}'?"
            ):
                manager.remove_schema(remove_schema)
                display_success(f"Removed custom schema: {remove_schema}")
            else:
                click.echo("Cancelled.")

        except ValueError as e:
            display_error(f"Error removing schema: {e}")
            raise SystemExit(1) from e
        except Exception as e:
            display_error(f"Unexpected error: {e}")
            raise SystemExit(1) from e
        return

    # ------------------------------------------------------------------
    # Validate argument combinations
    # ------------------------------------------------------------------
    provided_args = [
        ("--schema", schema_path is not None),
        ("--builtin", builtin_schema is not None),
    ]
    provided_count = sum(provided for _, provided in provided_args)
    if provided_count == 0:
        click.echo("One of --schema or --builtin is required", err=True)
        raise SystemExit(1)
    if provided_count > 1:
        provided_names = [name for name, provided in provided_args if provided]
        click.echo(f"Cannot use {', '.join(provided_names)} together", err=True)
        raise SystemExit(1)

    # Handle built-in or custom schema
    if builtin_schema:
        manager = get_schema_manager()
        try:
            # Try to load from schema manager (supports both built-in and custom)
            schema_data = manager.load_schema(builtin_schema)
            schema_type = (
                "built-in" if manager.is_builtin_schema(builtin_schema) else "custom"
            )
            click.echo(f"✓ Loaded {schema_type} schema: {builtin_schema}")

            # Save schema to file if requested
            if save_schema:
                save_schema_to_file(schema_data, save_schema)
                click.echo(f"✓ Schema saved to: {save_schema}")
                return

            # Create temporary file for the schema
            with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as tmp:
                json.dump(schema_data, tmp)
                schema_path = Path(tmp.name)
        except ValueError as e:
            click.echo(f"✗ Error with schema: {e}", err=True)
            click.echo("Available schemas:", err=True)
            all_schemas = manager.list_all_schemas()
            for schema_id in sorted(all_schemas.keys()):
                schema_type = (
                    "built-in" if manager.is_builtin_schema(schema_id) else "custom"
                )
                click.echo(f"  - {schema_id} ({schema_type})", err=True)
            raise SystemExit(1) from e

    # builtin_schema case is already handled above, schema_path is set

    # Validate schema if --validate-only flag is used
    if validate_only:
        try:
            import yaml
            from pydantic import ValidationError

            assert schema_path is not None
            content = schema_path.read_text("utf-8")
            if schema_path.suffix.lower() in {".yaml", ".yml"}:
                schema_data = yaml.safe_load(content)
            else:
                schema_data = json.loads(content)

            validated_schema = validate_schema_data(schema_data)

            # Prepare validation info for rich display
            validation_info: dict[str, Any] = {}
            if isinstance(validated_schema, list):
                validation_info["fields_count"] = len(validated_schema)
                validation_info["fields"] = [
                    {
                        "name": field.name,
                        "type": field.type,
                        "is_primary": field.is_primary,
                    }
                    for field in validated_schema
                ]
            else:
                validation_info["collection_name"] = validated_schema.collection_name
                validation_info["fields_count"] = len(validated_schema.fields)
                validation_info["fields"] = [
                    {
                        "name": field.name,
                        "type": field.type,
                        "is_primary": field.is_primary,
                    }
                    for field in validated_schema.fields
                ]

            # Get schema ID for display
            if schema_path:
                schema_id = schema_path.stem
            elif builtin_schema:
                schema_id = builtin_schema
            else:
                schema_id = "schema"

            display_schema_validation(schema_id, validation_info)
            return
        except ValidationError as e:
            click.echo("✗ Schema validation failed:", err=True)
            for error in e.errors():
                loc = " -> ".join(str(x) for x in error["loc"])
                click.echo(f"  • {loc}: {error['msg']}", err=True)
            click.echo(
                f"\nFor help with schema format, run: {sys.argv[0]} --schema-help",
                err=True,
            )
            raise SystemExit(1) from e
        except Exception as e:
            click.echo(f"✗ Error reading schema file: {e}", err=True)
            raise SystemExit(1) from e

    assert schema_path is not None
    df = generate_mock_data(schema_path, rows=rows, seed=seed)

    if output_path is None:
        # derive default file name using default data directory (~/.milvus-fake-data/data)
        try:
            content = schema_path.read_text("utf-8")
            data = json.loads(content)
            collection_name: str | None = (
                data.get("collection_name") if isinstance(data, dict) else None
            )
        except Exception:
            collection_name = None
        base_name = collection_name or schema_path.stem
        # Ensure target directory exists
        DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DEFAULT_DATA_DIR / f"{base_name}.{output_format}"

    _save_with_bulk_writer(df, schema_path, output_path, output_format)
    click.echo(f"Saved {rows} rows to {output_path}")
    if preview:
        click.echo("\nPreview (top 5 rows):")
        click.echo(df)


def _save_with_bulk_writer(
    df: pd.DataFrame, schema_path: Path, output_path: Path, fmt: str
) -> None:
    """Save using Milvus LocalBulkWriter for ingestion-ready files."""
    # Build CollectionSchema from original schema file
    import yaml

    schema_dict = yaml.safe_load(schema_path.read_text("utf-8"))
    fields = schema_dict.get("fields", schema_dict)
    col_fields = []

    # Build DataType mapping safely (not all enums exist in every version)
    def _maybe(name: str) -> DataType | None:
        return getattr(DataType, name, None)

    dtype_map: dict[str, DataType] = {
        k: v
        for k, v in {
            "BOOL": _maybe("BOOL"),
            "INT8": _maybe("INT8"),
            "INT16": _maybe("INT16"),
            "INT32": _maybe("INT32"),
            "INT64": _maybe("INT64"),
            "FLOAT": _maybe("FLOAT"),
            "DOUBLE": _maybe("DOUBLE"),
            "VARCHAR": _maybe("VARCHAR"),
            "JSON": _maybe("JSON"),
            "ARRAY": _maybe("ARRAY"),
            "BINARY_VECTOR": _maybe("BINARY_VECTOR"),
            "FLOAT_VECTOR": _maybe("FLOAT_VECTOR"),
            "FLOAT16_VECTOR": _maybe("FLOAT16_VECTOR"),
            "BFLOAT16_VECTOR": _maybe("BFLOAT16_VECTOR"),
            "INT8_VECTOR": _maybe("INT8_VECTOR"),
            "SPARSE_FLOAT_VECTOR": _maybe("SPARSE_FLOAT_VECTOR"),
        }.items()
        if v is not None
    }
    for f in fields:
        t = f["type"].upper()
        if "VECTOR" in t and "_VECTOR" not in t:
            t = t.replace("VECTOR", "_VECTOR")
        dt = dtype_map.get(t)
        if dt is None:
            continue  # skip unsupported in bulk writer
        params = {}
        if "dim" in f:
            params["dim"] = int(f["dim"])
        if "max_length" in f:
            params["max_length"] = int(f["max_length"])

        # Handle ARRAY type special parameters
        if t == "ARRAY":
            if "element_type" in f:
                element_type = f["element_type"].upper()
                element_dt = dtype_map.get(element_type)
                if element_dt is not None:
                    params["element_type"] = element_dt
                    # For ARRAY with VARCHAR elements, max_length applies to the element
                    if element_type == "VARCHAR" and "max_length" in f:
                        params["max_length"] = int(f["max_length"])
            if "max_capacity" in f:
                params["max_capacity"] = int(f["max_capacity"])

        fs = FieldSchema(
            name=f["name"],
            dtype=dt,
            is_primary=f.get("is_primary", False),
            auto_id=f.get("auto_id", False),
            nullable=f.get("nullable", False),
            **params,
        )
        col_fields.append(fs)
    col_schema = CollectionSchema(fields=col_fields, enable_dynamic_field=True)

    file_type_map = {
        "csv": BulkFileType.CSV,
        "json": BulkFileType.JSON,
        "parquet": BulkFileType.PARQUET,
        "npy": BulkFileType.NUMPY,
    }
    file_type = file_type_map[fmt.lower()]

    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with LocalBulkWriter(
        schema=col_schema,
        local_path=str(output_path.parent),
        segment_size=128 * 1024 * 1024,
        file_type=file_type,
    ) as writer:
        for row in df.to_dict(orient="records"):
            writer.append_row(row)
        writer.commit()
        # writer.batch_files returns list[list[str]]
        generated = writer.batch_files[0][0]
        # Rename/move generated file to desired output_path
        os.replace(generated, output_path)


if __name__ == "__main__":  # pragma: no cover
    # Allow ``python -m milvus_fake_data``
    sys.exit(main())
