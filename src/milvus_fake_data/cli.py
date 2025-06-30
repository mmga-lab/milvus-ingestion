"""Command-line interface for milvus-fake-data.

Usage::

    # Data generation
    milvus-fake-data generate --schema schema.json --rows 1000
    milvus-fake-data generate --builtin simple --rows 100 --preview

    # Schema management
    milvus-fake-data schema list
    milvus-fake-data schema show simple
    milvus-fake-data schema add my_schema schema.json

    # Utilities
    milvus-fake-data clean --yes

The script is installed as ``milvus-fake-data`` when the package is
installed via PDM/pip.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import click
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .generator import generate_mock_data
from .logging_config import (
    get_logger,
    log_error_with_context,
    log_performance,
    setup_logging,
)
from .models import get_schema_help, validate_schema_data
from .rich_display import (
    display_confirmation_prompt,
    display_error,
    display_schema_details,
    display_schema_list,
    display_schema_preview,
    display_schema_validation,
    display_success,
)
from .schema_manager import get_schema_manager

_OUTPUT_FORMATS = {"parquet", "json"}

# Default directory for generated data files: ~/.milvus-fake-data/data
DEFAULT_DATA_DIR = Path.home() / ".milvus-fake-data" / "data"


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging with detailed debug information.",
)
@click.pass_context
def main(ctx: click.Context, verbose: bool = False) -> None:
    """Generate mock data for Milvus with schema management."""
    # Setup logging first
    setup_logging(verbose=verbose, log_level="DEBUG" if verbose else "INFO")
    logger = get_logger(__name__)

    # Store verbose in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    logger.info(
        "Starting milvus-fake-data CLI",
        extra={"verbose": verbose},
    )


@main.command()
@click.option(
    "--schema",
    "schema_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to schema JSON/YAML file.",
)
@click.option(
    "--builtin",
    "builtin_schema",
    help="Use a built-in schema (e.g., 'ecommerce', 'documents').",
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
    type=click.Path(file_okay=False, path_type=Path),
    help="Output directory path (will create directory with data files + meta.json). Default: <collection_name>/",
)
@click.option("--seed", type=int, help="Random seed for reproducibility.")
@click.option(
    "--validate-only",
    is_flag=True,
    help="Only validate schema without generating data.",
)
@click.option(
    "--no-progress",
    is_flag=True,
    help="Disable progress bar display for large datasets.",
)
@click.option(
    "--batch-size",
    "batch_size",
    default=50000,
    show_default=True,
    type=int,
    help="Number of rows to generate and process in each batch (larger batches = better performance).",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="Auto-confirm all prompts and proceed without interactive confirmation.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Force overwrite output directory if it exists.",
)
@click.option(
    "--max-file-size",
    "max_file_size_mb",
    default=256,
    show_default=True,
    type=int,
    help="Maximum size per file in MB (for automatic file partitioning).",
)
@click.option(
    "--max-rows-per-file",
    "max_rows_per_file",
    default=1000000,
    show_default=True,
    type=int,
    help="Maximum rows per file (for automatic file partitioning).",
)
@click.pass_context
def generate(
    ctx: click.Context,
    schema_path: Path | None = None,
    builtin_schema: str | None = None,
    rows: int = 1000,
    output_format: str = "parquet",
    output_path: Path | None = None,
    seed: int | None = None,
    preview: bool = False,
    validate_only: bool = False,
    no_progress: bool = False,
    batch_size: int = 50000,
    yes: bool = False,
    force: bool = False,
    max_file_size_mb: int = 256,
    max_rows_per_file: int = 1000000,
) -> None:
    """Generate high-performance mock data from schema using optimized vectorized operations.

    This tool is optimized for large-scale data generation with NumPy vectorized operations
    (automatically utilizing multiple CPU cores through optimized BLAS), efficient memory
    management, and high-speed file I/O. Uses intelligent file partitioning for maximum
    performance on large datasets.

    Output is always a directory containing data files and collection schema.json file.
    """
    verbose = ctx.obj["verbose"]
    logger = get_logger(__name__)

    logger.info(
        "Starting data generation",
        extra={
            "rows": rows,
            "format": output_format,
            "verbose": verbose,
            "batch_size": batch_size,
            "seed": seed,
        },
    )

    # Validate argument combinations
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
        logger.debug("Loading builtin schema", schema_name=builtin_schema)
        manager = get_schema_manager()
        try:
            # Try to load from schema manager (supports both built-in and custom)
            schema_data = manager.load_schema(builtin_schema)
            schema_type = (
                "built-in" if manager.is_builtin_schema(builtin_schema) else "custom"
            )
            logger.info(
                "Schema loaded successfully",
                schema_name=builtin_schema,
                schema_type=schema_type,
            )
            click.echo(f"✓ Loaded {schema_type} schema: {builtin_schema}")

            # Create temporary file for the schema
            with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as tmp:
                json.dump(schema_data, tmp)
                schema_path = Path(tmp.name)
                logger.debug(
                    "Created temporary schema file", temp_file=str(schema_path)
                )
        except ValueError as e:
            log_error_with_context(
                e, {"schema_name": builtin_schema, "operation": "load_schema"}
            )
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

    # Load and validate schema for preview
    logger.debug("Loading schema for preview", schema_file=str(schema_path))
    try:
        import yaml

        content = schema_path.read_text("utf-8")
        if schema_path.suffix.lower() in {".yaml", ".yml"}:
            schema_data = yaml.safe_load(content)
        else:
            schema_data = json.loads(content)

        # Validate schema
        validated_schema = validate_schema_data(schema_data)

        # Extract fields and collection info
        if isinstance(validated_schema, list):
            fields = [
                field.model_dump(exclude_none=True, exclude_unset=True)
                for field in validated_schema
            ]
            collection_name = None
            schema_display_name = schema_path.stem
        else:
            fields = [
                field.model_dump(exclude_none=True, exclude_unset=True)
                for field in validated_schema.fields
            ]
            collection_name = validated_schema.collection_name
            schema_display_name = collection_name or schema_path.stem

        # Display schema preview
        generation_config = {
            "rows": rows,
            "batch_size": batch_size,
            "seed": seed,
            "format": output_format,
        }

        display_schema_preview(
            schema_name=schema_display_name,
            collection_name=collection_name,
            fields=fields,
            generation_config=generation_config,
        )

        # Add confirmation prompt (unless auto-confirmed with --yes or in preview mode)
        if not yes and not preview:
            # Calculate estimated size for confirmation display
            from .rich_display import _estimate_output_size

            estimated_size = _estimate_output_size(fields, rows)

            # Display formatted confirmation prompt
            display_confirmation_prompt(rows, estimated_size)
            # Simple yes/no prompt
            if not click.confirm("Proceed with data generation?", default=True):
                logger.info("Data generation cancelled by user")
                click.echo("❌ Operation cancelled.")
                return
            else:
                logger.info("Data generation confirmed by user")
        else:
            logger.info("Data generation auto-confirmed with --yes flag")
    except Exception as e:
        logger.error("Failed to load or validate schema for preview", error=str(e))
        click.echo(f"✗ Error loading schema: {e}", err=True)
        raise SystemExit(1) from e

    logger.info(
        "Starting data generation",
        schema_file=str(schema_path),
        rows=rows,
        batch_size=batch_size,
    )
    if output_path is None:
        # derive default file name using default data directory (~/.milvus-fake-data/data)
        try:
            content = schema_path.read_text("utf-8")
            data = json.loads(content)
            schema_collection_name: str | None = (
                data.get("collection_name") if isinstance(data, dict) else None
            )
        except Exception:
            schema_collection_name = None
        base_name = schema_collection_name or schema_path.stem
        # Ensure target directory exists
        DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DEFAULT_DATA_DIR / base_name
        logger.debug(
            "Output path determined", output_path=str(output_path), base_name=base_name
        )

    # Handle force cleanup of output directory
    if force and output_path.exists():
        import shutil

        logger.info(
            "Force cleanup enabled, removing existing output directory",
            output_path=str(output_path),
        )
        shutil.rmtree(output_path)
        logger.debug("Output directory removed successfully")

    # Use high-performance data generation (default and only mode)
    logger.info(
        "Starting high-performance data generation",
        output_dir=str(output_path),
        format=output_format,
        rows=rows,
    )
    try:
        # High-performance parallel generator (default and only mode)
        logger.info("Using vectorized high-performance generator")
        _save_with_high_performance_generator(
            schema_path,
            rows,
            output_path,
            output_format,
            batch_size=batch_size,
            seed=seed,
            show_progress=not no_progress,
            max_file_size_mb=max_file_size_mb,
            max_rows_per_file=max_rows_per_file,
        )
        # Calculate directory size for logging (output is always a directory now)
        total_size = sum(
            f.stat().st_size for f in output_path.rglob("*") if f.is_file()
        )
        file_size_mb = total_size / (1024 * 1024)

        logger.info(
            "Data generation completed successfully",
            rows=rows,
            output_file=str(output_path),
            file_size_mb=file_size_mb,
        )
        # Output is always a directory now
        click.echo(f"Saved {rows} rows to directory {output_path.resolve()}")
    except Exception as e:
        log_error_with_context(
            e,
            {
                "operation": "data_generation",
                "rows": rows,
                "output_path": str(output_path),
                "batch_size": batch_size,
            },
        )
        raise

    if preview:
        # For preview, generate a small sample
        preview_df = generate_mock_data(
            schema_path, rows=min(5, rows), seed=seed, show_progress=False
        )
        click.echo("\nPreview (top 5 rows):")
        click.echo(preview_df.head())


@main.group()
def schema() -> None:
    """Manage schemas (built-in and custom)."""
    pass


@schema.command("list")
def list_schemas() -> None:
    """List all available schemas."""
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

    click.echo(
        "\nFor detailed schema information: milvus-fake-data schema show <schema_id>"
    )


@schema.command()
@click.argument("schema_id")
def show(schema_id: str) -> None:
    """Show details of a specific schema."""
    manager = get_schema_manager()
    try:
        info = manager.get_schema_info(schema_id)
        if not info:
            display_error(
                f"Schema '{schema_id}' not found.",
                "Use 'milvus-fake-data schema list' to see available schemas.",
            )
            raise SystemExit(1)

        schema_data = manager.load_schema(schema_id)
        is_builtin = manager.is_builtin_schema(schema_id)

        display_schema_details(schema_id, info, schema_data, is_builtin)

    except Exception as e:
        display_error(f"Error showing schema: {e}")
        raise SystemExit(1) from e


@schema.command()
@click.argument("schema_id")
@click.argument(
    "schema_file", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
def add(schema_id: str, schema_file: Path) -> None:
    """Add a custom schema."""
    manager = get_schema_manager()
    try:
        # Load and validate schema
        try:
            import yaml

            content = schema_file.read_text("utf-8")
            if schema_file.suffix.lower() in {".yaml", ".yml"}:
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
        details += f"Usage: milvus-fake-data schema show {schema_id}"

        display_success(f"Added custom schema: {schema_id}", details)

    except ValueError as e:
        display_error(f"Error adding schema: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        display_error(f"Unexpected error: {e}")
        raise SystemExit(1) from e


@schema.command()
@click.argument("schema_id")
def remove(schema_id: str) -> None:
    """Remove a custom schema."""
    manager = get_schema_manager()
    try:
        if not manager.schema_exists(schema_id):
            display_error(f"Schema '{schema_id}' does not exist.")
            raise SystemExit(1)

        if manager.is_builtin_schema(schema_id):
            display_error(f"Cannot remove built-in schema '{schema_id}'.")
            raise SystemExit(1)

        if click.confirm(f"Are you sure you want to remove schema '{schema_id}'?"):
            manager.remove_schema(schema_id)
            display_success(f"Removed custom schema: {schema_id}")
        else:
            click.echo("Cancelled.")

    except ValueError as e:
        display_error(f"Error removing schema: {e}")
        raise SystemExit(1) from e
    except Exception as e:
        display_error(f"Unexpected error: {e}")
        raise SystemExit(1) from e


@schema.command()
def help() -> None:
    """Show schema format help and examples."""
    click.echo(get_schema_help())


@main.command()
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="Auto-confirm all prompts and proceed without interactive confirmation.",
)
def clean(yes: bool = False) -> None:
    """Clean up generated output files."""
    logger = get_logger(__name__)
    _handle_clean_command(yes, logger)


def _save_with_high_performance_generator(
    schema_path: Path,
    rows: int,
    output_path: Path,
    fmt: str,
    batch_size: int = 10000,
    seed: int | None = None,
    show_progress: bool = True,
    max_file_size_mb: int = 256,
    max_rows_per_file: int = 1000000,
) -> None:
    """Save using high-performance vectorized generator optimized for large-scale data."""
    import time

    from .optimized_writer import generate_data_optimized

    logger = get_logger(__name__)
    start_time = time.time()

    # Use larger batch size for high-performance mode
    optimized_batch_size = max(batch_size, 50000)

    logger.info(
        "Starting high-performance vectorized generator",
        schema_file=str(schema_path),
        output_dir=str(output_path),
        format=fmt,
        rows=rows,
        batch_size=optimized_batch_size,
    )

    try:
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                TimeElapsedColumn(),
            ) as progress:
                task = progress.add_task(
                    "Generating data with high-performance mode...", total=rows
                )

                # Progress callback to update the progress bar
                def update_progress(completed_rows: int) -> None:
                    progress.update(task, completed=completed_rows)

                # Run optimized generator with progress callback
                files_created = generate_data_optimized(
                    schema_path=schema_path,
                    rows=rows,
                    output_dir=output_path,
                    format=fmt,
                    batch_size=optimized_batch_size,
                    seed=seed,
                    max_file_size_mb=max_file_size_mb,
                    max_rows_per_file=max_rows_per_file,
                    progress_callback=update_progress,
                )

                # Ensure progress shows 100% at the end
                progress.update(task, completed=rows)
        else:
            # Run without progress bar
            files_created = generate_data_optimized(
                schema_path=schema_path,
                rows=rows,
                output_dir=output_path,
                format=fmt,
                batch_size=optimized_batch_size,
                seed=seed,
                max_file_size_mb=max_file_size_mb,
                max_rows_per_file=max_rows_per_file,
            )

        # Log performance metrics
        total_time = time.time() - start_time
        total_size = sum(
            Path(f).stat().st_size for f in files_created if Path(f).exists()
        )
        file_size_mb = total_size / (1024 * 1024)

        log_performance(
            "high_performance_generator",
            total_time,
            rows=rows,
            batch_size=optimized_batch_size,
            file_size_mb=file_size_mb,
        )
        logger.info(
            "High-performance generator completed",
            total_rows=rows,
            output_dir=str(output_path),
            file_size_mb=file_size_mb,
            duration_seconds=total_time,
            rows_per_second=rows / total_time if total_time > 0 else 0,
        )
    except Exception as e:
        logger.error(f"High-performance generator failed: {e}")
        raise


def _handle_clean_command(yes: bool, logger: Any) -> None:
    """Handle the clean command to remove generated output files."""
    # Default data directory
    data_dir = DEFAULT_DATA_DIR

    # Collect paths to clean
    paths_to_clean: list[tuple[str, Path | list[Path]]] = []

    # Check data directory
    if data_dir.exists() and any(data_dir.iterdir()):
        paths_to_clean.append(("Generated data directory", data_dir))

    # Check for any generated files in current directory
    current_dir = Path.cwd()
    generated_files: list[Path] = []
    for pattern in ["*.parquet", "*.csv", "*.npy"]:
        generated_files.extend(current_dir.glob(pattern))

    # For JSON files, be more selective to avoid schema files
    json_files = current_dir.glob("*.json")
    excluded_json_files = {
        "package.json",
        "pyproject.toml",
        "schema.json",
        "example_schema.json",
        "demo.json",
        "meta.json",
    }
    # Include JSON files that look like generated data files
    for json_file in json_files:
        if (
            json_file.name not in excluded_json_files
            and "schema" not in json_file.name.lower()
        ):
            generated_files.append(json_file)

    if generated_files:
        paths_to_clean.append(("Generated files in current directory", generated_files))

    if not paths_to_clean:
        display_success("No files or directories to clean.")
        logger.info("Clean command completed - nothing to clean")
        return

    # Display what will be cleaned
    click.echo("The following will be cleaned:")
    for description, path in paths_to_clean:
        if isinstance(path, list):
            click.echo(f"  • {description}:")
            for file_path in path:
                click.echo(f"    - {file_path}")
        else:
            click.echo(f"  • {description}: {path}")

    # Confirm deletion unless --yes flag is used
    if not yes:
        click.echo()
        if not click.confirm(
            "Are you sure you want to delete these files and directories?"
        ):
            click.echo("Clean operation cancelled.")
            logger.info("Clean command cancelled by user")
            return

    # Perform cleanup
    cleaned_items: list[str] = []
    errors: list[str] = []

    for _, path in paths_to_clean:
        try:
            if isinstance(path, list):
                # Handle list of files
                for file_path in path:
                    if file_path.exists():
                        file_path.unlink()
                        cleaned_items.append(str(file_path))
                        logger.debug(f"Removed file: {file_path}")
            else:
                # Handle directory
                if path.exists():
                    shutil.rmtree(path)
                    cleaned_items.append(str(path))
                    logger.debug(f"Removed directory: {path}")
        except Exception as e:
            error_msg = f"Failed to remove {path}: {e}"
            errors.append(error_msg)
            logger.error(error_msg)

    # Report results
    if cleaned_items:
        display_success(f"Successfully cleaned {len(cleaned_items)} items.")
        logger.info(
            "Clean command completed successfully",
            cleaned_items=len(cleaned_items),
            errors=len(errors),
        )

    if errors:
        click.echo("\nErrors occurred during cleanup:")
        for error in errors:
            display_error(error)


if __name__ == "__main__":  # pragma: no cover
    # Allow ``python -m milvus_fake_data``
    sys.exit(main())
