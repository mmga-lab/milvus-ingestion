"""Command-line interface for milvus-fake-data.

Usage::

    python -m milvus_fake_data.cli --schema schema.json --rows 1000  # default parquet
    python -m milvus_fake_data.cli --schema schema.yaml -f csv --out /tmp/mock.csv

The script is also installed as ``milvus-fake-data`` when the package is
installed via PDM/pip.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

import json
import tempfile

import click
import pandas as pd

from .generator import generate_mock_data

from pymilvus import FieldSchema, CollectionSchema, DataType
from pymilvus.bulk_writer import LocalBulkWriter, BulkFileType

_OUTPUT_FORMATS = {"parquet", "csv", "json", "npy"}


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--schema", "schema_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Path to schema JSON/YAML file.")
@click.option("-i", "--interactive", is_flag=True, help="Launch interactive schema builder.")
@click.option("--schema-out", "schema_out", type=click.Path(dir_okay=False, path_type=Path), help="Save interactive schema to file.")
@click.option("--rows", "-r", default=1000, show_default=True, type=int, help="Number of rows to generate.")
@click.option("-f", "--format", "output_format", default="parquet", show_default=True, type=click.Choice(sorted(_OUTPUT_FORMATS)), help="Output file format.")
@click.option("-p", "--preview", is_flag=True, help="Print first 5 rows to terminal after generation.")
@click.option("--out", "output_path", type=click.Path(dir_okay=False, path_type=Path), help="Output file path. Default: <collection_name>.<ext>")
@click.option("--seed", type=int, help="Random seed for reproducibility.")
def main(
    schema_path: Path | None,
    interactive: bool,
    schema_out: Path | None,
    rows: int,
    output_format: str,
    output_path: Path | None,
    seed: int | None,
    preview: bool = False,
) -> None:
    """Generate mock data from *schema_path* and write to disk using pandas or LocalBulkWriter."""
    # ------------------------------------------------------------------
    # Prepare schema (file path) depending on interactive flag
    # ------------------------------------------------------------------
    if interactive:
        if schema_path is not None:
            click.echo("Cannot use --schema together with --interactive", err=True)
            raise SystemExit(1)
        schema_dict = _interactive_schema_builder()
        if schema_out:
            schema_out.write_text(json.dumps(schema_dict, indent=2), encoding="utf-8")
            click.echo(f"Schema saved to {schema_out}")
        # write to temp file for generator
        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as tmp:
            json.dump(schema_dict, tmp)
            tmp_path = Path(tmp.name)
        schema_path = tmp_path
    else:
        if schema_path is None:
            click.echo("--schema is required unless --interactive is used", err=True)
            raise SystemExit(1)

    df = generate_mock_data(schema_path, rows=rows, seed=seed)

    if output_path is None:
        # derive default file name from schema collection or schema file stem
        try:
            content = Path(schema_path).read_text("utf-8")
            data = json.loads(content)
            collection_name: str | None = data.get("collection_name") if isinstance(data, dict) else None
        except Exception:
            data = {}
            collection_name = None
        base_name = collection_name or schema_path.stem
        output_path = schema_path.parent / f"{base_name}.{output_format}"

    _save_with_bulk_writer(df, schema_path, output_path, output_format)
    click.echo(f"Saved {rows} rows to {output_path}")
    if preview:
        click.echo("\nPreview (top 5 rows):")
        click.echo(df)


def _interactive_schema_builder() -> dict:
    """Launch command-line prompts to build a schema dict."""
    click.echo("\n=== Interactive Milvus Schema Builder ===")
    collection_name = click.prompt("Collection name", default="my_collection")
    fields: list[dict] = []
    while True:
        name = click.prompt("Field name")
        type_choices = [
            "Int8",
            "Int16",
            "Int32",
            "Int64",
            "Float",
            "Double",
            "Bool",
            "VarChar",
            "FloatVector",
            "BinaryVector",
            "Array",
            "JSON",
        ]
        f_type = click.prompt("Field type", type=click.Choice(type_choices, case_sensitive=False))
        field: dict = {"name": name, "type": f_type}
        # extra properties
        if click.confirm("Is primary key?", default=False):
            field["is_primary"] = True
            if click.confirm("Auto ID?", default=False):
                field["auto_id"] = True
        if click.confirm("Nullable?", default=False):
            field["nullable"] = True
        if f_type.lower() == "varchar":
            field["max_length"] = click.prompt("Max length", type=int, default=128)
        if "vector" in f_type.lower():
            field["dim"] = click.prompt("Dimension (dim)", type=int, default=128)
        if f_type.lower() == "array":
            elem_type = click.prompt("Element type", type=click.Choice([t for t in type_choices if t != "Array"], case_sensitive=False))
            field["element_type"] = elem_type
            field["max_capacity"] = click.prompt("Max capacity", type=int, default=5)
        fields.append(field)
        if not click.confirm("Add another field?", default=True):
            break
    schema = {"collection_name": collection_name, "fields": fields}
    click.echo("\nSchema preview:\n" + json.dumps(schema, indent=2))
    if not click.confirm("Looks good?", default=True):
        click.echo("Aborted.")
        raise SystemExit(1)
    return schema


def _save_with_bulk_writer(df: pd.DataFrame, schema_path: Path, output_path: Path, fmt: str) -> None:
    """Save using Milvus LocalBulkWriter for ingestion-ready files."""
    # Build CollectionSchema from original schema file
    import yaml  # type: ignore

    schema_dict = yaml.safe_load(schema_path.read_text("utf-8"))
    fields = schema_dict.get("fields", schema_dict)
    col_fields = []
    # Build DataType mapping safely (not all enums exist in every version)
    _maybe = lambda name: getattr(DataType, name, None)  # noqa: E731
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
