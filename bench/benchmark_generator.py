#!/usr/bin/env python3
"""
Benchmark script for testing data generation speed.

This script measures the performance of milvus-ingest data generation
across different configurations and schemas.
"""

import json
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import click
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.table import Table

# Add parent directory to path to import milvus_ingest
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from milvus_ingest.generator import generate_mock_data
from milvus_ingest.schema_manager import get_schema_manager


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    schema_name: str
    total_rows: int
    batch_size: int
    format: str
    time_elapsed: float
    rows_per_second: float
    file_count: int
    total_size_mb: float
    memory_peak_mb: float


class DataGeneratorBenchmark:
    """Benchmark suite for data generation performance."""

    def __init__(self, console: Console):
        self.console = console
        self.results: list[BenchmarkResult] = []

    def run_single_benchmark(
        self,
        schema_name: str,
        total_rows: int,
        batch_size: int = 50000,
        format: str = "parquet",
        warmup: bool = True,
    ) -> BenchmarkResult:
        """Run a single benchmark test."""
        # Get schema
        manager = get_schema_manager()
        schema = manager.get_schema(schema_name)

        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"

            # Warmup run if requested
            if warmup and total_rows >= 10000:
                warmup_rows = min(10000, total_rows // 10)
                self.console.print(f"[dim]Warming up with {warmup_rows} rows...[/dim]")
                with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
                    json.dump(schema, f)
                    f.flush()
                    generate_mock_data(
                        f.name,
                        rows=warmup_rows,
                        batch_size=batch_size,
                        output_format=format,
                        output_dir=str(output_dir),
                        show_progress=False,
                    )
                shutil.rmtree(output_dir, ignore_errors=True)

            # Measure memory before
            import psutil

            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            # Run actual benchmark
            start_time = time.time()

            with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
                json.dump(schema, f)
                f.flush()

                generate_mock_data(
                    f.name,
                    rows=total_rows,
                    batch_size=batch_size,
                    output_format=format,
                    output_dir=str(output_dir),
                    show_progress=False,
                )

            end_time = time.time()
            time_elapsed = end_time - start_time

            # Measure memory peak
            memory_peak = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_peak - memory_before

            # Calculate metrics
            rows_per_second = total_rows / time_elapsed

            # Count files and calculate total size
            file_count = 0
            total_size = 0
            if output_dir.exists():
                for file in output_dir.glob(f"*.{format}"):
                    file_count += 1
                    total_size += file.stat().st_size

            total_size_mb = total_size / (1024 * 1024)

            return BenchmarkResult(
                schema_name=schema_name,
                total_rows=total_rows,
                batch_size=batch_size,
                format=format,
                time_elapsed=time_elapsed,
                rows_per_second=rows_per_second,
                file_count=file_count,
                total_size_mb=total_size_mb,
                memory_peak_mb=memory_used,
            )

    def run_schema_benchmarks(
        self,
        schemas: list[str],
        row_counts: list[int],
        batch_sizes: list[int],
        formats: list[str] | None = None,
    ):
        """Run benchmarks for multiple schemas and configurations."""
        if formats is None:
            formats = ["parquet"]
        total_tests = len(schemas) * len(row_counts) * len(batch_sizes) * len(formats)

        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("Running benchmarks...", total=total_tests)

            for schema in schemas:
                for rows in row_counts:
                    for batch_size in batch_sizes:
                        for format in formats:
                            try:
                                self.console.print(
                                    f"\n[bold]Testing:[/bold] {schema} | "
                                    f"{rows:,} rows | batch={batch_size:,} | {format}"
                                )

                                result = self.run_single_benchmark(
                                    schema_name=schema,
                                    total_rows=rows,
                                    batch_size=batch_size,
                                    format=format,
                                )

                                self.results.append(result)

                                self.console.print(
                                    f"[green]✓[/green] {result.rows_per_second:,.0f} rows/sec | "
                                    f"{result.time_elapsed:.2f}s | "
                                    f"{result.file_count} files | "
                                    f"{result.total_size_mb:.1f}MB"
                                )

                            except Exception as e:
                                self.console.print(f"[red]✗ Error: {e}[/red]")

                            progress.advance(task)

    def display_results(self):
        """Display benchmark results in a table."""
        if not self.results:
            self.console.print("[yellow]No results to display[/yellow]")
            return

        # Group results by schema
        schema_groups = {}
        for result in self.results:
            if result.schema_name not in schema_groups:
                schema_groups[result.schema_name] = []
            schema_groups[result.schema_name].append(result)

        # Display results for each schema
        for schema_name, results in schema_groups.items():
            self.console.print(f"\n[bold]Schema: {schema_name}[/bold]")

            table = Table(title=f"Performance Results - {schema_name}")
            table.add_column("Rows", justify="right")
            table.add_column("Batch Size", justify="right")
            table.add_column("Format", justify="center")
            table.add_column("Time (s)", justify="right")
            table.add_column("Rows/sec", justify="right")
            table.add_column("Files", justify="right")
            table.add_column("Size (MB)", justify="right")
            table.add_column("Memory (MB)", justify="right")

            # Sort by rows then batch size
            results.sort(key=lambda r: (r.total_rows, r.batch_size))

            for result in results:
                table.add_row(
                    f"{result.total_rows:,}",
                    f"{result.batch_size:,}",
                    result.format,
                    f"{result.time_elapsed:.2f}",
                    f"{result.rows_per_second:,.0f}",
                    str(result.file_count),
                    f"{result.total_size_mb:.1f}",
                    f"{result.memory_peak_mb:.1f}",
                )

            self.console.print(table)

    def save_results(self, output_file: str):
        """Save benchmark results to a CSV file."""
        if not self.results:
            return

        df = pd.DataFrame(
            [
                {
                    "schema": r.schema_name,
                    "total_rows": r.total_rows,
                    "batch_size": r.batch_size,
                    "format": r.format,
                    "time_seconds": r.time_elapsed,
                    "rows_per_second": r.rows_per_second,
                    "file_count": r.file_count,
                    "total_size_mb": r.total_size_mb,
                    "memory_peak_mb": r.memory_peak_mb,
                    "mb_per_second": r.total_size_mb / r.time_elapsed
                    if r.time_elapsed > 0
                    else 0,
                }
                for r in self.results
            ]
        )

        df.to_csv(output_file, index=False)
        self.console.print(f"\n[green]Results saved to {output_file}[/green]")


@click.command()
@click.option(
    "--schemas",
    "-s",
    multiple=True,
    default=["simple", "ecommerce", "documents"],
    help="Schemas to benchmark (can specify multiple)",
)
@click.option(
    "--rows",
    "-r",
    multiple=True,
    type=int,
    default=[10000, 100000, 1000000],
    help="Row counts to test (can specify multiple)",
)
@click.option(
    "--batch-sizes",
    "-b",
    multiple=True,
    type=int,
    default=[10000, 50000, 100000],
    help="Batch sizes to test (can specify multiple)",
)
@click.option(
    "--formats",
    "-f",
    multiple=True,
    type=click.Choice(["parquet", "json"]),
    default=["parquet"],
    help="Output formats to test",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="benchmark_results.csv",
    help="Output CSV file for results",
)
@click.option("--quick", is_flag=True, help="Run quick benchmark with smaller datasets")
def main(schemas, rows, batch_sizes, formats, output, quick):
    """Benchmark milvus-ingest data generation performance."""
    console = Console()

    console.print("[bold blue]Milvus Ingest Performance Benchmark[/bold blue]\n")

    # Quick mode overrides
    if quick:
        schemas = ["simple"]
        rows = [1000, 10000, 100000]
        batch_sizes = [10000, 50000]
        formats = ["parquet"]

    # Validate schemas exist
    manager = get_schema_manager()
    all_schemas = manager.list_all_schemas()

    valid_schemas = []
    for schema in schemas:
        if schema in all_schemas:
            valid_schemas.append(schema)
        else:
            console.print(
                f"[yellow]Warning: Schema '{schema}' not found, skipping[/yellow]"
            )

    if not valid_schemas:
        console.print("[red]No valid schemas to benchmark![/red]")
        return

    # Run benchmarks
    benchmark = DataGeneratorBenchmark(console)

    console.print(f"Testing schemas: {', '.join(valid_schemas)}")
    console.print(f"Row counts: {', '.join(f'{r:,}' for r in rows)}")
    console.print(f"Batch sizes: {', '.join(f'{b:,}' for b in batch_sizes)}")
    console.print(f"Formats: {', '.join(formats)}\n")

    try:
        benchmark.run_schema_benchmarks(
            schemas=valid_schemas,
            row_counts=list(rows),
            batch_sizes=list(batch_sizes),
            formats=list(formats),
        )

        # Display and save results
        benchmark.display_results()
        benchmark.save_results(output)

    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark interrupted by user[/yellow]")
        if benchmark.results:
            benchmark.display_results()
            benchmark.save_results(output)


if __name__ == "__main__":
    main()
