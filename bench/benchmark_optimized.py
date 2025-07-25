#!/usr/bin/env python3
"""
Benchmark script specifically for testing OptimizedDataWriter performance.

This script directly tests the optimized writer to measure raw generation speed
without the overhead of the full pipeline.
"""

import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
import psutil
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from milvus_ingest.models import CollectionSchema
from milvus_ingest.optimized_writer import OptimizedDataWriter


@dataclass
class OptimizedBenchmarkResult:
    """Container for optimized writer benchmark results."""

    operation: str
    field_type: str
    dimensions: int | None
    rows: int
    batch_size: int
    time_elapsed: float
    rows_per_second: float
    memory_used_mb: float

    @property
    def throughput_mb_per_sec(self) -> float:
        """Calculate throughput in MB/s for vector fields."""
        if self.dimensions and self.field_type.endswith("Vector"):
            # Estimate size based on field type
            bytes_per_element = {
                "FloatVector": 4,
                "BinaryVector": 1 / 8,
                "Float16Vector": 2,
                "BFloat16Vector": 2,
                "SparseFloatVector": 8,  # Approximate
            }.get(self.field_type, 4)

            total_bytes = self.rows * self.dimensions * bytes_per_element
            total_mb = total_bytes / (1024 * 1024)
            return total_mb / self.time_elapsed if self.time_elapsed > 0 else 0
        return 0


class OptimizedWriterBenchmark:
    """Benchmark suite for OptimizedDataWriter performance."""

    def __init__(self, console: Console):
        self.console = console
        self.results: list[OptimizedBenchmarkResult] = []
        self.process = psutil.Process()

    def measure_memory(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def benchmark_vector_generation(
        self, vector_type: str, dimensions: int, rows: int, batch_size: int = 50000
    ) -> OptimizedBenchmarkResult:
        """Benchmark vector field generation."""
        memory_before = self.measure_memory()

        # Create minimal schema for testing
        schema = {
            "collection_name": "bench_collection",
            "fields": [
                {"name": "id", "type": "Int64", "is_primary": True, "auto_id": True},
                {"name": "vector", "type": vector_type, "dim": dimensions},
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = OptimizedDataWriter(
                schema=CollectionSchema(**schema),
                output_dir=tmpdir,
                batch_size=batch_size,
                format="parquet",
            )

            start_time = time.time()

            # Generate data
            total_written = 0
            while total_written < rows:
                current_batch = min(batch_size, rows - total_written)
                writer._write_batch(total_written, current_batch)
                total_written += current_batch

            writer._finalize_current_file()

            end_time = time.time()
            memory_after = self.measure_memory()

        time_elapsed = end_time - start_time
        rows_per_second = rows / time_elapsed if time_elapsed > 0 else 0
        memory_used = memory_after - memory_before

        return OptimizedBenchmarkResult(
            operation="vector_generation",
            field_type=vector_type,
            dimensions=dimensions,
            rows=rows,
            batch_size=batch_size,
            time_elapsed=time_elapsed,
            rows_per_second=rows_per_second,
            memory_used_mb=memory_used,
        )

    def benchmark_field_types(
        self, rows: int = 100000, batch_size: int = 50000
    ) -> list[OptimizedBenchmarkResult]:
        """Benchmark different field types."""
        results = []

        # Test configurations
        test_configs = [
            # Numeric types
            ("Int8", None),
            ("Int64", None),
            ("Float", None),
            ("Double", None),
            # String types
            ("VarChar", 256),
            ("JSON", None),
            # Vector types with different dimensions
            ("FloatVector", 128),
            ("FloatVector", 768),
            ("FloatVector", 1536),
            ("BinaryVector", 128),
            ("Float16Vector", 768),
            ("BFloat16Vector", 768),
        ]

        for field_type, param in test_configs:
            memory_before = self.measure_memory()

            # Build schema based on field type
            fields = [
                {"name": "id", "type": "Int64", "is_primary": True, "auto_id": True}
            ]

            field_def = {"name": "test_field", "type": field_type}
            if param:
                if field_type == "VarChar":
                    field_def["max_length"] = param
                elif "Vector" in field_type:
                    field_def["dim"] = param

            fields.append(field_def)

            schema = {"collection_name": "bench_collection", "fields": fields}

            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    writer = OptimizedDataWriter(
                        schema=CollectionSchema(**schema),
                        output_dir=tmpdir,
                        batch_size=batch_size,
                        format="parquet",
                    )

                    start_time = time.time()

                    # Generate data
                    total_written = 0
                    while total_written < rows:
                        current_batch = min(batch_size, rows - total_written)
                        writer._write_batch(total_written, current_batch)
                        total_written += current_batch

                    writer._finalize_current_file()

                    end_time = time.time()
                    memory_after = self.measure_memory()

                time_elapsed = end_time - start_time
                rows_per_second = rows / time_elapsed if time_elapsed > 0 else 0
                memory_used = memory_after - memory_before

                result = OptimizedBenchmarkResult(
                    operation="field_type_test",
                    field_type=field_type,
                    dimensions=param if "Vector" in field_type else None,
                    rows=rows,
                    batch_size=batch_size,
                    time_elapsed=time_elapsed,
                    rows_per_second=rows_per_second,
                    memory_used_mb=memory_used,
                )
                results.append(result)

                self.console.print(
                    f"[green]✓[/green] {field_type}"
                    f"{f' (dim={param})' if 'Vector' in field_type else ''}: "
                    f"{rows_per_second:,.0f} rows/sec"
                )

            except Exception as e:
                self.console.print(f"[red]✗ {field_type}: {e}[/red]")

        return results

    def benchmark_scaling(
        self,
        vector_type: str = "FloatVector",
        dimensions: int = 768,
        batch_sizes: list[int] | None = None,
        rows: int = 1000000,
    ) -> list[OptimizedBenchmarkResult]:
        """Benchmark performance with different batch sizes."""
        if batch_sizes is None:
            batch_sizes = [10000, 50000, 100000, 200000]
        results = []

        for batch_size in batch_sizes:
            self.console.print(f"\nTesting batch size: {batch_size:,}")

            result = self.benchmark_vector_generation(
                vector_type=vector_type,
                dimensions=dimensions,
                rows=rows,
                batch_size=batch_size,
            )
            results.append(result)

            self.console.print(
                f"[green]✓[/green] {result.rows_per_second:,.0f} rows/sec | "
                f"{result.throughput_mb_per_sec:.1f} MB/sec | "
                f"Memory: {result.memory_used_mb:.1f} MB"
            )

        return results

    def display_results(self):
        """Display all benchmark results."""
        if not self.results:
            return

        # Group by operation type
        by_operation = {}
        for result in self.results:
            if result.operation not in by_operation:
                by_operation[result.operation] = []
            by_operation[result.operation].append(result)

        for operation, results in by_operation.items():
            table = Table(title=f"Benchmark Results - {operation}")

            if operation == "field_type_test":
                table.add_column("Field Type", style="cyan")
                table.add_column("Dimension", justify="right")
                table.add_column("Rows/sec", justify="right", style="green")
                table.add_column("Time (s)", justify="right")
                table.add_column("Memory (MB)", justify="right")

                for r in sorted(results, key=lambda x: x.rows_per_second, reverse=True):
                    table.add_row(
                        r.field_type,
                        str(r.dimensions) if r.dimensions else "-",
                        f"{r.rows_per_second:,.0f}",
                        f"{r.time_elapsed:.2f}",
                        f"{r.memory_used_mb:.1f}",
                    )

            elif operation == "vector_generation":
                table.add_column("Vector Type", style="cyan")
                table.add_column("Dimension", justify="right")
                table.add_column("Batch Size", justify="right")
                table.add_column("Rows/sec", justify="right", style="green")
                table.add_column("MB/sec", justify="right", style="yellow")
                table.add_column("Memory (MB)", justify="right")

                for r in results:
                    table.add_row(
                        r.field_type,
                        str(r.dimensions),
                        f"{r.batch_size:,}",
                        f"{r.rows_per_second:,.0f}",
                        f"{r.throughput_mb_per_sec:.1f}",
                        f"{r.memory_used_mb:.1f}",
                    )

            self.console.print(table)


@click.command()
@click.option(
    "--mode",
    type=click.Choice(["field-types", "vector-scaling", "batch-scaling", "all"]),
    default="all",
    help="Benchmark mode to run",
)
@click.option(
    "--rows", "-r", type=int, default=100000, help="Number of rows to generate per test"
)
@click.option(
    "--dimensions",
    "-d",
    type=int,
    default=768,
    help="Vector dimensions for scaling tests",
)
@click.option(
    "--vector-type",
    "-v",
    type=click.Choice(
        ["FloatVector", "BinaryVector", "Float16Vector", "BFloat16Vector"]
    ),
    default="FloatVector",
    help="Vector type for scaling tests",
)
def main(mode, rows, dimensions, vector_type):
    """Benchmark OptimizedDataWriter performance."""
    console = Console()

    console.print(
        Panel.fit(
            "[bold blue]OptimizedDataWriter Performance Benchmark[/bold blue]\n"
            f"Mode: {mode} | Rows: {rows:,} | Vector: {vector_type} ({dimensions}d)",
            title="Benchmark Configuration",
        )
    )

    benchmark = OptimizedWriterBenchmark(console)

    try:
        if mode in ["field-types", "all"]:
            console.print("\n[bold]Testing Field Type Performance[/bold]")
            results = benchmark.benchmark_field_types(rows=rows)
            benchmark.results.extend(results)

        if mode in ["vector-scaling", "all"]:
            console.print("\n[bold]Testing Vector Dimension Scaling[/bold]")
            for dim in [128, 256, 512, 768, 1024, 1536]:
                result = benchmark.benchmark_vector_generation(
                    vector_type=vector_type, dimensions=dim, rows=rows
                )
                benchmark.results.append(result)
                console.print(
                    f"[green]✓[/green] {dim}d: {result.rows_per_second:,.0f} rows/sec | "
                    f"{result.throughput_mb_per_sec:.1f} MB/sec"
                )

        if mode in ["batch-scaling", "all"]:
            console.print("\n[bold]Testing Batch Size Scaling[/bold]")
            results = benchmark.benchmark_scaling(
                vector_type=vector_type,
                dimensions=dimensions,
                rows=rows * 10,  # Use more rows for batch scaling
            )
            benchmark.results.extend(results)

        # Display summary
        console.print("\n[bold]Summary[/bold]")
        benchmark.display_results()

        # Performance insights
        if benchmark.results:
            avg_rows_per_sec = np.mean([r.rows_per_second for r in benchmark.results])
            max_rows_per_sec = max(r.rows_per_second for r in benchmark.results)

            console.print("\n[bold green]Performance Summary:[/bold green]")
            console.print(f"Average throughput: {avg_rows_per_sec:,.0f} rows/sec")
            console.print(f"Peak throughput: {max_rows_per_sec:,.0f} rows/sec")

    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark interrupted by user[/yellow]")


if __name__ == "__main__":
    main()
