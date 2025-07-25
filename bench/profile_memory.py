#!/usr/bin/env python3
"""
Memory profiling script for data generation.

This script profiles memory usage during data generation to identify
potential memory leaks and optimization opportunities.
"""

import gc
import json
import sys
import tempfile
import time
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import psutil
from memory_profiler import memory_usage
from rich.console import Console

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from milvus_ingest.generator import generate_mock_data
from milvus_ingest.models import CollectionSchema
from milvus_ingest.optimized_writer import OptimizedDataWriter
from milvus_ingest.schema_manager import get_schema_manager


class MemoryProfiler:
    """Memory profiling utilities for data generation."""

    def __init__(self, console: Console):
        self.console = console
        self.process = psutil.Process()

    def profile_generation(
        self,
        schema_name: str,
        rows: int,
        batch_size: int = 50000,
        interval: float = 0.1,
    ) -> tuple[list[float], list[float]]:
        """Profile memory usage during data generation."""

        def generation_task():
            """Task to profile."""
            manager = get_schema_manager()
            schema = manager.get_schema(schema_name)

            with (
                tempfile.TemporaryDirectory() as tmpdir,
                tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f,
            ):
                json.dump(schema, f)
                f.flush()

                generate_mock_data(
                    f.name,
                    rows=rows,
                    batch_size=batch_size,
                    output_format="parquet",
                    output_dir=tmpdir,
                    show_progress=False,
                )

        # Profile memory usage
        self.console.print(f"Profiling memory for {schema_name} with {rows:,} rows...")

        start_time = time.time()
        mem_usage = memory_usage(
            generation_task,
            interval=interval,
            timestamps=True,
            include_children=True,
            multiprocess=True,
        )

        # Extract memory values and timestamps
        if mem_usage and isinstance(mem_usage[0], tuple):
            memories = [m[0] for m in mem_usage]
            timestamps = [m[1] - start_time for m in mem_usage]
        else:
            memories = mem_usage
            timestamps = list(range(len(memories)))

        return timestamps, memories

    def profile_batch_sizes(
        self, schema_name: str, rows: int, batch_sizes: list[int]
    ) -> dict[int, tuple[list[float], list[float]]]:
        """Profile memory usage for different batch sizes."""
        results = {}

        for batch_size in batch_sizes:
            self.console.print(f"\nProfiling batch size: {batch_size:,}")

            # Force garbage collection before each test
            gc.collect()
            time.sleep(0.5)

            timestamps, memories = self.profile_generation(
                schema_name=schema_name, rows=rows, batch_size=batch_size
            )

            results[batch_size] = (timestamps, memories)

            # Report peak memory
            peak_memory = max(memories) if memories else 0
            self.console.print(f"Peak memory: {peak_memory:.1f} MB")

        return results

    def analyze_memory_pattern(
        self, timestamps: list[float], memories: list[float]
    ) -> dict[str, float]:
        """Analyze memory usage pattern."""
        if not memories:
            return {}

        # Calculate statistics
        peak_memory = max(memories)
        avg_memory = np.mean(memories)
        baseline = memories[0] if memories else 0
        peak_increase = peak_memory - baseline

        # Detect potential leaks
        # Simple heuristic: if memory at end is >20% higher than beginning
        potential_leak = (
            (memories[-1] - baseline) / baseline > 0.2 if baseline > 0 else False
        )

        # Calculate memory growth rate
        if len(memories) > 1:
            growth_rate = (memories[-1] - memories[0]) / (
                timestamps[-1] - timestamps[0]
            )
        else:
            growth_rate = 0

        return {
            "peak_memory_mb": peak_memory,
            "avg_memory_mb": avg_memory,
            "baseline_mb": baseline,
            "peak_increase_mb": peak_increase,
            "growth_rate_mb_per_sec": growth_rate,
            "potential_leak": potential_leak,
            "final_memory_mb": memories[-1] if memories else 0,
        }

    def plot_memory_usage(
        self,
        results: dict[int, tuple[list[float], list[float]]],
        output_file: str = "memory_profile.png",
    ):
        """Plot memory usage over time for different configurations."""
        plt.figure(figsize=(12, 8))

        for batch_size, (timestamps, memories) in results.items():
            plt.plot(timestamps, memories, label=f"Batch size: {batch_size:,}")

        plt.xlabel("Time (seconds)")
        plt.ylabel("Memory Usage (MB)")
        plt.title("Memory Usage During Data Generation")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(output_file, dpi=300)
        self.console.print(
            f"\n[green]Memory profile plot saved to {output_file}[/green]"
        )

    def profile_optimized_writer(
        self, schema_name: str, rows: int, batch_size: int = 50000
    ) -> dict[str, float]:
        """Profile OptimizedDataWriter specifically."""
        manager = get_schema_manager()
        schema_dict = manager.get_schema(schema_name)
        schema = CollectionSchema(**schema_dict)

        # Memory checkpoints
        checkpoints = {}

        # Initial memory
        gc.collect()
        checkpoints["initial"] = self.process.memory_info().rss / 1024 / 1024

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create writer
            writer = OptimizedDataWriter(
                schema=schema,
                output_dir=tmpdir,
                batch_size=batch_size,
                format="parquet",
            )
            checkpoints["after_init"] = self.process.memory_info().rss / 1024 / 1024

            # Generate data in batches
            total_written = 0
            batch_memories = []

            while total_written < rows:
                current_batch = min(batch_size, rows - total_written)

                # Memory before batch
                mem_before = self.process.memory_info().rss / 1024 / 1024

                writer._write_batch(total_written, current_batch)

                # Memory after batch
                mem_after = self.process.memory_info().rss / 1024 / 1024
                batch_memories.append(mem_after - mem_before)

                total_written += current_batch

            checkpoints["after_generation"] = (
                self.process.memory_info().rss / 1024 / 1024
            )

            # Finalize
            writer._finalize_current_file()
            checkpoints["after_finalize"] = self.process.memory_info().rss / 1024 / 1024

        # Calculate statistics
        stats = {
            "initial_memory_mb": checkpoints["initial"],
            "initialization_cost_mb": checkpoints["after_init"]
            - checkpoints["initial"],
            "generation_cost_mb": checkpoints["after_generation"]
            - checkpoints["after_init"],
            "finalization_cost_mb": checkpoints["after_finalize"]
            - checkpoints["after_generation"],
            "total_memory_increase_mb": checkpoints["after_finalize"]
            - checkpoints["initial"],
            "avg_batch_memory_mb": np.mean(batch_memories) if batch_memories else 0,
            "max_batch_memory_mb": max(batch_memories) if batch_memories else 0,
        }

        return stats


@click.command()
@click.option("--schema", "-s", default="ecommerce", help="Schema to profile")
@click.option(
    "--rows", "-r", type=int, default=100000, help="Number of rows to generate"
)
@click.option(
    "--batch-sizes",
    "-b",
    multiple=True,
    type=int,
    default=[10000, 50000, 100000],
    help="Batch sizes to test",
)
@click.option("--plot", is_flag=True, help="Generate memory usage plot")
@click.option("--detailed", is_flag=True, help="Run detailed OptimizedWriter profiling")
def main(schema, rows, batch_sizes, plot, detailed):
    """Profile memory usage during data generation."""
    console = Console()

    console.print("[bold blue]Memory Profiler for Milvus Ingest[/bold blue]\n")

    profiler = MemoryProfiler(console)

    try:
        # Profile different batch sizes
        console.print(f"Profiling schema: {schema}")
        console.print(f"Rows: {rows:,}")
        console.print(f"Batch sizes: {', '.join(f'{b:,}' for b in batch_sizes)}\n")

        results = profiler.profile_batch_sizes(
            schema_name=schema, rows=rows, batch_sizes=list(batch_sizes)
        )

        # Analyze results
        console.print("\n[bold]Memory Usage Analysis[/bold]")

        for batch_size, (timestamps, memories) in results.items():
            analysis = profiler.analyze_memory_pattern(timestamps, memories)

            console.print(f"\n[cyan]Batch size: {batch_size:,}[/cyan]")
            console.print(f"  Peak memory: {analysis['peak_memory_mb']:.1f} MB")
            console.print(f"  Average memory: {analysis['avg_memory_mb']:.1f} MB")
            console.print(f"  Memory increase: {analysis['peak_increase_mb']:.1f} MB")
            console.print(
                f"  Growth rate: {analysis['growth_rate_mb_per_sec']:.2f} MB/sec"
            )

            if analysis["potential_leak"]:
                console.print("  [yellow]⚠️  Potential memory leak detected[/yellow]")

        # Generate plot if requested
        if plot and results:
            profiler.plot_memory_usage(results)

        # Detailed profiling
        if detailed:
            console.print("\n[bold]Detailed OptimizedWriter Profiling[/bold]")

            for batch_size in batch_sizes:
                console.print(f"\n[cyan]Batch size: {batch_size:,}[/cyan]")

                stats = profiler.profile_optimized_writer(
                    schema_name=schema, rows=rows, batch_size=batch_size
                )

                console.print(
                    f"  Initialization cost: {stats['initialization_cost_mb']:.1f} MB"
                )
                console.print(
                    f"  Generation cost: {stats['generation_cost_mb']:.1f} MB"
                )
                console.print(
                    f"  Finalization cost: {stats['finalization_cost_mb']:.1f} MB"
                )
                console.print(
                    f"  Average batch memory: {stats['avg_batch_memory_mb']:.1f} MB"
                )
                console.print(
                    f"  Max batch memory: {stats['max_batch_memory_mb']:.1f} MB"
                )
                console.print(
                    f"  Total increase: {stats['total_memory_increase_mb']:.1f} MB"
                )

    except KeyboardInterrupt:
        console.print("\n[yellow]Profiling interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise


if __name__ == "__main__":
    main()
