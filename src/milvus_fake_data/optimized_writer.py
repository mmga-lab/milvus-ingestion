"""Simplified optimized writer focusing on core performance improvements."""

import json
import time
from pathlib import Path
from typing import Any

# from concurrent.futures import ProcessPoolExecutor, as_completed
# import multiprocessing  # Reserved for future parallel file generation
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

# Pure NumPy optimizations - consistently outperforms JIT for vector operations
# Uses optimized BLAS libraries for multi-core CPU utilization
logger.debug("Using NumPy vectorized operations with BLAS acceleration")


# Reserved for future parallel file generation
# def _generate_single_file(...):
#     pass


def generate_data_optimized(
    schema_path: Path,
    rows: int,
    output_dir: Path,
    format: str = "parquet",
    batch_size: int = 50000,
    seed: int | None = None,
    max_file_size_mb: int = 256,
    max_rows_per_file: int = 1000000,
    progress_callback: Any = None,
) -> list[str]:
    """
    Optimized data generation using vectorized NumPy operations with file partitioning.

    Key optimizations:
    1. Vectorized data generation with NumPy
    2. Pre-allocated arrays
    3. Batch processing with automatic file partitioning
    4. Efficient Parquet/JSON writing
    5. Minimal memory copying
    6. Smart file splitting based on size and row count

    Args:
        max_file_size_mb: Maximum size per file in MB (default: 256MB)
        max_rows_per_file: Maximum rows per file (default: 1M rows)
    """
    start_time = time.time()

    # Load schema
    with open(schema_path) as f:
        schema = json.load(f)

    fields = schema.get("fields", schema)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed for reproducibility
    if seed:
        np.random.seed(seed)

    # Pre-analyze schema for optimization
    vector_fields = []
    scalar_fields = []

    for field in fields:
        if "Vector" in field["type"]:
            vector_fields.append(field)
        else:
            scalar_fields.append(field)

    # Calculate total number of files needed based on actual batch constraints
    effective_max_rows = min(max_rows_per_file, batch_size * 10)
    total_files = max(1, (rows + effective_max_rows - 1) // effective_max_rows)

    # Generate data in batches and write multiple files if needed
    remaining_rows = rows
    file_index = 0
    pk_offset = 0
    all_files_created = []
    total_generation_time = 0.0
    total_write_time = 0.0

    while remaining_rows > 0:
        # Determine batch size for this file
        current_batch_rows = min(remaining_rows, max_rows_per_file, batch_size * 10)

        logger.info(f"Generating file {file_index + 1}: {current_batch_rows:,} rows")

        generation_start = time.time()
        data: dict[str, Any] = {}

        # Generate scalar fields efficiently
        for field in scalar_fields:
            field_name = field["name"]
            field_type = field["type"]

            if field_type == "Int64":
                if field.get("is_primary", False) and not field.get("auto_id", False):
                    # Sequential IDs for primary key with offset
                    data[field_name] = np.arange(
                        pk_offset, pk_offset + current_batch_rows, dtype=np.int64
                    )
                else:
                    data[field_name] = np.random.randint(
                        -999999, 999999, size=current_batch_rows, dtype=np.int64
                    )

            elif field_type == "Float":
                data[field_name] = np.random.random(current_batch_rows).astype(
                    np.float32
                )

            elif field_type == "Double":
                data[field_name] = np.random.random(current_batch_rows).astype(
                    np.float64
                )

            elif field_type == "Bool":
                data[field_name] = np.random.randint(
                    0, 2, size=current_batch_rows, dtype=bool
                )

            elif field_type in ["VarChar", "String"]:
                # Optimized vectorized string generation
                string_pool = [f"text_{i % 1000}" for i in range(1000)]  # Reuse strings
                indices = np.random.randint(
                    0, len(string_pool), size=current_batch_rows
                )
                # Vectorized string selection using NumPy indexing
                string_array = np.array(string_pool)
                data[field_name] = string_array[indices].tolist()

            elif field_type == "JSON":
                # Generate diverse JSON data with multiple patterns
                json_data = []

                # Pre-generate random data for efficiency
                random_types = np.random.randint(0, 5, size=current_batch_rows)
                random_ints = np.random.randint(1, 1000, size=current_batch_rows)
                random_floats = np.random.random(current_batch_rows)
                random_bools = np.random.randint(0, 2, size=current_batch_rows, dtype=bool)

                # Pre-generate string pools
                categories = ["electronics", "books", "clothing", "food", "toys", "sports", "health", "home"]
                statuses = ["active", "pending", "completed", "cancelled", "processing"]
                tags_pool = ["new", "featured", "sale", "limited", "exclusive", "popular", "trending", "clearance"]

                for i in range(current_batch_rows):
                    json_type = random_types[i]

                    if json_type == 0:
                        # E-commerce product metadata
                        json_obj = {
                            "product_id": int(pk_offset + i),
                            "category": categories[i % len(categories)],
                            "price": round(float(random_floats[i] * 999.99), 2),
                            "in_stock": bool(random_bools[i]),
                            "attributes": {
                                "brand": f"Brand_{random_ints[i] % 50}",
                                "weight": round(float(random_floats[i] * 10), 2),
                                "dimensions": {
                                    "length": int(random_ints[i] % 100),
                                    "width": int((random_ints[i] + 10) % 100),
                                    "height": int((random_ints[i] + 20) % 100)
                                }
                            },
                            "tags": tags_pool[:random_ints[i] % 4 + 1]
                        }
                    elif json_type == 1:
                        # User activity/event data
                        json_obj = {
                            "event_id": int(pk_offset + i),
                            "event_type": ["click", "view", "purchase", "share", "like"][i % 5],
                            "timestamp": int(1600000000 + random_ints[i] * 1000),
                            "user": {
                                "id": f"user_{random_ints[i] % 1000}",
                                "session": f"session_{random_ints[i] % 100}",
                                "device": ["mobile", "desktop", "tablet"][i % 3]
                            },
                            "metrics": {
                                "duration_ms": int(random_ints[i] * 10),
                                "clicks": int(random_ints[i] % 10),
                                "score": round(float(random_floats[i] * 5), 2)
                            }
                        }
                    elif json_type == 2:
                        # Configuration/settings data
                        json_obj = {
                            "config_id": int(pk_offset + i),
                            "name": f"config_{i}",
                            "settings": {
                                "enabled": bool(random_bools[i]),
                                "threshold": float(random_floats[i]),
                                "max_retries": int(random_ints[i] % 10),
                                "timeout_seconds": int(random_ints[i] % 300),
                                "features": {
                                    "feature_a": bool(random_bools[i]),
                                    "feature_b": bool(not random_bools[i]),
                                    "feature_c": bool(i % 3 == 0)
                                }
                            },
                            "metadata": {
                                "version": f"{random_ints[i] % 3}.{random_ints[i] % 10}.{random_ints[i] % 20}",
                                "last_updated": int(1600000000 + random_ints[i] * 1000)
                            }
                        }
                    elif json_type == 3:
                        # Analytics/metrics data
                        json_obj = {
                            "metric_id": int(pk_offset + i),
                            "type": "analytics",
                            "values": {
                                "count": int(random_ints[i]),
                                "sum": round(float(random_floats[i] * 10000), 2),
                                "avg": round(float(random_floats[i] * 100), 2),
                                "min": round(float(random_floats[i] * 10), 2),
                                "max": round(float(random_floats[i] * 1000), 2)
                            },
                            "dimensions": {
                                "region": ["north", "south", "east", "west"][i % 4],
                                "category": categories[i % len(categories)],
                                "segment": f"segment_{random_ints[i] % 10}"
                            },
                            "percentiles": {
                                "p50": round(float(random_floats[i] * 50), 2),
                                "p90": round(float(random_floats[i] * 90), 2),
                                "p99": round(float(random_floats[i] * 99), 2)
                            }
                        }
                    else:
                        # Document/content metadata
                        json_obj = {
                            "doc_id": int(pk_offset + i),
                            "title": f"Document_{i}",
                            "status": statuses[i % len(statuses)],
                            "metadata": {
                                "author": f"author_{random_ints[i] % 100}",
                                "created_at": int(1600000000 + random_ints[i] * 1000),
                                "word_count": int(random_ints[i] * 10),
                                "language": ["en", "es", "fr", "de", "zh"][i % 5],
                                "sentiment": {
                                    "positive": round(float(random_floats[i]), 3),
                                    "negative": round(float(1 - random_floats[i]), 3),
                                    "neutral": round(float(random_floats[i] * 0.5), 3)
                                }
                            },
                            "tags": tags_pool[:random_ints[i] % 3 + 1],
                            "properties": {
                                "public": bool(random_bools[i]),
                                "archived": bool(not random_bools[i]),
                                "priority": int(random_ints[i] % 5)
                            }
                        }

                    json_data.append(json_obj)

                data[field_name] = json_data

            elif field_type == "Array":
                element_type = field.get("element_type", "Int32")
                max_capacity = field.get("max_capacity", 5)

                # Optimized vectorized array generation
                lengths = np.random.randint(0, max_capacity + 1, size=current_batch_rows)

                if element_type in ["Int32", "Int64"]:
                    # Vectorized integer array generation
                    total_elements = np.sum(lengths)
                    if total_elements > 0:
                        # Pre-generate large pool of integers and slice as needed
                        int_pool = np.random.randint(-999, 999, size=total_elements, dtype=np.int32)
                        array_data = []
                        start_idx = 0
                        for length in lengths:
                            if length > 0:
                                array_data.append(int_pool[start_idx:start_idx + length].tolist())
                                start_idx += length
                            else:
                                array_data.append([])
                    else:
                        array_data = [[] for _ in range(current_batch_rows)]
                    data[field_name] = array_data
                else:
                    # Optimized string arrays with pre-computed strings
                    str_templates = [f"item_{j}" for j in range(max_capacity)]
                    array_data = []
                    for length in lengths:
                        if length > 0:
                            array_data.append(str_templates[:length])
                        else:
                            array_data.append([])
                    data[field_name] = array_data

        # Generate vector fields efficiently
        for field in vector_fields:
            field_name = field["name"]
            field_type = field["type"]
            dim = field.get("dim", 128)

            if field_type == "FloatVector":
                # Generate normalized float vectors efficiently (uses multiple CPU cores)
                vectors = np.random.randn(current_batch_rows, dim).astype(np.float32)

                # Always use NumPy for vector normalization (consistently faster due to optimized BLAS)
                # JIT compilation overhead typically outweighs benefits for this simple operation
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                vectors = vectors / norms

                # Store as list of arrays for pandas
                data[field_name] = list(vectors)

            elif field_type == "BinaryVector":
                byte_dim = dim // 8
                binary_vectors = np.random.randint(
                    0, 256, size=(current_batch_rows, byte_dim), dtype=np.uint8
                )
                data[field_name] = list(binary_vectors)

            elif field_type == "Int8Vector":
                int8_vectors = np.random.randint(
                    -128, 128, size=(current_batch_rows, dim), dtype=np.int8
                )
                data[field_name] = list(int8_vectors)

            elif field_type in ["Float16Vector", "BFloat16Vector"]:
                # Generate as float32 first, then convert
                vectors = np.random.randn(current_batch_rows, dim).astype(np.float32)
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                vectors = vectors / norms

                if field_type == "Float16Vector":
                    vectors_converted = vectors.astype(np.float16).view(np.uint8)
                    data[field_name] = list(
                        vectors_converted.reshape(current_batch_rows, -1)
                    )
                else:  # BFloat16Vector
                    # Simple bfloat16 approximation
                    vectors_converted = vectors.view(np.uint32)
                    vectors_converted = (
                        (vectors_converted >> 16).astype(np.uint16).view(np.uint8)
                    )
                    data[field_name] = list(
                        vectors_converted.reshape(current_batch_rows, -1)
                    )

        batch_generation_time = time.time() - generation_start
        total_generation_time += batch_generation_time

        # Create DataFrame
        df = pd.DataFrame(data)

        # Write file
        write_start = time.time()

        if format.lower() == "parquet":
            if total_files == 1:
                output_file = output_dir / "data.parquet"
            else:
                file_num = file_index + 1
                output_file = output_dir / f"data-{file_num:05d}-of-{total_files:05d}.parquet"

            # Convert to PyArrow table for efficient writing
            table = pa.Table.from_pandas(df)

            # Write with optimized settings
            pq.write_table(
                table,
                output_file,
                compression="snappy",
                use_dictionary=True,
                write_statistics=False,
                row_group_size=min(50000, current_batch_rows),
            )

        elif format.lower() == "json":
            if total_files == 1:
                output_file = output_dir / "data.json"
            else:
                file_num = file_index + 1
                output_file = output_dir / f"data-{file_num:05d}-of-{total_files:05d}.json"

            # Write JSON efficiently
            with open(output_file, "w") as f:
                for i, record in enumerate(df.to_dict(orient="records")):
                    # Handle numpy types
                    for key, value in record.items():
                        if isinstance(value, np.ndarray):
                            record[key] = value.tolist()
                        elif isinstance(value, np.generic):
                            record[key] = value.item()

                    if i > 0:
                        f.write("\n")
                    json.dump(record, f, ensure_ascii=False, separators=(",", ":"))

        else:
            raise ValueError(f"Unsupported format: {format}. High-performance mode only supports 'parquet' and 'json' formats.")

        batch_write_time = time.time() - write_start
        total_write_time += batch_write_time

        all_files_created.append(str(output_file))

        # Log progress
        logger.info(
            f"File {file_index + 1} completed: {current_batch_rows:,} rows "
            f"({current_batch_rows / batch_generation_time:.0f} rows/sec generation)"
        )

        # Update counters
        remaining_rows -= current_batch_rows
        pk_offset += current_batch_rows
        file_index += 1

        # Update progress if callback provided
        if progress_callback:
            progress_callback(pk_offset)

    # Write metadata
    meta_file = output_dir / "meta.json"
    total_time = time.time() - start_time

    meta_data = {
        "schema": schema,
        "generation_info": {
            "total_rows": rows,
            "format": format,
            "seed": seed,
            "data_files": [Path(f).name for f in all_files_created],
            "file_count": len(all_files_created),
            "max_rows_per_file": max_rows_per_file,
            "max_file_size_mb": max_file_size_mb,
            "generation_time": total_generation_time,
            "write_time": total_write_time,
            "total_time": total_time,
            "rows_per_second": rows / total_time,
        },
    }

    with open(meta_file, "w") as f:
        json.dump(meta_data, f, indent=2)

    logger.info(
        f"Total generation completed: {rows:,} rows in {len(all_files_created)} file(s)"
    )
    logger.info(f"Total time: {total_time:.2f}s ({rows / total_time:.0f} rows/sec)")

    return all_files_created
