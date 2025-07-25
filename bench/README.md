# Benchmark Suite for Milvus Ingest

This directory contains benchmarking and profiling tools to measure the performance of milvus-ingest data generation.

## Overview

The benchmark suite includes three main tools:

1. **benchmark_generator.py** - Full pipeline benchmarking across different schemas and configurations
2. **benchmark_optimized.py** - Direct testing of OptimizedDataWriter performance
3. **profile_memory.py** - Memory usage profiling and leak detection

## Installation

First, install the required dependencies:

```bash
# From the project root
pdm install --dev

# Additional dependencies for benchmarking
pip install memory-profiler matplotlib psutil
```

## Usage

### 1. Full Pipeline Benchmark

Test the complete data generation pipeline with various schemas and configurations:

```bash
# Quick benchmark with default settings
python bench/benchmark_generator.py --quick

# Full benchmark with multiple schemas
python bench/benchmark_generator.py \
    --schemas simple ecommerce documents \
    --rows 10000 100000 1000000 \
    --batch-sizes 10000 50000 100000 \
    --output results.csv

# Benchmark specific schemas with custom settings
python bench/benchmark_generator.py \
    -s ecommerce -s users \
    -r 50000 -r 500000 -r 5000000 \
    -b 50000 -b 100000
```

Output includes:
- Rows per second throughput
- File count and total size
- Memory usage
- Time elapsed

### 2. OptimizedWriter Performance

Directly benchmark the OptimizedDataWriter to measure raw generation speed:

```bash
# Test all modes (field types, vector scaling, batch scaling)
python bench/benchmark_optimized.py --mode all

# Test specific field type performance
python bench/benchmark_optimized.py --mode field-types --rows 1000000

# Test vector dimension scaling
python bench/benchmark_optimized.py \
    --mode vector-scaling \
    --vector-type FloatVector \
    --rows 100000

# Test batch size impact
python bench/benchmark_optimized.py \
    --mode batch-scaling \
    --dimensions 768 \
    --rows 1000000
```

### 3. Memory Profiling

Profile memory usage to identify potential leaks and optimization opportunities:

```bash
# Basic memory profiling
python bench/profile_memory.py \
    --schema ecommerce \
    --rows 100000 \
    --batch-sizes 10000 50000 100000

# Generate memory usage plot
python bench/profile_memory.py \
    --schema documents \
    --rows 500000 \
    --plot

# Detailed OptimizedWriter profiling
python bench/profile_memory.py \
    --schema ecommerce \
    --rows 1000000 \
    --detailed
```

## Performance Expectations

Based on the optimized implementation, you should expect:

### Throughput by Field Type
- **Numeric fields**: 50,000-100,000+ rows/sec
- **String fields**: 30,000-70,000 rows/sec
- **Vector fields**: 10,000-50,000 rows/sec (depends on dimensions)
- **JSON fields**: 20,000-40,000 rows/sec

### Vector Performance by Dimensions
- **128d**: 40,000-50,000 rows/sec
- **768d**: 15,000-25,000 rows/sec
- **1536d**: 8,000-15,000 rows/sec

### Memory Usage
- Base overhead: ~100-200 MB
- Per batch (50K rows): ~50-200 MB depending on schema
- Scales linearly with batch size

## Benchmark Results Interpretation

### benchmark_generator.py Output

The CSV output contains:
- `schema`: Schema name tested
- `total_rows`: Number of rows generated
- `batch_size`: Batch size used
- `format`: Output format (parquet/json)
- `time_seconds`: Total time elapsed
- `rows_per_second`: Generation throughput
- `file_count`: Number of files created
- `total_size_mb`: Total output size
- `memory_peak_mb`: Peak memory usage
- `mb_per_second`: Data throughput in MB/s

### Identifying Bottlenecks

1. **Low rows/sec with high memory**: Batch size too large
2. **Consistent memory growth**: Potential memory leak
3. **Sudden performance drops**: File I/O bottleneck
4. **High CPU, low throughput**: Complex field generation

## Tips for Optimization

1. **Batch Size Selection**
   - Start with 50,000 rows
   - Increase for simple schemas
   - Decrease for complex/high-dimensional vectors

2. **Memory Management**
   - Monitor peak memory usage
   - Adjust batch size based on available RAM
   - Use memory profiler to identify leaks

3. **Schema Optimization**
   - Minimize vector dimensions where possible
   - Use appropriate string lengths
   - Consider field nullability impact

## Example Benchmark Report

```
Schema: ecommerce
================
Rows      Batch     Format   Time(s)  Rows/sec   Files  Size(MB)  Memory(MB)
100,000   50,000    parquet  8.5      11,765     1      45.2      125.3
1,000,000 50,000    parquet  87.3     11,458     4      452.1     187.6
5,000,000 100,000   parquet  425.8    11,742     20     2,260.5   245.8

Performance Summary:
- Average throughput: 11,655 rows/sec
- Peak throughput: 11,765 rows/sec
- Optimal batch size: 50,000-100,000 rows
```

## Contributing

When adding new benchmarks:
1. Follow the existing pattern of result collection
2. Include memory profiling where relevant
3. Add documentation for new metrics
4. Test across different schemas and scales