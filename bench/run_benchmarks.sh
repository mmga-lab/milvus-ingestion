#!/bin/bash
# Quick benchmark runner script for milvus-ingest performance testing

set -e

echo "🚀 Milvus Ingest Benchmark Suite"
echo "================================"

# Check if required dependencies are installed
echo "Checking dependencies..."
python -c "import memory_profiler, matplotlib, psutil" 2>/dev/null || {
    echo "❌ Missing dependencies. Installing..."
    pip install memory-profiler matplotlib psutil
}

# Create results directory
mkdir -p bench_results
cd bench_results

echo ""
echo "📊 Running Quick Performance Benchmark"
echo "--------------------------------------"
python ../bench/benchmark_generator.py --quick --output quick_benchmark.csv

echo ""
echo "🔬 Running Field Type Performance Tests"
echo "---------------------------------------"
python ../bench/benchmark_optimized.py --mode field-types --rows 50000

echo ""
echo "💾 Running Memory Profile"
echo "--------------------------"
python ../bench/profile_memory.py --schema simple --rows 50000 --batch-sizes 10000 25000 50000 --plot

echo ""
echo "✅ Benchmarks completed! Results saved in bench_results/"
echo ""
echo "Files created:"
ls -la *.csv *.png 2>/dev/null || echo "No result files found"

echo ""
echo "🎯 For more detailed benchmarking, try:"
echo "  python bench/benchmark_generator.py --schemas ecommerce documents --rows 100000 500000"
echo "  python bench/benchmark_optimized.py --mode all --rows 1000000"
echo "  python bench/profile_memory.py --schema ecommerce --rows 1000000 --detailed"