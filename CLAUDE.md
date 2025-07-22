# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
pdm install        # Install with development dependencies (default behavior)
pdm install --prod # Install production dependencies only
```

### Code Quality & Testing
```bash
# Formatting
pdm run ruff format src tests

# Linting & Type Checking
pdm run ruff check src tests
pdm run mypy src

# Testing
pdm run pytest                                          # Run all tests
pdm run pytest tests/test_generator.py                  # Run specific test file
pdm run pytest tests/test_generator.py::TestClass      # Run specific test class
pdm run pytest tests/test_generator.py::test_function  # Run specific test function
pdm run pytest --cov=src --cov-report=html             # Run tests with coverage
pdm run pytest --cov=src --cov-report=term-missing     # Show missing lines in terminal
pdm run pytest -v -s                                    # Verbose output with print statements
pdm run pytest -m "not slow"                            # Skip slow tests
pdm run pytest -m integration                           # Run only integration tests

# Combined quality checks (via Makefile)
make lint          # Run ruff format + check + mypy
make test          # Run pytest
make test-cov      # Run tests with coverage report
make check         # Run lint + test together
make clean         # Clean build artifacts and caches
make build         # Build the package
make publish       # Publish to PyPI
make security      # Run security checks (safety, pip-audit, bandit)
```

### Test Environment Configuration

**Local Test Environment (Recommended):**
Use the Docker Compose setup in `deploy/docker-compose.yml` to start local test services:

```bash
# Start local test environment
cd deploy/
docker-compose up -d

# Environment variables for local testing
export MILVUS_URI=http://127.0.0.1:19530
export MINIO_HOST=127.0.0.1
export MINIO_ACCESS_KEY=minioadmin
export MINIO_SECRET_KEY=minioadmin
export MINIO_BUCKET=a-bucket

# Or use the .env file in project root
cp .env.example .env  # Edit with your values
```

**Note:** The `deploy/docker-compose.yml` provides a complete local testing stack including Milvus and MinIO.

### Building & Publishing
```bash
pdm build         # Build the package
pdm publish       # Publish to PyPI (requires PDM_PUBLISH_TOKEN)
```

### CLI Usage for Testing
```bash
# Install in development mode first
pdm install

# High-performance data generation commands (optimized for large-scale datasets)
milvus-fake-data generate --builtin simple --rows 100000 --preview        # Generate 100K rows
milvus-fake-data generate --schema example_schema.json --rows 1000000     # Generate 1M rows
milvus-fake-data generate --schema schema.json --rows 5000000 --batch-size 100000 # Use large batch size

# Additional generation options
milvus-fake-data generate --schema schema.json --validate-only            # Validate schema without generating
milvus-fake-data generate --schema schema.json --rows 1000000 --no-progress # Disable progress bar
milvus-fake-data generate --schema schema.json --max-file-size 512       # Set max file size (MB)
milvus-fake-data generate --schema schema.json --max-rows-per-file 500000 # Set max rows per file
milvus-fake-data generate --schema schema.json --out mydata --force   # Force overwrite existing output

# Schema management commands
milvus-fake-data schema list                    # List all schemas
milvus-fake-data schema show simple            # Show schema details
milvus-fake-data schema add myschema file.json # Add custom schema
milvus-fake-data schema remove myschema        # Remove custom schema
milvus-fake-data schema help                   # Schema format help

# Utility commands
milvus-fake-data clean                         # Clean up generated files

# Upload to S3/MinIO (standalone upload, useful for separate upload/import workflow)
milvus-fake-data upload --local-path ./output --s3-path s3://bucket/prefix/              # Upload to AWS S3
milvus-fake-data upload --local-path ./output --s3-path s3://bucket/prefix/ --endpoint-url http://localhost:9000  # Upload to MinIO
milvus-fake-data upload --local-path ./output --s3-path s3://bucket/prefix/ --no-verify-ssl  # Disable SSL verification
milvus-fake-data upload --local-path ./output --s3-path s3://bucket/prefix/ --access-key-id KEY --secret-access-key SECRET  # With credentials

# Send data to Milvus
# Direct insert to Milvus (reads local parquet and JSON files and creates collection)
milvus-fake-data to-milvus insert ./output                                # Insert to local Milvus
milvus-fake-data to-milvus insert ./output --uri http://192.168.1.100:19530 --token your-token  # Remote Milvus with auth
milvus-fake-data to-milvus insert ./output --drop-if-exists               # Drop existing collection and recreate
milvus-fake-data to-milvus insert ./output --collection-name my_collection --batch-size 5000  # Custom settings

# Bulk import to Milvus (upload + import in one step)
# Note: Combines upload and import for convenience, includes auto-collection creation
milvus-fake-data to-milvus import --local-path ./output/ --s3-path data/ --bucket my-bucket --endpoint-url http://minio:9000  # Upload and import
milvus-fake-data to-milvus import --local-path ./output/ --s3-path data/ --bucket my-bucket --endpoint-url http://minio:9000 --collection-name my_collection  # Override collection name
milvus-fake-data to-milvus import --local-path ./output/ --s3-path data/ --bucket my-bucket --endpoint-url http://minio:9000 --wait  # Wait for completion
milvus-fake-data to-milvus import --local-path ./output/ --s3-path data/ --bucket my-bucket --endpoint-url http://minio:9000 --access-key-id key --secret-access-key secret  # With credentials
milvus-fake-data to-milvus import --local-path ./output/ --s3-path data/ --bucket my-bucket --endpoint-url http://minio:9000 --drop-if-exists  # Drop and recreate
```

## Architecture Overview

### Project Structure
```
milvus-fake-data/
├── src/milvus_fake_data/       # Main package
│   ├── cli.py                  # CLI entry point
│   ├── optimized_writer.py     # High-performance data generation
│   ├── models.py               # Pydantic schema models
│   ├── schema_manager.py       # Schema management
│   ├── milvus_inserter.py      # Direct Milvus insertion
│   ├── milvus_importer.py      # Bulk import from S3/MinIO
│   ├── uploader.py             # S3/MinIO uploads
│   └── schemas/                # Built-in schema JSON files
├── tests/                      # Test suite
│   ├── conftest.py            # Common test fixtures
│   └── test_*.py              # Test modules
├── pyproject.toml             # PDM configuration and metadata
├── Makefile                   # Common development tasks
└── README.md                  # User documentation
```

### Core Modules
- **cli.py**: Command-line interface with Click framework, supports generation, schema management, upload, and Milvus integration
- **optimized_writer.py**: High-performance vectorized data generation using NumPy operations for large datasets
- **generator.py**: Core data generation logic that wraps optimized_writer for compatibility
- **models.py**: Pydantic models for schema validation with comprehensive error messages and field type definitions, includes dynamic field support
- **schema_manager.py**: Manages built-in and custom schemas, stores user schemas in ~/.milvus-fake-data/schemas
- **builtin_schemas.py**: Built-in schema definitions with metadata (13 schemas: simple, ecommerce, documents, images, users, videos, news, audio_transcripts, ai_conversations, face_recognition, ecommerce_partitioned, cardinality_demo, dynamic_example)
- **rich_display.py**: Rich terminal formatting and user interface components
- **logging_config.py**: Structured logging with loguru
- **exceptions.py**: Custom exception classes (MilvusFakeDataError, SchemaError, UnsupportedFieldTypeError, GenerationError)
- **uploader.py**: S3/MinIO upload functionality with boto3, supports AWS S3 and S3-compatible storage
- **milvus_inserter.py**: Direct insert to Milvus using PyMilvus client with automatic collection creation, supports both Parquet and JSON files
- **milvus_importer.py**: Bulk import using PyMilvus bulk_import API for pre-uploaded files, supports both Parquet and JSON files

### High-Performance Data Flow
1. Schema validation using Pydantic models (models.py)
2. **Vectorized data generation** using NumPy operations (optimized_writer.py)
3. **Direct PyArrow integration** for efficient Parquet writing
4. **Large batch processing** (50K+ rows per batch)
5. **Automatic file partitioning** (256MB chunks, 1M rows/file by default)
6. **Multi-format support**: Parquet (binary, high-performance) and JSON (standard array format)
7. **Dynamic field support**: Generate additional fields stored in `$meta` for flexible schemas

### Output Structure
The tool always generates a directory (not a single file) containing:
- One or more data files in Parquet or JSON format (automatically partitioned based on size/row limits)
- `meta.json` file with collection metadata and generation details

### Supported Data Formats
- **Parquet** (default): High-performance binary format optimized for analytics workloads
- **JSON**: Standard JSON array format `[{}, {}, {}...]` compatible with Milvus bulk import

### JSON Format Details
The tool generates JSON files in standard array format for maximum compatibility:

```json
[
  {
    "id": 1,
    "name": "Product 1",
    "price": 19.99,
    "embedding": [0.1, 0.2, 0.3, ...],
    "$meta": {
      "dynamic_field1": "value1",
      "dynamic_field2": 42
    }
  },
  {
    "id": 2,
    "name": "Product 2", 
    "price": 29.99,
    "embedding": [0.4, 0.5, 0.6, ...],
    "$meta": {
      "dynamic_field1": "value2",
      "dynamic_field3": true
    }
  }
]
```

**JSON Format Features:**
- **Standard Array**: Direct JSON array `[{}, {}...]` without wrapper objects
- **Milvus Compatible**: Works with both direct insert and bulk import
- **Dynamic Fields**: Supports `$meta` field for additional dynamic properties
- **Multi-Format Read**: Insert/import commands auto-detect JSON vs Parquet files
- **Flexible Structure**: Supports JSONL, single objects, and legacy formats for compatibility

### Dynamic Field Support
The tool supports Milvus dynamic fields through the `$meta` field:

```json
{
  "collection_name": "dynamic_example",
  "enable_dynamic_field": true,
  "dynamic_fields": [
    {
      "name": "author",
      "type": "String", 
      "probability": 0.8,
      "values": ["Alice", "Bob", "Charlie"]
    },
    {
      "name": "views",
      "type": "Int",
      "probability": 0.9,
      "min": 1,
      "max": 10000
    }
  ],
  "fields": [...]
}
```

**Dynamic Field Types**: String, Int, Float, Bool, Array, JSON
**Storage**: All dynamic fields are stored in the `$meta` field during generation
**Import**: The `$meta` field is automatically unpacked during Milvus insertion

### Schema System
The tool supports two types of schemas:
- **Built-in schemas**: Located in src/milvus_fake_data/schemas/ as JSON files
- **Custom schemas**: User-managed schemas stored in ~/.milvus-fake-data/schemas/ with metadata

### Supported Milvus Field Types
- **Numeric**: Int8, Int16, Int32, Int64, Float, Double, Bool
- **Text**: VarChar, String (require max_length parameter)
- **Complex**: JSON, Array (Array requires element_type and max_capacity)
- **Vectors**: FloatVector, BinaryVector, Float16Vector, BFloat16Vector, SparseFloatVector (all require dim parameter)
- **Note**: Int8Vector is not supported in current Milvus versions

## High-Performance Implementation Details

### Performance Optimizations
**Vectorized Operations:**
- NumPy batch generation (10-20x faster than loops)
- Vectorized L2 normalization for FloatVectors  
- Pre-allocated arrays to avoid memory reallocation
- String pools for text field generation

**Efficient File I/O:**
- Direct PyArrow integration for Parquet writing
- Snappy compression for optimal speed/size ratio
- Large row groups (50K+ rows) for better I/O efficiency
- Streaming writes to manage memory usage

**Memory Optimization:**
- Large batch sizes (50K+ rows) for better CPU utilization
- Minimal data copying between operations
- Object dtype for multidimensional arrays in pandas

### Schema Validation
Uses Pydantic for comprehensive validation with helpful error messages. The validation ensures:
- Exactly one primary key field
- Required parameters for each field type (e.g., dim for vectors, max_length for strings)
- Proper field name format (alphanumeric + underscores)
- auto_id only for Int64 primary keys

### Performance Benchmarks
**Typical Performance (on modern hardware):**

**Data Generation:**
- **Small datasets** (<10K rows): 5,000-10,000 rows/sec
- **Medium datasets** (10K-100K rows): 7,000-15,000 rows/sec  
- **Large datasets** (100K+ rows): 10,000-25,000 rows/sec
- **Vector-heavy schemas**: 5,000-15,000 rows/sec
- **Text-heavy schemas**: 10,000-30,000 rows/sec

**Format Performance:**
- **Parquet**: Fastest generation and smallest file size, optimized for analytics
- **JSON**: Slower generation but human-readable, good for debugging and compatibility
- **File Size**: Parquet ~30-50% smaller than JSON for the same data
- **Insert Performance**: Both formats have similar Milvus insertion speeds

### Testing Approach
- pytest with coverage reporting
- Performance regression tests for large datasets
- Tests cover validation, generation, CLI interface, and schema management
- Integration tests for end-to-end workflows with 100K+ row datasets
- Test fixtures in tests/conftest.py for common scenarios

## JSON Format Usage Examples

### Basic JSON Generation
```bash
# Generate simple JSON data
milvus-fake-data generate --builtin simple --rows 1000 --format json --out json_data

# Preview the JSON structure
head -n 1 json_data/data.json | python -m json.tool
```

### Dynamic Fields with JSON
```bash  
# Generate data with dynamic fields
milvus-fake-data generate --builtin dynamic_example --rows 500 --format json --out dynamic_data

# The output will include $meta field with dynamic content:
# {
#   "title": "Product Title",
#   "embedding": [...],
#   "$meta": {
#     "author": "John Doe", 
#     "views": 1234,
#     "rating": 4.5
#   }
# }
```

### Multi-Format Workflow
```bash
# Generate both formats for comparison
milvus-fake-data generate --builtin ecommerce --rows 10000 --format parquet --out parquet_data
milvus-fake-data generate --builtin ecommerce --rows 10000 --format json --out json_data

# Both can be used interchangeably
milvus-fake-data to-milvus insert ./parquet_data --collection-name parquet_collection
milvus-fake-data to-milvus insert ./json_data --collection-name json_collection
```

### Advanced JSON Import
```bash
# Generate large JSON dataset with dynamic fields  
milvus-fake-data generate --builtin dynamic_example --rows 100000 --format json --out large_json

# Upload and import to Milvus with custom settings
milvus-fake-data to-milvus import \
  --local-path ./large_json \
  --s3-path large-dataset/ \
  --bucket production-bucket \
  --endpoint-url http://minio:9000 \
  --collection-name production_collection \
  --wait \
  --timeout 600
```

## Common Development Workflows

### Adding a New Built-in Schema
1. Create a JSON schema file in `src/milvus_fake_data/schemas/`
2. Follow the existing schema format with proper field definitions
3. Test the schema: `milvus-fake-data generate --builtin your_schema --rows 1000 --preview`
4. Add documentation in README.md

### Adding a New Field Type
1. Update `models.py` to add the field type in `FieldType` enum
2. Add validation logic in the corresponding field model (e.g., `VectorField`, `ScalarField`)
3. Implement generation logic in `optimized_writer.py` in the `_generate_field_data` method
4. Add tests in `tests/test_models.py` and `tests/test_optimized_writer.py`

### Performance Testing
```bash
# Test with large datasets
pdm run python -m cProfile -o profile.stats src/milvus_fake_data/cli.py generate --builtin simple --rows 1000000

# Analyze profile
pdm run python -m pstats profile.stats

# Memory profiling
pdm run python -m memory_profiler src/milvus_fake_data/cli.py generate --builtin simple --rows 1000000
```

### Debugging Tips
1. Use `--preview` flag to inspect first 5 rows without full generation
2. Use `--validate-only` to check schema without generating data
3. Enable debug logging: `export LOGURU_LEVEL=DEBUG`
4. For performance issues, check batch size with `--batch-size` parameter

## Important Notes

### High-Performance Focus
This tool is **optimized for large-scale data generation** (100K+ rows). Key design principles:
- **Default batch size**: 50,000 rows (vs industry standard 1,000-10,000)
- **Vectorized operations**: All data generation uses NumPy vectorization
- **Memory efficiency**: Streaming writes prevent memory exhaustion on large datasets
- **CPU optimization**: Designed to utilize modern multi-core processors effectively

### Error Handling
The tool provides detailed error messages for schema validation failures, pointing users to exact issues and providing example fixes.

### Recommended Usage Patterns
**For maximum performance:**
```bash
# Use large batch sizes for better CPU utilization
milvus-fake-data generate --schema schema.json --rows 1000000 --batch-size 100000

# For very large datasets, monitor memory usage
milvus-fake-data generate --schema schema.json --rows 10000000 --batch-size 50000
```

### Environment Variables
- MILVUS_FAKE_DATA_SCHEMA_DIR: Override default schema directory for testing
- AWS_ACCESS_KEY_ID: AWS/MinIO access key ID for S3 uploads
- AWS_SECRET_ACCESS_KEY: AWS/MinIO secret access key for S3 uploads

### Hardware Recommendations
**For optimal performance:**
- **CPU**: Modern multi-core processor (4+ cores recommended)
- **Memory**: 8GB+ RAM for datasets >1M rows
- **Storage**: SSD recommended for large file I/O operations

### Python API Usage
The package can be used programmatically:
```python
from milvus_fake_data.generator import generate_mock_data
from milvus_fake_data.schema_manager import get_schema_manager
from milvus_fake_data.builtin_schemas import load_builtin_schema

# Use schema manager for built-in schemas
manager = get_schema_manager()
schema = manager.get_schema("simple")

# Generate data from schema
df = generate_mock_data("schema.json", rows=1000, seed=42)

# Add custom schema programmatically  
custom_schema = {
    "collection_name": "my_collection",
    "fields": [
        {"name": "id", "type": "Int64", "is_primary": True},
        {"name": "vector", "type": "FloatVector", "dim": 128}
    ]
}
manager.add_schema("my_custom", custom_schema, "Description", ["tag"])
```

### S3/MinIO Upload Feature
The tool includes built-in support for uploading generated data to S3-compatible storage:

**Features:**
- Support for AWS S3 and MinIO (or any S3-compatible storage)
- Automatic bucket creation if it doesn't exist
- Progress tracking for large uploads
- Batch upload with error handling
- Credential support via CLI options or environment variables

**Usage Examples:**
```bash
# Upload to AWS S3 (using default credentials)
milvus-fake-data upload ./output s3://my-bucket/data/

# Upload to MinIO with custom endpoint
milvus-fake-data upload ./output s3://my-bucket/data/ \
    --endpoint-url http://localhost:9000 \
    --access-key-id minioadmin \
    --secret-access-key minioadmin

# Upload with environment variables
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
milvus-fake-data upload ./output s3://my-bucket/data/
```

### Direct Import to Milvus
The tool can directly import generated data to Milvus:

**Milvus Import Features:**
- Automatic collection creation from metadata
- Smart index creation based on vector dimensions
- Batch processing for high-performance import
- Support for authentication and custom databases
- Connection testing before import

**Usage Examples:**
```bash
# Generate data and insert to local Milvus
milvus-fake-data generate --builtin ecommerce --rows 100000 --out products/
milvus-fake-data to-milvus insert ./products/

# Insert to remote Milvus with authentication
milvus-fake-data to-milvus insert ./products/ \
    --uri http://192.168.1.100:19530 \
    --token your-token \
    --collection-name product_vectors
```

### Bulk Import Feature
The tool includes support for bulk importing pre-uploaded files from MinIO/S3 to Milvus using the PyMilvus bulk_import API:

**Features:**
- **Combined upload and import** in one step for convenience
- Import data directories (output from 'generate' command) to Milvus collection
- Automatic upload to S3/MinIO before import
- Wait for import completion with timeout support
- Check import job progress
- List all import jobs with filtering
- **Automatic collection creation from meta.json (same as insert command)**

**Usage Examples:**
```bash
# Upload and import in one step (collection name from meta.json)
milvus-fake-data to-milvus import --local-path ./output/ --s3-path data/ --bucket my-bucket --endpoint-url http://minio:9000

# Upload and import with custom collection name and wait for completion
milvus-fake-data to-milvus import --collection-name my_collection --local-path ./output/ --s3-path data/ --bucket my-bucket --endpoint-url http://minio:9000 --wait --timeout 300
```

**Data Import Workflows:**

### Option 1: One-Step Import (Recommended for most users)
```bash
# Generate data (Parquet format - default)
milvus-fake-data generate --builtin simple --rows 1000000 --out ./output

# Generate data (JSON format)
milvus-fake-data generate --builtin simple --rows 1000000 --format json --out ./output

# Upload and import in one step (supports both Parquet and JSON files)
milvus-fake-data to-milvus import \
  --local-path ./output/ \
  --s3-path data/ \
  --bucket my-bucket \
  --endpoint-url http://minio:9000 \
  --wait
```

### Option 2: Separate Upload and Import
```bash
# Generate data with dynamic fields (JSON format recommended)
milvus-fake-data generate --builtin dynamic_example --rows 100000 --format json --out ./output

# Step 1: Upload to S3/MinIO (supports mixed file types)
milvus-fake-data upload ./output s3://my-bucket/data/ --endpoint-url http://minio:9000

# Step 2: Import (for pre-uploaded files - would need separate import command)
# Note: This workflow is useful when you need to upload once and import multiple times
```

### Option 3: Direct Insert (No S3/MinIO)
```bash
# Generate data (auto-detects Parquet and JSON files)
milvus-fake-data generate --builtin simple --rows 100000 --format json --out ./output

# Direct insert to Milvus (works with both file formats)
milvus-fake-data to-milvus insert ./output --uri http://milvus:19530
```

**Method Comparison:**

1. **One-Step Import (`to-milvus import`)**: 
   - **Automatically uploads files to MinIO/S3 then imports**
   - Uses Milvus bulk import API for high-performance imports
   - Suitable for very large datasets (millions of rows)
   - Asynchronous operation with job tracking
   - **Auto-collection creation from meta.json (same as insert)**
   - **Most convenient for typical workflows**

2. **Direct Insert (`to-milvus insert`)**:
   - Reads data directly from local Parquet and JSON files
   - Creates collection and inserts data in one operation
   - Suitable for smaller to medium datasets
   - Synchronous operation with progress bar
   - **No S3/MinIO required**
   - **Multi-format support**: Auto-detects and processes both file types

3. **Separate Upload (`upload`)**:
   - Standalone upload to S3/MinIO
   - **Useful for batch operations or when upload/import happen at different times**
   - Can be combined with external import tools or processes

**Important Notes:**
- Files must be already uploaded to MinIO/S3 before using bulk import
- Each import file should not exceed 16 GB
- Maximum 1024 files per import request
- Maximum 1024 concurrent import requests

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce batch size: `--batch-size 10000`
   - Enable file partitioning: `--max-rows-per-file 500000`
   - Monitor memory usage during generation

2. **Slow Generation Performance**
   - Increase batch size for better CPU utilization: `--batch-size 100000`
   - Use Parquet format (default) for best I/O performance
   - Ensure you're using a modern multi-core CPU

3. **Schema Validation Errors**
   - Check field type requirements (e.g., `dim` for vectors, `max_length` for strings)
   - Ensure exactly one primary key field
   - Validate schema first: `--validate-only`

4. **Milvus Connection Issues**
   - Verify URI format: `http://host:port` (not `https://`)
   - Check authentication token if required
   - Test connection with simple schema first

5. **S3/MinIO Upload Failures**
   - Verify endpoint URL for MinIO: `--endpoint-url http://minio:9000`
   - Check credentials: `--access-key-id KEY --secret-access-key SECRET`
   - For SSL issues: `--no-verify-ssl`