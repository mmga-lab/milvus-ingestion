# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
pdm install        # Install production dependencies
pdm install-dev    # Install with development dependencies (use this for development)
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
pdm run pytest --cov=src --cov-report=html             # Run tests with coverage

# Combined quality checks
make lint test     # Run linting and tests together
```

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
milvus-fake-data generate --schema schema.json --rows 5000000 --workers 8 # Use 8 parallel workers

# Additional generation options
milvus-fake-data generate --schema schema.json --validate-only            # Validate schema without generating
milvus-fake-data generate --schema schema.json --rows 1000000 --no-progress # Disable progress bar
milvus-fake-data generate --schema schema.json --max-file-size 512       # Set max file size (MB)
milvus-fake-data generate --schema schema.json --max-rows-per-file 500000 # Set max rows per file
milvus-fake-data generate --schema schema.json --output mydata --force   # Force overwrite existing output

# Schema management commands
milvus-fake-data schema list                    # List all schemas
milvus-fake-data schema show simple            # Show schema details
milvus-fake-data schema add myschema file.json # Add custom schema
milvus-fake-data schema remove myschema        # Remove custom schema
milvus-fake-data schema help                   # Schema format help

# Utility commands
milvus-fake-data clean                         # Clean up generated files

# Upload to S3/MinIO
milvus-fake-data upload ./output s3://bucket/prefix/              # Upload to AWS S3
milvus-fake-data upload ./output s3://bucket/prefix/ --endpoint-url http://localhost:9000  # Upload to MinIO
milvus-fake-data upload ./output s3://bucket/prefix/ --no-verify-ssl  # Disable SSL verification
milvus-fake-data upload ./output s3://bucket/prefix/ --access-key-id KEY --secret-access-key SECRET  # With credentials

# Import to Milvus
milvus-fake-data to-milvus ./output                              # Import to local Milvus
milvus-fake-data to-milvus ./output --host 192.168.1.100 --user root --password Milvus  # Remote Milvus with auth
milvus-fake-data to-milvus ./output --drop-if-exists             # Drop existing collection and recreate
milvus-fake-data to-milvus ./output --collection-name my_collection --batch-size 5000  # Custom settings

# Import to Zilliz Cloud
milvus-fake-data to-zilliz ./output --uri https://in03-xxx.api.gcp-us-west1.zillizcloud.com --token YOUR_TOKEN
milvus-fake-data to-zilliz ./output --uri https://in03-xxx.api.gcp-us-west1.zillizcloud.com --token YOUR_TOKEN --drop-if-exists
```

## Architecture Overview

### Core Modules
- **cli.py**: High-performance command-line interface optimized for large-scale data generation
- **optimized_writer.py**: Vectorized data generation engine using NumPy operations for maximum performance
- **generator.py**: Legacy data generation logic (for compatibility)
- **models.py**: Pydantic models for schema validation with comprehensive error messages
- **schema_manager.py**: Manages custom and built-in schemas, stores user schemas in ~/.milvus-fake-data/schemas
- **builtin_schemas.py**: Pre-built schemas for common use cases (e-commerce, documents, images, etc.)
- **rich_display.py**: Rich terminal formatting for CLI output
- **logging_config.py**: Structured logging with loguru
- **uploader.py**: S3/MinIO upload functionality with support for AWS S3 and S3-compatible storage
- **milvus_importer.py**: Direct import to Milvus with collection creation and indexing
- **zilliz_importer.py**: Direct import to Zilliz Cloud with optimized settings

### High-Performance Data Flow
1. Schema validation using Pydantic models (models.py)
2. **Vectorized data generation** using NumPy operations (optimized_writer.py)
3. **Direct PyArrow integration** for efficient Parquet writing
4. **Large batch processing** (50K+ rows per batch)
5. **Automatic file partitioning** (256MB chunks, 1M rows/file by default)
6. Output optimized for: Parquet (primary), JSON (fast serialization)

### Output Structure
The tool always generates a directory (not a single file) containing:
- One or more data files (automatically partitioned based on size/row limits)
- `meta.json` file with collection metadata and generation details

### Schema System
The tool supports two types of schemas:
- **Built-in schemas**: Located in src/milvus_fake_data/schemas/ as JSON files
- **Custom schemas**: User-managed schemas stored in ~/.milvus-fake-data/schemas/ with metadata

### Supported Milvus Field Types
- Numeric: Int8, Int16, Int32, Int64, Float, Double, Bool
- Text: VarChar, String (require max_length)
- Complex: JSON, Array (require element_type and max_capacity)
- Vectors: FloatVector, BinaryVector, Float16Vector, BFloat16Vector, Int8Vector, SparseFloatVector (require dim)

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
- **Small datasets** (<10K rows): 5,000-10,000 rows/sec
- **Medium datasets** (10K-100K rows): 7,000-15,000 rows/sec  
- **Large datasets** (100K+ rows): 10,000-25,000 rows/sec
- **Vector-heavy schemas**: 5,000-15,000 rows/sec
- **Text-heavy schemas**: 10,000-30,000 rows/sec

### Testing Approach
- pytest with coverage reporting
- Performance regression tests for large datasets
- Tests cover validation, generation, CLI interface, and schema management
- Integration tests for end-to-end workflows with 100K+ row datasets
- Test fixtures in tests/conftest.py for common scenarios

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
The package can also be used programmatically:
```python
from milvus_fake_data import SchemaManager

# Use built-in schemas
schema_manager = SchemaManager()
schema = schema_manager.get_schema("simple")

# Add custom schema programmatically
custom_schema = {
    "collection_name": "my_collection",
    "fields": [
        {"name": "id", "type": "Int64", "is_primary": True},
        {"name": "vector", "type": "FloatVector", "dim": 128}
    ]
}
schema_manager.add_schema("my_custom", custom_schema)
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

### Direct Import to Milvus/Zilliz Cloud
The tool can directly import generated data to Milvus or Zilliz Cloud:

**Milvus Import Features:**
- Automatic collection creation from metadata
- Smart index creation based on vector dimensions
- Batch processing for high-performance import
- Support for authentication and custom databases
- Connection testing before import

**Zilliz Cloud Import Features:**
- Optimized settings for cloud environment
- Smaller batch sizes for better network performance
- Advanced index selection (HNSW for large vectors)
- Progress tracking for long-running imports
- Automatic flush management

**Usage Examples:**
```bash
# Generate data and import to local Milvus
milvus-fake-data generate --builtin ecommerce --rows 100000 --output products/
milvus-fake-data to-milvus ./products/

# Import to Zilliz Cloud
milvus-fake-data to-zilliz ./products/ \
    --uri https://in03-xxx.api.gcp-us-west1.zillizcloud.com \
    --token YOUR_API_TOKEN \
    --drop-if-exists

# Import to remote Milvus with authentication
milvus-fake-data to-milvus ./products/ \
    --host 192.168.1.100 \
    --user root \
    --password Milvus \
    --collection-name product_vectors
```