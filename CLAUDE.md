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

# Or use the .env.example file in project root
cp .env.example .env  # Create and edit with your values
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
milvus-ingest generate --builtin simple --rows 100000 --preview        # Generate 100K rows
milvus-ingest generate --schema example_schema.json --rows 1000000     # Generate 1M rows
milvus-ingest generate --schema schema.json --rows 5000000 --batch-size 100000 # Use large batch size

# Additional generation options
milvus-ingest generate --schema schema.json --validate-only            # Validate schema without generating
milvus-ingest generate --schema schema.json --rows 1000000 --no-progress # Disable progress bar
milvus-ingest generate --schema schema.json --max-file-size 512       # Set max file size (MB)
milvus-ingest generate --schema schema.json --max-rows-per-file 500000 # Set max rows per file
milvus-ingest generate --schema schema.json --out mydata --force   # Force overwrite existing output

# Schema management commands
milvus-ingest schema list                    # List all schemas
milvus-ingest schema show simple            # Show schema details
milvus-ingest schema add myschema file.json # Add custom schema
milvus-ingest schema remove myschema        # Remove custom schema
milvus-ingest schema help                   # Schema format help

# Utility commands
milvus-ingest clean                         # Clean up generated files

# Upload to S3/MinIO (standalone upload, useful for separate upload/import workflow)
milvus-ingest upload --local-path ./output --s3-path s3://bucket/prefix/              # Upload to AWS S3
milvus-ingest upload --local-path ./output --s3-path s3://bucket/prefix/ --endpoint-url http://localhost:9000  # Upload to MinIO
milvus-ingest upload --local-path ./output --s3-path s3://bucket/prefix/ --no-verify-ssl  # Disable SSL verification
milvus-ingest upload --local-path ./output --s3-path s3://bucket/prefix/ --access-key-id KEY --secret-access-key SECRET  # With credentials

# Send data to Milvus
# Direct insert to Milvus (reads local parquet and JSON files and creates collection)
milvus-ingest to-milvus insert ./output                                # Insert to local Milvus
milvus-ingest to-milvus insert ./output --uri http://192.168.1.100:19530 --token your-token  # Remote Milvus with auth
milvus-ingest to-milvus insert ./output --drop-if-exists               # Drop existing collection and recreate
milvus-ingest to-milvus insert ./output --collection-name my_collection --batch-size 5000  # Custom settings

# Bulk import to Milvus (upload + import in one step)
# Note: Combines upload and import for convenience, includes auto-collection creation
milvus-ingest to-milvus import --local-path ./output/ --s3-path data/ --bucket my-bucket --endpoint-url http://minio:9000  # Upload and import
milvus-ingest to-milvus import --local-path ./output/ --s3-path data/ --bucket my-bucket --endpoint-url http://minio:9000 --collection-name my_collection  # Override collection name
milvus-ingest to-milvus import --local-path ./output/ --s3-path data/ --bucket my-bucket --endpoint-url http://minio:9000 --wait  # Wait for completion
milvus-ingest to-milvus import --local-path ./output/ --s3-path data/ --bucket my-bucket --endpoint-url http://minio:9000 --access-key-id key --secret-access-key secret  # With credentials
milvus-ingest to-milvus import --local-path ./output/ --s3-path data/ --bucket my-bucket --endpoint-url http://minio:9000 --drop-if-exists  # Drop and recreate
```

## Architecture Overview

### Project Structure
```
milvus-ingest/
├── src/milvus_fake_data/       # Main package
│   ├── cli.py                  # CLI entry point
│   ├── optimized_writer.py     # High-performance data generation
│   ├── models.py               # Pydantic schema models
│   ├── schema_manager.py       # Schema management
│   ├── milvus_inserter.py      # Direct Milvus insertion
│   ├── milvus_importer.py      # Bulk import from S3/MinIO
│   ├── uploader.py             # S3/MinIO uploads
│   ├── builtin_schemas.py      # Built-in schema definitions
│   ├── generator.py            # Legacy generator (for compatibility)
│   ├── rich_display.py         # Rich terminal formatting
│   ├── logging_config.py       # Loguru-based structured logging
│   ├── exceptions.py           # Custom exception classes
│   └── schemas/                # 15 built-in schema JSON files
├── tests/                      # Test suite
│   ├── conftest.py            # Common test fixtures
│   └── test_*.py              # Test modules
├── pyproject.toml             # PDM configuration and metadata
├── Makefile                   # Common development tasks
└── README.md                  # User documentation
```

### High-Level Architecture

This is a high-performance mock data generator for Milvus vector databases with several key design principles:

1. **Performance-First Design**: Uses vectorized NumPy operations to achieve 10,000-100,000+ rows/second generation speed. The `optimized_writer.py` module handles batch processing and memory-efficient streaming.

2. **Schema-Driven Generation**: All data generation is driven by JSON schemas validated with Pydantic models. Schemas define field types, dimensions, cardinality, and generation modes (faker patterns, ranges, custom values).

3. **Flexible Output Pipeline**:
   - Generate → Parquet/JSON files → Direct insert to Milvus
   - Generate → Parquet files → Upload to S3/MinIO → Bulk import to Milvus

4. **Smart File Partitioning**: Automatically splits output into multiple files based on configurable limits (default: 256MB or 1M rows per file) to prevent memory issues and optimize import performance.

5. **Built-in Schema Library**: Includes 15 pre-configured schemas for common use cases (simple vectors, e-commerce, documents, images, etc.) with support for custom schemas.

6. **CLI Architecture**: Click-based command groups (generate, schema, upload, to-milvus, clean) with rich terminal output for better user experience.

## Testing Workflow Memories

- **使用.env文件中的设置进行测试**：在本地测试环境中，可以通过 `.env.example` 文件复制并配置测试所需的环境变量，确保测试使用正确的配置。例如，使用 `cp .env.example .env` 并根据需要编辑文件中的值。