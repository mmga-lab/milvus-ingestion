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

# Test the CLI (backward compatible - defaults to generate command)
milvus-fake-data --builtin simple --rows 100 --preview
milvus-fake-data --schema example_schema.json --rows 1000

# New grouped CLI structure
milvus-fake-data generate --builtin simple --rows 100 --preview
milvus-fake-data generate --schema example_schema.json --rows 1000

# Schema management commands
milvus-fake-data schema list                    # List all schemas
milvus-fake-data schema show simple            # Show schema details
milvus-fake-data schema add myschema file.json # Add custom schema
milvus-fake-data schema remove myschema        # Remove custom schema
milvus-fake-data schema help                   # Schema format help

# Utility commands
milvus-fake-data clean                         # Clean up generated files
```

## Architecture Overview

### Core Modules
- **cli.py**: Command-line interface with Click groups, provides both backward compatibility and new grouped structure for better organization
- **generator.py**: Core data generation logic, takes schema files and produces pandas DataFrames with mock data
- **models.py**: Pydantic models for schema validation with comprehensive error messages
- **schema_manager.py**: Manages custom and built-in schemas, stores user schemas in ~/.milvus-fake-data/schemas
- **builtin_schemas.py**: Pre-built schemas for common use cases (e-commerce, documents, images, etc.)
- **rich_display.py**: Rich terminal formatting for CLI output
- **logging_config.py**: Structured logging with loguru

### Data Flow
1. Schema validation using Pydantic models (models.py)
2. Data generation using Faker library (generator.py)
3. Output in multiple formats: Parquet, CSV, JSON, NumPy

### Schema System
The tool supports two types of schemas:
- **Built-in schemas**: Located in src/milvus_fake_data/schemas/ as JSON files
- **Custom schemas**: User-managed schemas stored in ~/.milvus-fake-data/schemas/ with metadata

### Supported Milvus Field Types
- Numeric: Int8, Int16, Int32, Int64, Float, Double, Bool
- Text: VarChar, String (require max_length)
- Complex: JSON, Array (require element_type and max_capacity)
- Vectors: FloatVector, BinaryVector, Float16Vector, BFloat16Vector, Int8Vector, SparseFloatVector (require dim)

## Key Implementation Details

### Schema Validation
Uses Pydantic for comprehensive validation with helpful error messages. The validation ensures:
- Exactly one primary key field
- Required parameters for each field type (e.g., dim for vectors, max_length for strings)
- Proper field name format (alphanumeric + underscores)
- auto_id only for Int64 primary keys

### Data Generation Strategy
- Uses Faker library for realistic mock data
- Supports seeded generation for reproducibility
- Handles field constraints (min/max values, string lengths)
- Generates appropriate vector dimensions and array capacities

### Testing Approach
- pytest with coverage reporting
- Tests cover validation, generation, CLI interface, and schema management
- Integration tests for end-to-end workflows
- Test fixtures in tests/conftest.py for common scenarios

## Important Notes

### Error Handling
The tool provides detailed error messages for schema validation failures, pointing users to exact issues and providing example fixes.

### Performance
For large datasets, the generator supports batch processing to manage memory usage efficiently.

### Environment Variables
- MILVUS_FAKE_DATA_SCHEMA_DIR: Override default schema directory for testing