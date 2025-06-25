# milvus-fake-data

A Python tool for generating mock data for Milvus collections based on schema definitions. This tool helps developers create realistic test data for Milvus vector databases quickly and efficiently.

## Features

- 🎯 **Built-in schemas** for common use cases (e-commerce, documents, images, users, etc.)
- 📚 **Custom schema management** - Add, show, list, and remove your own schemas
- 🚀 Generate mock data from JSON/YAML schema files
- 🔧 Support for all Milvus field types (Int8, Int16, Int32, Int64, Float, Double, Bool, VarChar, FloatVector, BinaryVector, Array, JSON)
- ✅ **Pydantic-based schema validation** with helpful error messages
- 📊 Multiple output formats (Parquet, CSV, JSON, NumPy)
- 🌱 Reproducible data generation with seed support
- 📦 Milvus-ready bulk writer integration
- 🎨 Customizable field constraints and properties
- 🔍 Schema validation and help commands
- 🏠 **Unified schema access** - Use custom schemas just like built-in ones

## Installation

```bash
# Install from PyPI (when published)
pip install milvus-fake-data

# Or install from source
git clone https://github.com/your-org/milvus-fake-data.git
cd milvus-fake-data
pdm install
```

## Quick Start

### Option 1: Use Built-in Schemas (Recommended)

Get started instantly with pre-built schemas for common use cases:

```bash
# List all available built-in schemas
milvus-fake-data --list-schemas

# Generate data using a built-in schema
milvus-fake-data --builtin simple --rows 1000

# Generate e-commerce product data
milvus-fake-data --builtin ecommerce --rows 5000 --preview

# Save a built-in schema for customization
milvus-fake-data --builtin documents --save-schema my_documents.json
```

Available built-in schemas:
- **simple** - Basic example with common field types
- **ecommerce** - Product catalog with search embeddings  
- **documents** - Document search with semantic embeddings
- **images** - Image gallery with visual similarity
- **users** - User profiles with behavioral embeddings
- **videos** - Video library with multimodal embeddings
- **news** - News articles with sentiment analysis

### Option 2: Create a Custom Schema File

Create a JSON or YAML file describing your Milvus collection schema:

```json
{
  "collection_name": "my_collection",
  "fields": [
    {
      "name": "id",
      "type": "Int64",
      "is_primary": true
    },
    {
      "name": "title",
      "type": "VarChar",
      "max_length": 256
    },
    {
      "name": "embedding",
      "type": "FloatVector",
      "dim": 128
    }
  ]
}
```

### Option 3: Generate Mock Data from Custom Schema

```bash
# Generate 1000 rows of mock data
milvus-fake-data --schema my_schema.json --rows 1000

# Generate CSV format with preview
milvus-fake-data --schema my_schema.json --rows 500 -f csv --preview
```

### Option 4: Manage Custom Schemas

Store and reuse your own schemas for consistent data generation:

```bash
# Create a custom schema file
cat > my_product_schema.json << 'EOF'
{
  "collection_name": "products",
  "fields": [
    {"name": "id", "type": "Int64", "is_primary": true},
    {"name": "name", "type": "VarChar", "max_length": 200},
    {"name": "category", "type": "VarChar", "max_length": 50},
    {"name": "price", "type": "Float", "min": 1.0, "max": 1000.0},
    {"name": "description_embedding", "type": "FloatVector", "dim": 384}
  ]
}
EOF

# Add it to your schema library
milvus-fake-data --add-schema my_products:my_product_schema.json
# Enter description: "Custom product catalog schema"
# Enter use cases: "e-commerce, testing"

# List all your schemas (built-in + custom)
milvus-fake-data --list-all-schemas

# Use your custom schema like a built-in one
milvus-fake-data --builtin my_products --rows 1000 --preview

# Show detailed information about any schema
milvus-fake-data --show-schema my_products
```

### Option 5: Use in Python Code

```python
from milvus_fake_data.generator import generate_mock_data
from milvus_fake_data.schema_manager import get_schema_manager
from milvus_fake_data.builtin_schemas import load_builtin_schema
from tempfile import NamedTemporaryFile
import json

# Use the schema manager to work with schemas
manager = get_schema_manager()

# List all available schemas
all_schemas = manager.list_all_schemas()
print("Available schemas:", list(all_schemas.keys()))

# Load any schema (built-in or custom)
schema = manager.load_schema("ecommerce")  # Built-in
# schema = manager.load_schema("my_products")  # Custom

# Generate data from schema
with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(schema, f)
    df = generate_mock_data(f.name, rows=1000, seed=42)

print(df.head())

# Add a custom schema programmatically
custom_schema = {
    "collection_name": "my_collection",
    "fields": [
        {"name": "id", "type": "Int64", "is_primary": True},
        {"name": "text", "type": "VarChar", "max_length": 100},
        {"name": "vector", "type": "FloatVector", "dim": 256}
    ]
}

manager.add_schema("my_custom", custom_schema, "Custom schema", ["testing"])
print("Added custom schema!")
```

## Schema Format

### Supported Field Types

| Type | Description | Required Parameters | Optional Parameters |
|------|-------------|-------------------|-------------------|
| `Int8`, `Int16`, `Int32`, `Int64` | Integer types | - | `min`, `max` |
| `Float`, `Double` | Floating point | - | `min`, `max` |
| `Bool` | Boolean | - | - |
| `VarChar` | Variable length string | `max_length` | - |
| `JSON` | JSON object | - | - |
| `FloatVector` | Float vector | `dim` | - |
| `BinaryVector` | Binary vector | `dim` | - |
| `Array` | Array type | `element_type`, `max_capacity` | - |

### Field Properties

- `is_primary`: Mark field as primary key
- `auto_id`: Auto-generate primary key values
- `nullable`: Allow null values (10% probability)

### Example Schema

```yaml
collection_name: "product_catalog"
fields:
  - name: "product_id"
    type: "Int64"
    is_primary: true
  
  - name: "title"
    type: "VarChar"
    max_length: 200
    nullable: false
  
  - name: "price"
    type: "Float"
    min: 0.01
    max: 9999.99
  
  - name: "features"
    type: "FloatVector"
    dim: 256
  
  - name: "tags"
    type: "Array"
    element_type: "VarChar"
    max_capacity: 10
  
  - name: "metadata"
    type: "JSON"
    nullable: true
```

## CLI Reference

```bash
milvus-fake-data [OPTIONS]

Options:
  --schema PATH              Path to schema JSON/YAML file
  --builtin SCHEMA_ID        Use built-in or custom schema
  --list-schemas             List all available built-in schemas
  --list-all-schemas         List all schemas (built-in and custom)
  --save-schema PATH         Save schema to file (use with --builtin)
  --add-schema ID:FILE       Add a custom schema (format: 'schema_id:schema_file.json')
  --show-schema SCHEMA_ID    Show details of a schema (built-in or custom)
  --remove-schema SCHEMA_ID  Remove a custom schema
  --validate-only           Only validate schema without generating data
  --schema-help             Show schema format help and examples
  --rows INTEGER            Number of rows to generate [default: 1000]
  -f, --format [parquet|csv|json|npy]  Output format [default: parquet]
  -p, --preview             Print first 5 rows after generation
  --out PATH                Output file path
  --seed INTEGER            Random seed for reproducibility
  -h, --help                Show help message
```

### Schema Management Commands

```bash
# List built-in schemas only
milvus-fake-data --list-schemas

# List all schemas (built-in and custom)
milvus-fake-data --list-all-schemas

# Show details of any schema
milvus-fake-data --show-schema ecommerce
milvus-fake-data --show-schema my_custom_schema

# Add a custom schema
milvus-fake-data --add-schema my_schema:my_schema.json

# Remove a custom schema
milvus-fake-data --remove-schema my_schema

# Generate data with built-in schema
milvus-fake-data --builtin ecommerce --rows 1000

# Generate data with custom schema
milvus-fake-data --builtin my_custom_schema --rows 1000

# Save any schema for customization
milvus-fake-data --builtin news --save-schema news_schema.json

# Validate any schema
milvus-fake-data --builtin users --validate-only
```

## Development

This project uses PDM for dependency management and follows Python best practices.

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/your-org/milvus-fake-data.git
cd milvus-fake-data

# Install dependencies
pdm install
```

### Code Quality

```bash
# Format code
pdm run ruff format src tests

# Lint code
pdm run ruff check src tests
pdm run mypy src

# Run tests
pdm run pytest
pdm run pytest --cov=src tests/
```

### Running Tests

```bash
# Run all tests
pdm run pytest

# Run with coverage
pdm run pytest --cov=src --cov-report=html

# Run specific test file
pdm run pytest tests/test_generator.py
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and ensure code quality checks pass
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built for the [Milvus](https://milvus.io/) vector database
- Uses [Faker](https://faker.readthedocs.io/) for realistic data generation
- Powered by [Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/)
