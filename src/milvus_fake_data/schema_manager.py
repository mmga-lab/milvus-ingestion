"""Schema management functionality for adding and showing custom schemas."""

import json
import os
from pathlib import Path
from typing import Any

from .builtin_schemas import get_schema_info, list_builtin_schemas, load_builtin_schema
from .logging_config import get_logger
from .models import validate_schema_data


class SchemaManager:
    """Manages custom and built-in schemas."""

    def __init__(self, schema_dir: Path | None = None):
        """Initialize schema manager.

        Args:
            schema_dir: Directory to store custom schemas. Defaults to ~/.milvus-fake-data/schemas
        """
        self.logger = get_logger(__name__)

        if schema_dir is None:
            # Allow override via environment variable for testing
            env_path = os.environ.get("MILVUS_FAKE_DATA_SCHEMA_DIR")
            if env_path:
                schema_dir = Path(env_path)
            else:
                schema_dir = Path.home() / ".milvus-fake-data" / "schemas"

        self.schema_dir = Path(schema_dir)
        self.schema_dir.mkdir(parents=True, exist_ok=True)
        self.logger.debug("Schema manager initialized", schema_dir=str(self.schema_dir))

        # Create metadata file if it doesn't exist
        self.metadata_file = self.schema_dir / "metadata.json"
        if not self.metadata_file.exists():
            self._save_metadata({})
            self.logger.debug(
                "Created new metadata file", metadata_file=str(self.metadata_file)
            )

    def _load_metadata(self) -> dict[str, Any]:
        """Load schema metadata."""
        try:
            with open(self.metadata_file, encoding="utf-8") as f:
                metadata: dict[str, Any] = json.load(f)
                self.logger.debug(
                    "Metadata loaded successfully", schema_count=len(metadata)
                )
                return metadata
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.warning(
                "Failed to load metadata, returning empty dict", error=str(e)
            )
            return {}

    def _save_metadata(self, metadata: dict[str, Any]) -> None:
        """Save schema metadata."""
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def add_schema(
        self,
        schema_id: str,
        schema_data: dict[str, Any],
        description: str = "",
        use_cases: list[str] | None = None,
    ) -> None:
        """Add a custom schema.

        Args:
            schema_id: Unique identifier for the schema
            schema_data: Schema definition
            description: Schema description
            use_cases: List of use cases

        Raises:
            ValueError: If schema_id already exists or schema is invalid
        """
        if self.schema_exists(schema_id):
            raise ValueError(
                f"Schema '{schema_id}' already exists. Use update_schema to modify it."
            )

        # Validate schema
        try:
            validate_schema_data(schema_data)
        except Exception as e:
            raise ValueError(f"Invalid schema: {e}") from e

        # Save schema file
        schema_file = self.schema_dir / f"{schema_id}.json"
        with open(schema_file, "w", encoding="utf-8") as f:
            json.dump(schema_data, f, indent=2, ensure_ascii=False)

        # Update metadata
        metadata = self._load_metadata()
        metadata[schema_id] = {
            "name": schema_data.get("collection_name", schema_id),
            "description": description or schema_data.get("description", ""),
            "use_cases": use_cases or [],
            "fields_count": len(schema_data.get("fields", [])),
            "vector_dims": self._extract_vector_dims(schema_data),
            "file": f"{schema_id}.json",
            "type": "custom",
        }
        self._save_metadata(metadata)

    def update_schema(
        self,
        schema_id: str,
        schema_data: dict[str, Any],
        description: str = "",
        use_cases: list[str] | None = None,
    ) -> None:
        """Update an existing custom schema.

        Args:
            schema_id: Schema identifier
            schema_data: Updated schema definition
            description: Updated description
            use_cases: Updated use cases

        Raises:
            ValueError: If schema doesn't exist or is built-in, or schema is invalid
        """
        if not self.schema_exists(schema_id):
            raise ValueError(
                f"Schema '{schema_id}' does not exist. Use add_schema to create it."
            )

        if self.is_builtin_schema(schema_id):
            raise ValueError(f"Cannot update built-in schema '{schema_id}'.")

        # Validate schema
        try:
            validate_schema_data(schema_data)
        except Exception as e:
            raise ValueError(f"Invalid schema: {e}") from e

        # Update schema file
        schema_file = self.schema_dir / f"{schema_id}.json"
        with open(schema_file, "w", encoding="utf-8") as f:
            json.dump(schema_data, f, indent=2, ensure_ascii=False)

        # Update metadata
        metadata = self._load_metadata()
        metadata[schema_id].update(
            {
                "name": schema_data.get("collection_name", schema_id),
                "description": description or schema_data.get("description", ""),
                "use_cases": use_cases or [],
                "fields_count": len(schema_data.get("fields", [])),
                "vector_dims": self._extract_vector_dims(schema_data),
            }
        )
        self._save_metadata(metadata)

    def remove_schema(self, schema_id: str) -> None:
        """Remove a custom schema.

        Args:
            schema_id: Schema identifier

        Raises:
            ValueError: If schema doesn't exist or is built-in
        """
        if not self.schema_exists(schema_id):
            raise ValueError(f"Schema '{schema_id}' does not exist.")

        if self.is_builtin_schema(schema_id):
            raise ValueError(f"Cannot remove built-in schema '{schema_id}'.")

        # Remove schema file
        schema_file = self.schema_dir / f"{schema_id}.json"
        if schema_file.exists():
            schema_file.unlink()

        # Remove from metadata
        metadata = self._load_metadata()
        if schema_id in metadata:
            del metadata[schema_id]
            self._save_metadata(metadata)

    def load_schema(self, schema_id: str) -> dict[str, Any]:
        """Load a schema by ID.

        Args:
            schema_id: Schema identifier

        Returns:
            Schema data

        Raises:
            ValueError: If schema doesn't exist
        """
        if self.is_builtin_schema(schema_id):
            return load_builtin_schema(schema_id)

        if not self.schema_exists(schema_id):
            raise ValueError(f"Schema '{schema_id}' does not exist.")

        schema_file = self.schema_dir / f"{schema_id}.json"
        with open(schema_file, encoding="utf-8") as f:
            return json.load(f)  # type: ignore[no-any-return]

    def get_schema_info(self, schema_id: str) -> dict[str, Any] | None:
        """Get schema information.

        Args:
            schema_id: Schema identifier

        Returns:
            Schema information or None if not found
        """
        # Check built-in schemas first
        builtin_info = get_schema_info(schema_id)
        if builtin_info:
            return builtin_info

        # Check custom schemas
        metadata = self._load_metadata()
        return metadata.get(schema_id)

    def list_all_schemas(self) -> dict[str, dict[str, Any]]:
        """List all schemas (built-in + custom).

        Returns:
            Dictionary of schema_id -> schema_info
        """
        # Start with built-in schemas
        all_schemas = list_builtin_schemas()

        # Add custom schemas
        custom_metadata = self._load_metadata()
        all_schemas.update(custom_metadata)

        return all_schemas

    def list_custom_schemas(self) -> dict[str, dict[str, Any]]:
        """List only custom schemas.

        Returns:
            Dictionary of schema_id -> schema_info
        """
        return self._load_metadata()

    def schema_exists(self, schema_id: str) -> bool:
        """Check if a schema exists.

        Args:
            schema_id: Schema identifier

        Returns:
            True if schema exists
        """
        # Check built-in schemas
        if get_schema_info(schema_id):
            return True

        # Check custom schemas
        custom_metadata = self._load_metadata()
        return schema_id in custom_metadata

    def is_builtin_schema(self, schema_id: str) -> bool:
        """Check if a schema is built-in.

        Args:
            schema_id: Schema identifier

        Returns:
            True if schema is built-in
        """
        return get_schema_info(schema_id) is not None

    def _extract_vector_dims(self, schema_data: dict[str, Any]) -> list[int]:
        """Extract vector dimensions from schema.

        Args:
            schema_data: Schema definition

        Returns:
            List of vector dimensions
        """
        dims = []
        for field in schema_data.get("fields", []):
            if field.get("type", "").endswith("Vector"):
                dim = field.get("dim")
                if dim and dim not in dims:
                    dims.append(dim)
        return sorted(dims)


# Global schema manager instance
_schema_manager = None


def get_schema_manager() -> SchemaManager:
    """Get the global schema manager instance."""
    global _schema_manager
    if _schema_manager is None:
        _schema_manager = SchemaManager()
    return _schema_manager


def reset_schema_manager() -> None:
    """Reset the global schema manager instance (for testing)."""
    global _schema_manager
    _schema_manager = None
