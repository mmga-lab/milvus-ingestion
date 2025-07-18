"""Bulk import functionality for Milvus using bulk_import API."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pymilvus.bulk_writer import bulk_import, get_import_progress, list_import_jobs
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from .logging_config import get_logger


class MilvusBulkImporter:
    """Handle bulk importing data to Milvus using bulk_import API."""

    def __init__(
        self,
        uri: str = "http://localhost:19530",
        token: str = "",
        db_name: str = "default",
    ):
        """Initialize Milvus connection.

        Args:
            uri: Milvus server URI (e.g., http://localhost:19530)
            token: Token for authentication
            db_name: Database name
        """
        self.uri = uri
        self.token = token
        self.db_name = db_name
        self.logger = get_logger(__name__)

    def bulk_import_files(
        self,
        collection_name: str,
        files: list[str],
        show_progress: bool = True,
    ) -> str:
        """Start bulk import job.

        Args:
            collection_name: Target collection name
            files: List of file paths to import
            show_progress: Show progress bar

        Returns:
            Job ID for the import task
        """
        try:
            # Prepare files as list of lists (each inner list is a batch)
            file_batches = [[f] for f in files]

            # Start bulk import using bulk_writer
            resp = bulk_import(
                url=self.uri,
                collection_name=collection_name,
                files=file_batches,
            )

            # Extract job ID from response
            job_id = resp.json()["data"]["jobId"]

            if show_progress:
                self.logger.info(f"Bulk import started with job ID: {job_id}")

            return job_id

        except Exception as e:
            self.logger.error(f"Failed to start bulk import: {e}")
            raise

    def wait_for_completion(
        self,
        job_id: str,
        timeout: int = 300,
        show_progress: bool = True,
    ) -> bool:
        """Wait for bulk import job to complete.

        Args:
            job_id: Import job ID
            timeout: Timeout in seconds
            show_progress: Show progress bar

        Returns:
            True if import completed successfully
        """
        start_time = time.time()

        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task(
                    "Waiting for import completion...", total=timeout
                )

                while time.time() - start_time < timeout:
                    # Check job status
                    resp = get_import_progress(
                        url=self.uri,
                        job_id=job_id,
                    )
                    job_info = resp.json()["data"]
                    state = job_info.get("state", "unknown")

                    if state == "ImportCompleted":
                        progress.update(task, completed=timeout)
                        self.logger.info(
                            f"Bulk import completed successfully: {job_id}"
                        )
                        return True
                    elif state == "ImportFailed":
                        progress.update(task, completed=timeout)
                        reason = job_info.get("reason", "Unknown error")
                        self.logger.error(f"Bulk import failed: {reason}")
                        return False

                    # Update progress
                    elapsed = time.time() - start_time
                    progress.update(task, completed=min(elapsed, timeout))

                    time.sleep(2)

        else:
            # Wait without progress bar
            while time.time() - start_time < timeout:
                resp = get_import_progress(
                    url=self.uri,
                    job_id=job_id,
                )
                job_info = resp.json()["data"]
                state = job_info.get("state", "unknown")

                if state == "ImportCompleted":
                    self.logger.info(f"Bulk import completed successfully: {job_id}")
                    return True
                elif state == "ImportFailed":
                    reason = job_info.get("reason", "Unknown error")
                    self.logger.error(f"Bulk import failed: {reason}")
                    return False

                time.sleep(2)

        # Timeout reached
        self.logger.error(f"Bulk import timeout after {timeout} seconds")
        return False

    def list_import_jobs(
        self,
        collection_name: str | None = None,
        show_progress: bool = True,
    ) -> list[dict[str, Any]]:
        """List all import jobs.

        Args:
            collection_name: Filter by collection name
            show_progress: Show progress bar

        Returns:
            List of import job information
        """
        try:
            if collection_name:
                resp = list_import_jobs(
                    url=self.uri,
                    collection_name=collection_name,
                )
            else:
                resp = list_import_jobs(
                    url=self.uri,
                )

            jobs = resp.json()["data"]["records"]

            if show_progress:
                self.logger.info(f"Found {len(jobs)} import jobs")

            return jobs

        except Exception as e:
            self.logger.error(f"Failed to list import jobs: {e}")
            raise


def prepare_file_paths(files: list[str]) -> list[str]:
    """Prepare file paths for bulk import.

    Args:
        files: List of file paths or directories

    Returns:
        List of prepared file paths
    """
    prepared_files = []

    for file_path in files:
        path = Path(file_path)

        if path.is_dir():
            # Add all parquet files in directory
            parquet_files = list(path.glob("*.parquet"))
            if parquet_files:
                prepared_files.extend([str(f) for f in parquet_files])
            else:
                # Add all files in directory
                all_files = [f for f in path.iterdir() if f.is_file()]
                prepared_files.extend([str(f) for f in all_files])
        else:
            # Add single file
            prepared_files.append(str(path))

    return prepared_files
