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
            # Log import preparation details
            self.logger.info(f"Preparing bulk import for collection: {collection_name}")
            self.logger.info(f"Target Milvus URI: {self.uri}")
            self.logger.info(f"Database: {self.db_name}")
            self.logger.info(f"Number of files to import: {len(files)}")

            # Log file details
            for i, file_path in enumerate(files, 1):
                if file_path.startswith("s3://"):
                    self.logger.info(f"File {i}: {file_path} (S3/MinIO)")
                else:
                    # Check if local file exists and get size
                    path = Path(file_path)
                    if path.exists():
                        size_mb = path.stat().st_size / (1024 * 1024)
                        self.logger.info(f"File {i}: {file_path} ({size_mb:.2f} MB)")
                    else:
                        self.logger.info(f"File {i}: {file_path} (path not found locally)")

            # Prepare files as list of lists (each inner list is a batch)
            file_batches = [[f] for f in files]
            self.logger.info(f"Organized files into {len(file_batches)} import batches")

            # Start bulk import using bulk_writer
            self.logger.info("Initiating bulk import request to Milvus...")
            resp = bulk_import(
                url=self.uri,
                collection_name=collection_name,
                files=file_batches,
            )

            # Extract job ID from response
            response_data = resp.json()
            job_id = response_data["data"]["jobId"]

            self.logger.info("✓ Bulk import request accepted successfully")
            self.logger.info(f"Job ID: {job_id}")
            self.logger.info(f"Collection: {collection_name}")
            self.logger.info("Status: Import job queued and will be processed asynchronously")

            return job_id

        except Exception as e:
            self.logger.error(f"Failed to start bulk import: {e}")
            self.logger.error(f"Collection: {collection_name}")
            self.logger.error(f"Files: {files}")
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
                    progress_percent = job_info.get("progress", 0)
                    imported_rows = job_info.get("importedRows", 0)
                    total_rows = job_info.get("totalRows", 0)
                    file_size = job_info.get("fileSize", 0)

                    # Log detailed progress information
                    elapsed = time.time() - start_time
                    if elapsed % 10 < 2:  # Log every 10 seconds
                        self.logger.info(f"Import progress update for job {job_id}:")
                        self.logger.info(f"  State: {state}")
                        self.logger.info(f"  Progress: {progress_percent}%")
                        self.logger.info(f"  Imported rows: {imported_rows:,} / {total_rows:,}")
                        self.logger.info(f"  File size processed: {file_size:,} bytes")
                        self.logger.info(f"  Elapsed time: {elapsed:.1f}s")

                    if state == "ImportCompleted" or state == "Completed":
                        progress.update(task, completed=timeout)
                        self.logger.info("🎉 Bulk import completed successfully!")
                        self.logger.info(f"Job ID: {job_id}")
                        self.logger.info(f"Total rows imported: {imported_rows:,}")
                        self.logger.info(f"Total file size: {file_size:,} bytes")
                        self.logger.info(f"Total time: {elapsed:.2f}s")
                        if imported_rows > 0 and elapsed > 0:
                            rate = imported_rows / elapsed
                            self.logger.info(f"Import rate: {rate:.0f} rows/second")
                        return True
                    elif state == "ImportFailed" or state == "Failed":
                        progress.update(task, completed=timeout)
                        reason = job_info.get("reason", "Unknown error")
                        self.logger.error("❌ Bulk import failed!")
                        self.logger.error(f"Job ID: {job_id}")
                        self.logger.error(f"Failure reason: {reason}")
                        self.logger.error(f"State: {state}")
                        self.logger.error(f"Progress when failed: {progress_percent}%")
                        self.logger.error(f"Rows imported before failure: {imported_rows:,}")
                        return False

                    # Update progress
                    progress.update(task, completed=min(elapsed, timeout))

                    time.sleep(2)

        else:
            # Wait without progress bar
            self.logger.info(f"Monitoring import job {job_id} (no progress bar)")
            last_log_time = 0

            while time.time() - start_time < timeout:
                resp = get_import_progress(
                    url=self.uri,
                    job_id=job_id,
                )
                job_info = resp.json()["data"]
                state = job_info.get("state", "unknown")
                progress_percent = job_info.get("progress", 0)
                imported_rows = job_info.get("importedRows", 0)
                total_rows = job_info.get("totalRows", 0)
                file_size = job_info.get("fileSize", 0)

                # Log detailed progress information every 10 seconds
                elapsed = time.time() - start_time
                if elapsed - last_log_time >= 10:
                    self.logger.info(f"Import progress update for job {job_id}:")
                    self.logger.info(f"  State: {state}")
                    self.logger.info(f"  Progress: {progress_percent}%")
                    self.logger.info(f"  Imported rows: {imported_rows:,} / {total_rows:,}")
                    self.logger.info(f"  File size processed: {file_size:,} bytes")
                    self.logger.info(f"  Elapsed time: {elapsed:.1f}s")
                    last_log_time = elapsed

                if state == "ImportCompleted" or state == "Completed":
                    self.logger.info("🎉 Bulk import completed successfully!")
                    self.logger.info(f"Job ID: {job_id}")
                    self.logger.info(f"Total rows imported: {imported_rows:,}")
                    self.logger.info(f"Total file size: {file_size:,} bytes")
                    self.logger.info(f"Total time: {elapsed:.2f}s")
                    if imported_rows > 0 and elapsed > 0:
                        rate = imported_rows / elapsed
                        self.logger.info(f"Import rate: {rate:.0f} rows/second")
                    return True
                elif state == "ImportFailed" or state == "Failed":
                    reason = job_info.get("reason", "Unknown error")
                    self.logger.error("❌ Bulk import failed!")
                    self.logger.error(f"Job ID: {job_id}")
                    self.logger.error(f"Failure reason: {reason}")
                    self.logger.error(f"State: {state}")
                    self.logger.error(f"Progress when failed: {progress_percent}%")
                    self.logger.error(f"Rows imported before failure: {imported_rows:,}")
                    return False

                time.sleep(2)

        # Timeout reached
        elapsed = time.time() - start_time
        self.logger.error(f"⏰ Bulk import timeout after {timeout} seconds")
        self.logger.error(f"Job ID: {job_id}")
        self.logger.error(f"Final state: {job_info.get('state', 'unknown')}")
        self.logger.error(f"Progress at timeout: {job_info.get('progress', 0)}%")
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
            self.logger.info(f"Listing import jobs from Milvus: {self.uri}")
            if collection_name:
                self.logger.info(f"Filtering by collection: {collection_name}")
                resp = list_import_jobs(
                    url=self.uri,
                    collection_name=collection_name,
                )
            else:
                self.logger.info("Listing all import jobs")
                resp = list_import_jobs(
                    url=self.uri,
                )

            jobs = resp.json()["data"]["records"]

            self.logger.info(f"📋 Found {len(jobs)} import jobs")

            # Log summary of jobs by state
            if jobs:
                states = {}
                for job in jobs:
                    state = job.get("state", "unknown")
                    states[state] = states.get(state, 0) + 1

                self.logger.info("Job summary by state:")
                for state, count in states.items():
                    self.logger.info(f"  {state}: {count} jobs")

                # Log details of recent jobs
                recent_jobs = sorted(jobs, key=lambda x: x.get("jobId", ""), reverse=True)[:5]
                self.logger.info(f"Recent {min(5, len(jobs))} jobs:")
                for job in recent_jobs:
                    job_id = job.get("jobId", "unknown")
                    state = job.get("state", "unknown")
                    collection = job.get("collectionName", "unknown")
                    imported_rows = job.get("importedRows", 0)
                    self.logger.info(f"  Job {job_id}: {state} | Collection: {collection} | Rows: {imported_rows:,}")

            return jobs

        except Exception as e:
            self.logger.error(f"Failed to list import jobs: {e}")
            self.logger.error(f"URI: {self.uri}")
            if collection_name:
                self.logger.error(f"Collection filter: {collection_name}")
            raise


def prepare_file_paths(files: list[str]) -> list[str]:
    """Prepare file paths for bulk import.

    Args:
        files: List of file paths or directories

    Returns:
        List of prepared file paths
    """
    logger = get_logger(__name__)
    logger.info("Preparing file paths for bulk import")
    logger.info(f"Input paths: {len(files)} items")

    prepared_files = []

    for file_path in files:
        path = Path(file_path)
        logger.info(f"Processing path: {file_path}")

        if file_path.startswith("s3://"):
            # S3/MinIO path - use as is
            prepared_files.append(file_path)
            logger.info(f"  Added S3 path: {file_path}")
        elif path.is_dir():
            # Local directory - find files
            logger.info(f"  Processing directory: {path}")
            parquet_files = list(path.glob("*.parquet"))
            if parquet_files:
                prepared_files.extend([str(f) for f in parquet_files])
                logger.info(f"  Found {len(parquet_files)} parquet files")
                for pf in parquet_files:
                    size_mb = pf.stat().st_size / (1024 * 1024)
                    logger.info(f"    {pf.name} ({size_mb:.2f} MB)")
            else:
                # Add all files in directory
                all_files = [f for f in path.iterdir() if f.is_file()]
                prepared_files.extend([str(f) for f in all_files])
                logger.info(f"  Found {len(all_files)} other files")
                for af in all_files:
                    size_mb = af.stat().st_size / (1024 * 1024)
                    logger.info(f"    {af.name} ({size_mb:.2f} MB)")
        elif path.exists():
            # Single local file
            prepared_files.append(str(path))
            size_mb = path.stat().st_size / (1024 * 1024)
            logger.info(f"  Added file: {path.name} ({size_mb:.2f} MB)")
        else:
            # Path doesn't exist locally, might be S3 or remote
            prepared_files.append(file_path)
            logger.info(f"  Added path (not found locally): {file_path}")

    logger.info(f"✓ Prepared {len(prepared_files)} files for import")
    return prepared_files
