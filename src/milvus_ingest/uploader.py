"""S3/MinIO upload functionality for generated data files."""

from __future__ import annotations

import os
from pathlib import Path  # noqa: TC003
from typing import Any
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from .logging_config import get_logger
from .rich_display import display_error, display_info


class S3Uploader:
    """Handle uploads to S3-compatible storage (S3, MinIO, etc.)."""

    def __init__(
        self,
        endpoint_url: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        region_name: str = "us-east-1",
        verify_ssl: bool = True,
    ):
        """Initialize S3 client.

        Args:
            endpoint_url: S3-compatible endpoint URL (e.g., http://localhost:9000 for MinIO)
            access_key_id: AWS access key ID (can also be set via AWS_ACCESS_KEY_ID env var)
            secret_access_key: AWS secret access key (can also be set via AWS_SECRET_ACCESS_KEY env var)
            region_name: AWS region name (default: us-east-1)
            verify_ssl: Whether to verify SSL certificates (default: True)
        """
        self.logger = get_logger(__name__)

        # Get credentials from environment if not provided
        if not access_key_id:
            access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
        if not secret_access_key:
            secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

        # Create S3 client
        try:
            self.s3_client = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                region_name=region_name,
                verify=verify_ssl,
            )
            self.endpoint_url = endpoint_url
            self.logger.info(
                "S3 client initialized",
                extra={
                    "endpoint": endpoint_url or "AWS S3",
                    "region": region_name,
                },
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize S3 client: {e}")
            raise

    def upload_directory(
        self,
        local_path: Path,
        bucket: str,
        prefix: str = "",
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """Upload a directory to S3/MinIO.

        Args:
            local_path: Local directory path containing files to upload
            bucket: S3 bucket name
            prefix: Optional prefix (folder) in the bucket
            show_progress: Whether to show upload progress

        Returns:
            Dictionary with upload statistics
        """
        if not local_path.exists():
            raise FileNotFoundError(f"Directory not found: {local_path}")

        if not local_path.is_dir():
            raise ValueError(f"Path is not a directory: {local_path}")

        # Check if bucket exists, create if not
        self._ensure_bucket_exists(bucket)

        # Collect all files to upload
        files_to_upload = []
        total_size = 0

        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                files_to_upload.append(file_path)
                total_size += file_path.stat().st_size

        if not files_to_upload:
            display_info(f"No files found in {local_path}")
            return {"uploaded_files": 0, "total_size": 0, "failed_files": []}

        # Upload files
        uploaded_files = 0
        failed_files = []

        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            ) as progress:
                task = progress.add_task(
                    f"Uploading {len(files_to_upload)} files",
                    total=len(files_to_upload),
                )

                for file_path in files_to_upload:
                    # Calculate S3 key
                    relative_path = file_path.relative_to(local_path)
                    relative_path_str = str(relative_path).replace(
                        "\\", "/"
                    )  # Convert to forward slashes

                    # Build S3 key, ensuring no double slashes
                    if prefix:
                        # Remove trailing slash from prefix if present
                        clean_prefix = prefix.rstrip("/")
                        s3_key = f"{clean_prefix}/{relative_path_str}"
                    else:
                        s3_key = relative_path_str

                    try:
                        self._upload_file(file_path, bucket, s3_key)
                        uploaded_files += 1
                        progress.update(task, advance=1)
                    except Exception as e:
                        self.logger.error(f"Failed to upload {file_path}: {e}")
                        failed_files.append({"file": str(file_path), "error": str(e)})
                        progress.update(task, advance=1)
        else:
            # Upload without progress bar
            for file_path in files_to_upload:
                relative_path = file_path.relative_to(local_path)
                relative_path_str = str(relative_path).replace(
                    "\\", "/"
                )  # Convert to forward slashes

                # Build S3 key, ensuring no double slashes
                if prefix:
                    # Remove trailing slash from prefix if present
                    clean_prefix = prefix.rstrip("/")
                    s3_key = f"{clean_prefix}/{relative_path_str}"
                else:
                    s3_key = relative_path_str

                try:
                    self._upload_file(file_path, bucket, s3_key)
                    uploaded_files += 1
                except Exception as e:
                    self.logger.error(f"Failed to upload {file_path}: {e}")
                    failed_files.append({"file": str(file_path), "error": str(e)})

        return {
            "uploaded_files": uploaded_files,
            "failed_files": failed_files,
            "total_size": total_size,
            "bucket": bucket,
            "prefix": prefix,
        }

    def _ensure_bucket_exists(self, bucket: str) -> None:
        """Ensure bucket exists, create if it doesn't."""
        try:
            self.s3_client.head_bucket(Bucket=bucket)
            self.logger.debug(f"Bucket '{bucket}' exists")
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                # Bucket doesn't exist, create it
                try:
                    if self.endpoint_url and "amazonaws.com" not in (
                        self.endpoint_url or ""
                    ):
                        # For MinIO and other S3-compatible services
                        self.s3_client.create_bucket(Bucket=bucket)
                    else:
                        # For AWS S3, need to specify LocationConstraint for non-us-east-1
                        response = self.s3_client.get_bucket_location(Bucket=bucket)
                        region = response.get("LocationConstraint", "us-east-1")
                        if region and region != "us-east-1":
                            self.s3_client.create_bucket(
                                Bucket=bucket,
                                CreateBucketConfiguration={
                                    "LocationConstraint": region
                                },
                            )
                        else:
                            self.s3_client.create_bucket(Bucket=bucket)
                    self.logger.info(f"Created bucket '{bucket}'")
                except Exception as create_error:
                    self.logger.error(
                        f"Failed to create bucket '{bucket}': {create_error}"
                    )
                    raise
            else:
                # Other error
                self.logger.error(f"Error checking bucket '{bucket}': {e}")
                raise

    def _upload_file(self, file_path: Path, bucket: str, key: str) -> None:
        """Upload a single file to S3."""
        try:
            with open(file_path, "rb") as file_data:
                self.s3_client.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=file_data,
                )
            self.logger.debug(f"Uploaded {file_path} to s3://{bucket}/{key}")
        except NoCredentialsError as e:
            raise ValueError(
                "No credentials found. Please provide access_key_id and secret_access_key "
                "or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables."
            ) from e
        except Exception as e:
            self.logger.error(f"Failed to upload {file_path}: {e}")
            raise

    def test_connection(self) -> bool:
        """Test connection to S3/MinIO."""
        try:
            # Try to list buckets as a connection test
            response = self.s3_client.list_buckets()
            self.logger.info(
                f"Successfully connected to S3. Found {len(response['Buckets'])} buckets."
            )
            return True
        except NoCredentialsError:
            display_error(
                "No credentials found. Please provide credentials via:\n"
                "  - Command line options (--access-key-id, --secret-access-key)\n"
                "  - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)\n"
                "  - AWS credentials file (~/.aws/credentials)"
            )
            return False
        except Exception as e:
            display_error(f"Failed to connect to S3: {e}")
            return False


def parse_s3_url(url: str) -> tuple[str, str]:
    """Parse S3 URL into bucket and prefix.

    Args:
        url: S3 URL in format s3://bucket/prefix or s3://bucket

    Returns:
        Tuple of (bucket, prefix)
    """
    if not url.startswith("s3://"):
        raise ValueError(f"Invalid S3 URL format: {url}. Must start with 's3://'")

    parsed = urlparse(url)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")

    if not bucket:
        raise ValueError(f"No bucket specified in URL: {url}")

    return bucket, prefix
