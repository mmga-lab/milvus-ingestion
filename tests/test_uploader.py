"""Tests for S3/MinIO upload functionality."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError, NoCredentialsError

from milvus_fake_data.uploader import S3Uploader, parse_s3_url


class TestParseS3Url:
    """Test S3 URL parsing."""

    def test_parse_valid_url_with_prefix(self):
        """Test parsing S3 URL with bucket and prefix."""
        bucket, prefix = parse_s3_url("s3://my-bucket/path/to/data")
        assert bucket == "my-bucket"
        assert prefix == "path/to/data"

    def test_parse_valid_url_without_prefix(self):
        """Test parsing S3 URL with bucket only."""
        bucket, prefix = parse_s3_url("s3://my-bucket")
        assert bucket == "my-bucket"
        assert prefix == ""

    def test_parse_valid_url_with_trailing_slash(self):
        """Test parsing S3 URL with trailing slash."""
        bucket, prefix = parse_s3_url("s3://my-bucket/path/")
        assert bucket == "my-bucket"
        assert prefix == "path/"

    def test_parse_invalid_url_format(self):
        """Test parsing invalid URL format."""
        with pytest.raises(ValueError, match="Invalid S3 URL format"):
            parse_s3_url("http://my-bucket/path")

    def test_parse_url_without_bucket(self):
        """Test parsing URL without bucket."""
        with pytest.raises(ValueError, match="No bucket specified"):
            parse_s3_url("s3://")


class TestS3Uploader:
    """Test S3Uploader class."""

    @patch("boto3.client")
    def test_init_with_credentials(self, mock_boto_client):
        """Test initialization with provided credentials."""
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client

        uploader = S3Uploader(
            endpoint_url="http://localhost:9000",
            access_key_id="test_key",
            secret_access_key="test_secret",
            region_name="us-west-2",
            verify_ssl=False,
        )

        mock_boto_client.assert_called_once_with(
            "s3",
            endpoint_url="http://localhost:9000",
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
            region_name="us-west-2",
            verify=False,
        )
        assert uploader.s3_client == mock_client
        assert uploader.endpoint_url == "http://localhost:9000"

    @patch("boto3.client")
    def test_init_with_env_credentials(self, mock_boto_client):
        """Test initialization with environment variables."""
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client

        with patch.dict(
            os.environ,
            {
                "AWS_ACCESS_KEY_ID": "env_key",
                "AWS_SECRET_ACCESS_KEY": "env_secret",
            },
        ):
            uploader = S3Uploader()

        mock_boto_client.assert_called_once_with(
            "s3",
            endpoint_url=None,
            aws_access_key_id="env_key",
            aws_secret_access_key="env_secret",
            region_name="us-east-1",
            verify=True,
        )

    @patch("boto3.client")
    def test_test_connection_success(self, mock_boto_client):
        """Test successful connection test."""
        mock_client = MagicMock()
        mock_client.list_buckets.return_value = {"Buckets": [{"Name": "bucket1"}]}
        mock_boto_client.return_value = mock_client

        uploader = S3Uploader()
        result = uploader.test_connection()

        assert result is True
        mock_client.list_buckets.assert_called_once()

    @patch("boto3.client")
    def test_test_connection_no_credentials(self, mock_boto_client):
        """Test connection test with no credentials."""
        mock_client = MagicMock()
        mock_client.list_buckets.side_effect = NoCredentialsError()
        mock_boto_client.return_value = mock_client

        uploader = S3Uploader()
        result = uploader.test_connection()

        assert result is False

    @patch("boto3.client")
    def test_test_connection_error(self, mock_boto_client):
        """Test connection test with other errors."""
        mock_client = MagicMock()
        mock_client.list_buckets.side_effect = Exception("Connection error")
        mock_boto_client.return_value = mock_client

        uploader = S3Uploader()
        result = uploader.test_connection()

        assert result is False

    @patch("boto3.client")
    def test_ensure_bucket_exists_already_exists(self, mock_boto_client):
        """Test ensuring bucket that already exists."""
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client

        uploader = S3Uploader()
        uploader._ensure_bucket_exists("test-bucket")

        mock_client.head_bucket.assert_called_once_with(Bucket="test-bucket")
        mock_client.create_bucket.assert_not_called()

    @patch("boto3.client")
    def test_ensure_bucket_exists_create_new(self, mock_boto_client):
        """Test creating new bucket."""
        mock_client = MagicMock()
        # Simulate bucket not found error
        error_response = {"Error": {"Code": "404"}}
        mock_client.head_bucket.side_effect = ClientError(error_response, "HeadBucket")
        mock_boto_client.return_value = mock_client

        uploader = S3Uploader(endpoint_url="http://localhost:9000")
        uploader._ensure_bucket_exists("test-bucket")

        mock_client.head_bucket.assert_called_once_with(Bucket="test-bucket")
        mock_client.create_bucket.assert_called_once_with(Bucket="test-bucket")

    @patch("boto3.client")
    def test_upload_file_success(self, mock_boto_client):
        """Test successful file upload."""
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client

        uploader = S3Uploader()

        # Create a temporary file
        with patch("builtins.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file

            uploader._upload_file(Path("/tmp/test.txt"), "bucket", "key")

        mock_client.put_object.assert_called_once_with(
            Bucket="bucket",
            Key="key",
            Body=mock_file,
        )

    @patch("boto3.client")
    def test_upload_directory_empty_directory(self, mock_boto_client, tmp_path):
        """Test uploading empty directory."""
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client

        uploader = S3Uploader()
        result = uploader.upload_directory(tmp_path, "test-bucket", "prefix/")

        assert result["uploaded_files"] == 0
        assert result["failed_files"] == []
        assert result["total_size"] == 0

    @patch("boto3.client")
    def test_upload_directory_with_files(self, mock_boto_client, tmp_path):
        """Test uploading directory with files."""
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client

        # Create test files
        (tmp_path / "file1.txt").write_text("content1")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "file2.txt").write_text("content2")

        uploader = S3Uploader()
        result = uploader.upload_directory(
            tmp_path, "test-bucket", "prefix", show_progress=False
        )

        assert result["uploaded_files"] == 2
        assert result["failed_files"] == []
        assert result["bucket"] == "test-bucket"
        assert result["prefix"] == "prefix"

        # Verify put_object was called correctly
        assert mock_client.put_object.call_count == 2

    @patch("boto3.client")
    def test_upload_directory_with_failures(self, mock_boto_client, tmp_path):
        """Test uploading directory with some failures."""
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client

        # Create test files
        (tmp_path / "file1.txt").write_text("content1")
        (tmp_path / "file2.txt").write_text("content2")

        # Make the second upload fail
        mock_client.put_object.side_effect = [None, Exception("Upload failed")]

        uploader = S3Uploader()
        result = uploader.upload_directory(
            tmp_path, "test-bucket", "prefix", show_progress=False
        )

        assert result["uploaded_files"] == 1
        assert len(result["failed_files"]) == 1
        assert result["failed_files"][0]["error"] == "Upload failed"

    def test_upload_directory_nonexistent_path(self):
        """Test uploading non-existent directory."""
        uploader = S3Uploader()

        with pytest.raises(FileNotFoundError):
            uploader.upload_directory(Path("/nonexistent"), "bucket", "prefix")

    def test_upload_directory_file_path(self, tmp_path):
        """Test uploading file instead of directory."""
        # Create a file
        file_path = tmp_path / "file.txt"
        file_path.write_text("content")

        uploader = S3Uploader()

        with pytest.raises(ValueError, match="Path is not a directory"):
            uploader.upload_directory(file_path, "bucket", "prefix")
