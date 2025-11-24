"""
Shared utility functions for data platform backend
"""
import logging
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from azure.identity import DefaultAzureCredential
from azure.storage.filedatalake import DataLakeServiceClient
from azure.keyvault.secrets import SecretClient
from azure.core.exceptions import ResourceNotFoundError
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from .constants import MAX_RETRIES, RETRY_DELAY, RETRY_BACKOFF

logger = logging.getLogger(__name__)


class AzureStorageHelper:
    """Helper class for Azure Data Lake operations"""
    
    def __init__(self, account_name: str):
        self.account_name = account_name
        self.credential = DefaultAzureCredential()
        self.service_client = DataLakeServiceClient(
            account_url=f"https://{account_name}.dfs.core.windows.net",
            credential=self.credential
        )
    
    def read_json(self, container: str, path: str) -> Dict:
        """Read JSON file from Data Lake"""
        try:
            file_system_client = self.service_client.get_file_system_client(container)
            file_client = file_system_client.get_file_client(path)
            
            download = file_client.download_file()
            content = download.readall()
            return json.loads(content)
        except Exception as e:
            logger.error(f"Error reading JSON from {container}/{path}: {e}")
            raise
    
    def write_json(self, container: str, path: str, data: Dict):
        """Write JSON file to Data Lake"""
        try:
            file_system_client = self.service_client.get_file_system_client(container)
            file_client = file_system_client.get_file_client(path)
            
            content = json.dumps(data, indent=2)
            file_client.upload_data(content, overwrite=True)
            logger.info(f"Successfully wrote JSON to {container}/{path}")
        except Exception as e:
            logger.error(f"Error writing JSON to {container}/{path}: {e}")
            raise
    
    def read_parquet(self, container: str, path: str) -> pd.DataFrame:
        """Read Parquet file from Data Lake"""
        try:
            file_system_client = self.service_client.get_file_system_client(container)
            file_client = file_system_client.get_file_client(path)
            
            download = file_client.download_file()
            content = download.readall()
            
            import io
            return pd.read_parquet(io.BytesIO(content))
        except Exception as e:
            logger.error(f"Error reading Parquet from {container}/{path}: {e}")
            raise
    
    def write_parquet(self, container: str, path: str, df: pd.DataFrame):
        """Write Parquet file to Data Lake"""
        try:
            file_system_client = self.service_client.get_file_system_client(container)
            file_client = file_system_client.get_file_client(path)
            
            import io
            buffer = io.BytesIO()
            df.to_parquet(buffer, index=False)
            buffer.seek(0)
            
            file_client.upload_data(buffer.read(), overwrite=True)
            logger.info(f"Successfully wrote Parquet to {container}/{path}")
        except Exception as e:
            logger.error(f"Error writing Parquet to {container}/{path}: {e}")
            raise
    
    def list_paths(self, container: str, path: str) -> List[str]:
        """List all paths in a directory"""
        try:
            file_system_client = self.service_client.get_file_system_client(container)
            paths = file_system_client.get_paths(path=path)
            return [p.name for p in paths]
        except Exception as e:
            logger.error(f"Error listing paths in {container}/{path}: {e}")
            raise
    
    def move_file(self, container: str, source: str, destination: str):
        """Move file within Data Lake"""
        try:
            file_system_client = self.service_client.get_file_system_client(container)
            source_client = file_system_client.get_file_client(source)
            
            # Copy to destination
            dest_client = file_system_client.get_file_client(destination)
            dest_client.upload_data(source_client.download_file().readall(), overwrite=True)
            
            # Delete source
            source_client.delete_file()
            logger.info(f"Moved {source} to {destination}")
        except Exception as e:
            logger.error(f"Error moving file: {e}")
            raise


class KeyVaultHelper:
    """Helper class for Azure Key Vault operations"""
    
    def __init__(self, vault_url: str):
        self.credential = DefaultAzureCredential()
        self.client = SecretClient(vault_url=vault_url, credential=self.credential)
    
    def get_secret(self, secret_name: str) -> str:
        """Get secret from Key Vault"""
        try:
            secret = self.client.get_secret(secret_name)
            return secret.value
        except ResourceNotFoundError:
            logger.error(f"Secret {secret_name} not found in Key Vault")
            raise
        except Exception as e:
            logger.error(f"Error retrieving secret {secret_name}: {e}")
            raise


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=RETRY_DELAY, max=60)
)
def retry_with_backoff(func):
    """Decorator for retrying functions with exponential backoff"""
    return func


def validate_schema(data: Dict, schema: Dict) -> bool:
    """Validate data against JSON schema"""
    from jsonschema import validate, ValidationError
    
    try:
        validate(instance=data, schema=schema)
        return True
    except ValidationError as e:
        logger.error(f"Schema validation error: {e}")
        return False


def calculate_data_quality_score(
    completeness: float,
    validity: float,
    uniqueness: float
) -> float:
    """Calculate overall data quality score"""
    weights = {
        "completeness": 0.4,
        "validity": 0.4,
        "uniqueness": 0.2
    }
    
    score = (
        completeness * weights["completeness"] +
        validity * weights["validity"] +
        uniqueness * weights["uniqueness"]
    )
    
    return round(score, 4)


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """Split list into chunks"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def sanitize_column_name(name: str) -> str:
    """Sanitize column name for Spark/Parquet compatibility"""
    # Replace spaces and special characters
    sanitized = name.strip().lower()
    sanitized = sanitized.replace(" ", "_")
    sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in sanitized)
    
    # Remove consecutive underscores
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    
    return sanitized.strip("_")


def generate_partition_path(timestamp: datetime, partition_by: str = "date") -> str:
    """Generate partition path based on timestamp"""
    if partition_by == "date":
        return f"year={timestamp.year}/month={timestamp.month:02d}/day={timestamp.day:02d}"
    elif partition_by == "hour":
        return f"year={timestamp.year}/month={timestamp.month:02d}/day={timestamp.day:02d}/hour={timestamp.hour:02d}"
    elif partition_by == "month":
        return f"year={timestamp.year}/month={timestamp.month:02d}"
    else:
        raise ValueError(f"Invalid partition_by value: {partition_by}")


def get_file_extension(filename: str) -> str:
    """Get file extension"""
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def format_bytes(size: int) -> str:
    """Format bytes to human-readable format"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


class MetricsCollector:
    """Simple metrics collector for monitoring"""
    
    def __init__(self):
        self.metrics = {}
    
    def increment(self, metric_name: str, value: int = 1):
        """Increment a counter metric"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = 0
        self.metrics[metric_name] += value
    
    def set_gauge(self, metric_name: str, value: float):
        """Set a gauge metric"""
        self.metrics[metric_name] = value
    
    def get_metrics(self) -> Dict:
        """Get all metrics"""
        return self.metrics.copy()
    
    def reset(self):
        """Reset all metrics"""
        self.metrics = {}