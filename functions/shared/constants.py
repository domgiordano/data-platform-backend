# Shared constants for the data platform backend
import os

# Data Lake Containers
CONTAINER_RAW = "raw"
CONTAINER_BRONZE = "bronze"
CONTAINER_SILVER = "silver"
CONTAINER_GOLD = "gold"
CONTAINER_AI = "ai"
CONTAINER_SEARCH = "search"

# Folder Paths
RAW_INCOMING = "incoming"
RAW_ARCHIVE = "archive"
BRONZE_TABLES = "tables"
SILVER_TABLES = "tables"
GOLD_AGGREGATES = "aggregates"
GOLD_REPORTS = "reports"
AI_MODELS = "models"
AI_EMBEDDINGS = "embeddings"
SEARCH_INDEXES = "indexes"
SEARCH_DOCUMENTS = "documents"

# Data Quality Thresholds
MIN_COMPLETENESS = 0.95
MIN_VALIDITY = 0.98
MAX_DUPLICATES = 0.01

# Processing Batch Sizes
SPARK_BATCH_SIZE = 10000
API_BATCH_SIZE = 100
EMBEDDING_BATCH_SIZE = 16

# OpenAI Settings
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
OPENAI_GPT_MODEL = "gpt-4"
OPENAI_MAX_TOKENS = 8000
OPENAI_TEMPERATURE = 0.3

# Search Settings
SEARCH_TOP_K = 10
SEARCH_SEMANTIC_CONFIG = "default"

# Supported File Types
SUPPORTED_DOCUMENT_TYPES = [".pdf", ".docx", ".txt", ".md", ".html", ".json"]
SUPPORTED_IMAGE_TYPES = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]

# Schema Versions
BRONZE_SCHEMA_VERSION = "1.0"
SILVER_SCHEMA_VERSION = "1.0"
GOLD_SCHEMA_VERSION = "1.0"

# Status Codes
STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

# Retry Settings
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
RETRY_BACKOFF = 2

# Monitoring
LOG_LEVEL = "INFO"
METRICS_INTERVAL = 60  # seconds

# Azure Storage
STORAGE_ACCOUNT = os.environ.get("STORAGE_ACCOUNT_NAME")