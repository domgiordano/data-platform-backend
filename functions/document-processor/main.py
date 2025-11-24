"""
Azure Function: Document Processor
Triggered when new documents are uploaded to raw container
Performs initial processing and moves to bronze layer
"""
import azure.functions as func
import logging
import json
import hashlib
from pypdf import PdfReader
from docx import Document
import io
from datetime import datetime
import os

from functions.shared.constants import *
from functions.shared.utils import AzureStorageHelper, validate_schema

app = func.FunctionApp()

# Configuration
storage_helper = AzureStorageHelper(STORAGE_ACCOUNT)

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


def extract_text_from_file(file_path: str, file_type: str, content: bytes) -> str:
    """Extract text content from various file types"""
    try:
        if file_type in ['.txt', '.md']:
            return content.decode('utf-8')
        
        elif file_type == '.pdf':
            # Use pypdf or azure form recognizer in production
            try:
                reader = PdfReader(io.BytesIO(content))
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
            except ImportError:
                logger.warning("pypdf not available, returning raw content")
                return content.decode('utf-8', errors='ignore')
        
        elif file_type == '.docx':
            # Use python-docx in production
            try:
                doc = Document(io.BytesIO(content))
                text = "\n".join([para.text for para in doc.paragraphs])
                return text
            except ImportError:
                logger.warning("python-docx not available")
                return content.decode('utf-8', errors='ignore')
        
        elif file_type in ['.html', '.htm']:
            # Use BeautifulSoup for HTML parsing in production
            return content.decode('utf-8', errors='ignore')
        
        elif file_type == '.json':
            data = json.loads(content.decode('utf-8'))
            # Extract text fields from JSON
            return json.dumps(data, indent=2)
        
        else:
            logger.warning(f"Unsupported file type: {file_type}")
            return content.decode('utf-8', errors='ignore')
    
    except Exception as e:
        logger.error(f"Error extracting text from {file_type}: {e}")
        return ""


def generate_document_id(file_name: str, content: bytes) -> str:
    """Generate unique document ID based on content hash"""
    content_hash = hashlib.md5(content).hexdigest()
    safe_name = file_name.replace(' ', '_').replace('/', '_')
    return f"{safe_name}_{content_hash[:8]}"


def create_bronze_record(
    file_name: str,
    file_path: str,
    content: bytes,
    source_system: str = "blob_trigger"
) -> dict:
    """Create bronze layer record"""
    
    # Get file metadata
    file_type = os.path.splitext(file_name)[1].lower().replace('.', '')
    file_size = len(content)
    
    # Extract text content
    raw_content = extract_text_from_file(file_path, f".{file_type}", content)
    
    # Generate document ID
    document_id = generate_document_id(file_name, content)
    
    # Generate checksum
    checksum = hashlib.md5(raw_content.encode('utf-8')).hexdigest()
    
    # Create bronze record
    bronze_record = {
        "document_id": document_id,
        "source_system": source_system,
        "source_url": None,
        "file_name": file_name,
        "file_type": file_type,
        "file_size_bytes": file_size,
        "ingestion_timestamp": datetime.utcnow().isoformat() + "Z",
        "raw_content": raw_content,
        "metadata": {
            "created_date": datetime.utcnow().isoformat() + "Z",
            "tags": []
        },
        "checksum": checksum,
        "processing_status": STATUS_PENDING,
        "schema_version": BRONZE_SCHEMA_VERSION
    }
    
    return bronze_record


@app.blob_trigger(
    arg_name="myblob",
    path="raw/incoming/{name}",
    connection="AzureWebJobsStorage"
)
def document_processor(myblob: func.InputStream):
    """
    Triggered when a new document is uploaded to raw/incoming
    Processes and moves to bronze layer
    """
    logger.info(f"Processing blob: {myblob.name}")
    logger.info(f"Blob size: {myblob.length} bytes")
    
    try:
        # Read blob content
        content = myblob.read()
        file_name = os.path.basename(myblob.name)
        
        # Check file type
        file_ext = os.path.splitext(file_name)[1].lower()
        if file_ext not in SUPPORTED_DOCUMENT_TYPES:
            logger.warning(f"Unsupported file type: {file_ext}")
            # Move to failed folder
            storage_helper.move_file(
                CONTAINER_RAW,
                myblob.name,
                f"{RAW_ARCHIVE}/failed/{file_name}"
            )
            return
        
        # Create bronze record
        bronze_record = create_bronze_record(file_name, myblob.name, content)
        
        # Validate against schema
        # Load schema (in production, cache this)
        schema_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            '..',
            'schemas',
            'bronze',
            'document.json'
        )
        
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        
        if not validate_schema(bronze_record, schema):
            logger.error(f"Bronze record validation failed for {file_name}")
            # Move to failed
            storage_helper.move_file(
                CONTAINER_RAW,
                myblob.name,
                f"{RAW_ARCHIVE}/failed/{file_name}"
            )
            return
        
        # Write to bronze layer
        bronze_path = f"{BRONZE_TABLES}/{bronze_record['document_id']}.json"
        storage_helper.write_json(CONTAINER_BRONZE, bronze_path, bronze_record)
        
        logger.info(f"Successfully processed {file_name} -> {bronze_record['document_id']}")
        
        # Move original file to archive
        archive_path = f"{RAW_ARCHIVE}/{datetime.utcnow().strftime('%Y/%m/%d')}/{file_name}"
        storage_helper.move_file(CONTAINER_RAW, myblob.name, archive_path)
        
        logger.info(f"Archived original file to {archive_path}")
        
        # Log metrics
        logger.info(json.dumps({
            "event": "document_processed",
            "document_id": bronze_record['document_id'],
            "file_name": file_name,
            "file_type": bronze_record['file_type'],
            "file_size": bronze_record['file_size_bytes'],
            "status": "success"
        }))
    
    except Exception as e:
        logger.error(f"Error processing {myblob.name}: {e}", exc_info=True)
        
        # Log error metrics
        logger.error(json.dumps({
            "event": "document_processing_failed",
            "file_name": myblob.name,
            "error": str(e),
            "status": "failed"
        }))
        
        # Move to failed folder
        try:
            file_name = os.path.basename(myblob.name)
            storage_helper.move_file(
                CONTAINER_RAW,
                myblob.name,
                f"{RAW_ARCHIVE}/failed/{file_name}"
            )
        except Exception as move_error:
            logger.error(f"Failed to move file to failed folder: {move_error}")