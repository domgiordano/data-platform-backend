"""
Azure Function: Event Handler
Handles events from Event Hub (pipeline completions, errors, etc.)
Triggers downstream processes and sends notifications
"""
import azure.functions as func
import logging
import json
from datetime import datetime
from azure.storage.filedatalake import DataLakeServiceClient
from azure.identity import DefaultAzureCredential
import os

from functions.shared.constants import *
from functions.shared.utils import AzureStorageHelper, MetricsCollector

app = func.FunctionApp()

# Configuration
storage_helper = AzureStorageHelper(STORAGE_ACCOUNT)
metrics = MetricsCollector()

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


def handle_bronze_complete_event(event_data: dict):
    """Handle bronze layer processing completion"""
    document_id = event_data.get("document_id")
    logger.info(f"Bronze processing completed for {document_id}")
    
    # Trigger silver layer processing
    # In production, this would trigger an ADF pipeline or Synapse notebook
    logger.info(f"Triggering silver layer processing for {document_id}")
    
    # Update metrics
    metrics.increment("bronze_documents_completed")


def handle_silver_complete_event(event_data: dict):
    """Handle silver layer processing completion"""
    document_id = event_data.get("document_id")
    logger.info(f"Silver processing completed for {document_id}")
    
    # Trigger gold layer AI enrichment
    logger.info(f"Triggering gold layer AI enrichment for {document_id}")
    
    # Update metrics
    metrics.increment("silver_documents_completed")


def handle_gold_complete_event(event_data: dict):
    """Handle gold layer enrichment completion"""
    document_id = event_data.get("document_id")
    logger.info(f"Gold enrichment completed for {document_id}")
    
    # Trigger search index update
    logger.info(f"Triggering search index update for {document_id}")
    
    # Update metrics
    metrics.increment("gold_documents_completed")


def handle_pipeline_failure_event(event_data: dict):
    """Handle pipeline failure"""
    pipeline_name = event_data.get("pipeline_name")
    error_message = event_data.get("error_message")
    document_id = event_data.get("document_id")
    
    logger.error(f"Pipeline {pipeline_name} failed for {document_id}: {error_message}")
    
    # Update document status
    try:
        # Mark document as failed in appropriate layer
        layer = event_data.get("layer", "unknown")
        
        if layer == "bronze":
            path = f"{BRONZE_TABLES}/{document_id}.json"
            doc = storage_helper.read_json(CONTAINER_BRONZE, path)
            doc["processing_status"] = STATUS_FAILED
            doc["error_message"] = error_message
            storage_helper.write_json(CONTAINER_BRONZE, path, doc)
        
        # Send alert (would integrate with monitoring system)
        logger.warning(f"Alert sent for pipeline failure: {pipeline_name}")
        
    except Exception as e:
        logger.error(f"Error handling pipeline failure: {e}")
    
    # Update metrics
    metrics.increment("pipeline_failures")


def handle_quality_check_event(event_data: dict):
    """Handle data quality check results"""
    document_id = event_data.get("document_id")
    quality_score = event_data.get("quality_score", 0.0)
    issues = event_data.get("issues", [])
    
    logger.info(f"Quality check for {document_id}: score={quality_score}")
    
    if quality_score < MIN_COMPLETENESS:
        logger.warning(f"Low quality score for {document_id}: {quality_score}")
        logger.warning(f"Issues: {', '.join(issues)}")
        
        # Could quarantine low-quality documents
        metrics.increment("quality_check_warnings")
    
    # Update metrics
    metrics.set_gauge("avg_quality_score", quality_score)


@app.event_hub_message_trigger(
    arg_name="events",
    event_hub_name="dataplatform-events",
    connection="EventHubConnection"
)
def event_handler(events: func.EventHubEvent):
    """
    Process events from Event Hub
    Routes events to appropriate handlers
    """
    try:
        # Get event data
        event_body = events.get_body().decode('utf-8')
        event_data = json.loads(event_body)
        
        event_type = event_data.get("event_type")
        timestamp = event_data.get("timestamp", datetime.utcnow().isoformat())
        
        logger.info(f"Received event: {event_type} at {timestamp}")
        
        # Route to appropriate handler
        if event_type == "bronze_processing_complete":
            handle_bronze_complete_event(event_data)
        
        elif event_type == "silver_processing_complete":
            handle_silver_complete_event(event_data)
        
        elif event_type == "gold_enrichment_complete":
            handle_gold_complete_event(event_data)
        
        elif event_type == "pipeline_failure":
            handle_pipeline_failure_event(event_data)
        
        elif event_type == "quality_check_complete":
            handle_quality_check_event(event_data)
        
        else:
            logger.warning(f"Unknown event type: {event_type}")
            metrics.increment("unknown_events")
        
        # Log event processing
        logger.info(json.dumps({
            "event": "event_processed",
            "event_type": event_type,
            "timestamp": timestamp,
            "status": "success"
        }))
        
        # Log metrics periodically
        if metrics.metrics.get("events_processed", 0) % 100 == 0:
            logger.info(f"Metrics: {json.dumps(metrics.get_metrics())}")
    
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in event: {e}")
        metrics.increment("invalid_events")
    
    except Exception as e:
        logger.error(f"Error processing event: {e}", exc_info=True)
        metrics.increment("event_processing_errors")


@app.timer_trigger(
    arg_name="timer",
    schedule="0 */15 * * * *"  # Every 15 minutes
)
def metrics_reporter(timer: func.TimerRequest):
    """
    Periodic metrics reporting
    Reports accumulated metrics and resets counters
    """
    try:
        current_metrics = metrics.get_metrics()
        
        if current_metrics:
            logger.info("=== Metrics Report ===")
            for metric_name, value in current_metrics.items():
                logger.info(f"{metric_name}: {value}")
            
            # In production, send to monitoring system (App Insights, etc.)
            logger.info(json.dumps({
                "event": "metrics_report",
                "metrics": current_metrics,
                "timestamp": datetime.utcnow().isoformat()
            }))
            
            # Reset counters (but keep gauges)
            metrics.reset()
        
        logger.info("Metrics report completed")
    
    except Exception as e:
        logger.error(f"Error reporting metrics: {e}", exc_info=True)


@app.function_name("health_check")
@app.route(route="health", methods=["GET"])
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """Health check endpoint"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics.get_metrics()
        }
        
        return func.HttpResponse(
            json.dumps(health_status),
            mimetype="application/json",
            status_code=200
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return func.HttpResponse(
            json.dumps({"status": "unhealthy", "error": str(e)}),
            mimetype="application/json",
            status_code=500
        )