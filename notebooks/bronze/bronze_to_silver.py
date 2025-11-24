"""
Bronze to Silver Transformation
Cleanses and standardizes raw documents
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, udf, current_timestamp, length, split, size
)
from pyspark.sql.types import StringType, ArrayType, StructType, StructField
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from functions.shared.constants import CONTAINER_BRONZE, CONTAINER_SILVER
import re
from typing import List, Dict
import sys

# Initialize Spark
spark = SparkSession.builder \
    .appName("Bronze_to_Silver_Transform") \
    .getOrCreate()

# Configuration
STORAGE_ACCOUNT = spark.conf.get("spark.storage.account")
TEXT_ANALYTICS_ENDPOINT = spark.conf.get("spark.textanalytics.endpoint")
TEXT_ANALYTICS_KEY = spark.conf.get("spark.textanalytics.key")


class TextCleaner:
    """Text cleaning and normalization"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\;\:]', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace("'", "'").replace("'", "'")
        
        return text.strip()
    
    @staticmethod
    def extract_title(content: str, filename: str) -> str:
        """Extract or generate title"""
        # Try to extract first heading or first line
        lines = content.split('\n')
        for line in lines[:5]:
            line = line.strip()
            if line and len(line) < 200:
                return line
        
        # Fallback to filename
        return filename.replace('_', ' ').replace('-', ' ').title()
    
    @staticmethod
    def detect_language(text: str) -> str:
        """Simple language detection (would use Azure Text Analytics in production)"""
        # Simplified - in production use Azure Text Analytics
        # Check for common English words
        english_words = ['the', 'and', 'is', 'to', 'in', 'of', 'a', 'for']
        text_lower = text.lower()
        
        count = sum(1 for word in english_words if f' {word} ' in text_lower)
        return 'en' if count >= 3 else 'unknown'


class TextAnalyzer:
    """Azure Text Analytics integration"""
    
    def __init__(self, endpoint: str, key: str):
        credential = AzureKeyCredential(key)
        self.client = TextAnalyticsClient(endpoint, credential)
    
    def analyze_batch(self, documents: List[Dict]) -> List[Dict]:
        """Analyze batch of documents"""
        results = []
        
        try:
            # Extract entities
            entity_results = self.client.recognize_entities(
                [{"id": d["id"], "text": d["text"][:5000]} for d in documents]
            )
            
            # Extract key phrases
            keyphrase_results = self.client.extract_key_phrases(
                [{"id": d["id"], "text": d["text"][:5000]} for d in documents]
            )
            
            # Sentiment analysis
            sentiment_results = self.client.analyze_sentiment(
                [{"id": d["id"], "text": d["text"][:5000]} for d in documents]
            )
            
            # Combine results
            for doc in documents:
                doc_id = doc["id"]
                
                # Find matching results
                entities = next((r for r in entity_results if r.id == doc_id), None)
                keyphrases = next((r for r in keyphrase_results if r.id == doc_id), None)
                sentiment = next((r for r in sentiment_results if r.id == doc_id), None)
                
                result = {
                    "document_id": doc_id,
                    "entities": [],
                    "key_phrases": [],
                    "sentiment": {}
                }
                
                if entities and not entities.is_error:
                    result["entities"] = [
                        {
                            "text": e.text,
                            "type": e.category,
                            "confidence": e.confidence_score
                        }
                        for e in entities.entities
                    ]
                
                if keyphrases and not keyphrases.is_error:
                    result["key_phrases"] = keyphrases.key_phrases
                
                if sentiment and not sentiment.is_error:
                    result["sentiment"] = {
                        "label": sentiment.sentiment,
                        "score": sentiment.confidence_scores.positive
                    }
                
                results.append(result)
        
        except Exception as e:
            print(f"Error in text analysis: {e}")
            # Return empty results on error
            results = [
                {
                    "document_id": d["id"],
                    "entities": [],
                    "key_phrases": [],
                    "sentiment": {}
                }
                for d in documents
            ]
        
        return results


def calculate_data_quality(row):
    """Calculate data quality metrics"""
    quality_score = 0.0
    issues = []
    
    # Check completeness
    required_fields = ['document_id', 'title', 'content']
    missing = [f for f in required_fields if not row.get(f)]
    
    if not missing:
        quality_score += 0.4
    else:
        issues.append(f"Missing fields: {', '.join(missing)}")
    
    # Check content length
    content = row.get('content', '')
    if len(content) >= 100:
        quality_score += 0.3
    else:
        issues.append("Content too short")
    
    # Check for valid language
    if row.get('language') and row['language'] != 'unknown':
        quality_score += 0.3
    else:
        issues.append("Language not detected")
    
    return {
        "completeness_score": quality_score,
        "validity_score": 1.0 if not issues else 0.5,
        "quality_issues": issues
    }


def process_bronze_to_silver():
    """Main processing function"""
    print("Starting Bronze to Silver transformation...")
    
    # Read bronze data
    bronze_path = f"abfss://{CONTAINER_BRONZE}@{STORAGE_ACCOUNT}.dfs.core.windows.net/tables/"
    bronze_df = spark.read.parquet(bronze_path)
    
    print(f"Loaded {bronze_df.count()} records from bronze layer")
    
    # Initialize text analyzer
    analyzer = TextAnalyzer(TEXT_ANALYTICS_ENDPOINT, TEXT_ANALYTICS_KEY)
    cleaner = TextCleaner()
    
    # Clean and normalize text
    clean_text_udf = udf(cleaner.clean_text, StringType())
    extract_title_udf = udf(cleaner.extract_title, StringType())
    detect_language_udf = udf(cleaner.detect_language, StringType())
    
    silver_df = bronze_df \
        .withColumn("content", clean_text_udf(col("raw_content"))) \
        .withColumn("title", extract_title_udf(col("raw_content"), col("file_name"))) \
        .withColumn("language", detect_language_udf(col("content"))) \
        .withColumn("word_count", size(split(col("content"), " "))) \
        .withColumn("char_count", length(col("content"))) \
        .withColumn("processed_timestamp", current_timestamp())
    
    # Collect documents for batch analysis (process in chunks)
    documents_to_analyze = silver_df.select("document_id", "content").collect()
    
    batch_size = 25  # Azure Text Analytics batch limit
    all_analysis_results = []
    
    for i in range(0, len(documents_to_analyze), batch_size):
        batch = documents_to_analyze[i:i + batch_size]
        docs = [{"id": row.document_id, "text": row.content} for row in batch]
        
        analysis_results = analyzer.analyze_batch(docs)
        all_analysis_results.extend(analysis_results)
    
    # Create analysis results DataFrame
    analysis_schema = StructType([
        StructField("document_id", StringType(), True),
        StructField("entities", ArrayType(StructType([
            StructField("text", StringType()),
            StructField("type", StringType()),
            StructField("confidence", StringType())
        ]))),
        StructField("key_phrases", ArrayType(StringType())),
        StructField("sentiment", StructType([
            StructField("label", StringType()),
            StructField("score", StringType())
        ]))
    ])
    
    analysis_df = spark.createDataFrame(all_analysis_results, schema=analysis_schema)
    
    # Join analysis results with silver data
    silver_df = silver_df.join(analysis_df, "document_id", "left")
    
    # Add data quality metrics
    quality_udf = udf(calculate_data_quality, StructType([
        StructField("completeness_score", StringType()),
        StructField("validity_score", StringType()),
        StructField("quality_issues", ArrayType(StringType()))
    ]))
    
    silver_df = silver_df.withColumn("data_quality", quality_udf(silver_df))
    
    # Write to silver layer
    silver_path = f"abfss://{CONTAINER_SILVER}@{STORAGE_ACCOUNT}.dfs.core.windows.net/tables/"
    
    silver_df.write \
        .mode("overwrite") \
        .partitionBy("language", "processed_timestamp") \
        .parquet(silver_path)
    
    print(f"Successfully wrote {silver_df.count()} records to silver layer")
    
    # Log statistics
    print("\nProcessing Statistics:")
    print(f"- Total documents: {silver_df.count()}")
    print(f"- Languages detected: {silver_df.select('language').distinct().count()}")
    print(f"- Average word count: {silver_df.agg({'word_count': 'avg'}).collect()[0][0]}")
    print(f"- Average entities per doc: {silver_df.select(size('entities').alias('entity_count')).agg({'entity_count': 'avg'}).collect()[0][0]}")


if __name__ == "__main__":
    try:
        process_bronze_to_silver()
        print("\n✅ Bronze to Silver transformation completed successfully")
    except Exception as e:
        print(f"\n❌ Error in Bronze to Silver transformation: {e}")
        sys.exit(1)