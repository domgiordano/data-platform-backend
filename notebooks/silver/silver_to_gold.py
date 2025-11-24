"""
Silver to Gold Transformation with AI Enrichment
Adds business insights, embeddings, and prepares for search
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp
from functions.shared.constants import OPENAI_GPT_MODEL, OPENAI_EMBEDDING_MODEL, CONTAINER_AI, CONTAINER_GOLD, CONTAINER_SILVER, CONTAINER_BRONZE
import sys
import openai
from typing import List, Dict
import json

# Initialize Spark
spark = SparkSession.builder \
    .appName("Silver_to_Gold_AI_Enrichment") \
    .getOrCreate()

# Configuration
STORAGE_ACCOUNT = spark.conf.get("spark.storage.account")
OPENAI_ENDPOINT = spark.conf.get("spark.openai.endpoint")
OPENAI_KEY = spark.conf.get("spark.openai.key")


class AIEnricher:
    """AI-powered document enrichment using Azure OpenAI"""
    
    def __init__(self, endpoint: str, api_key: str):
        openai.api_type = "azure"
        openai.api_base = endpoint
        openai.api_key = api_key
        openai.api_version = "2023-05-15"
    
    def generate_executive_summary(self, content: str, max_length: int = 500) -> str:
        """Generate executive summary using GPT-4"""
        try:
            # Truncate content if too long
            content = content[:4000] if len(content) > 4000 else content
            
            prompt = f"""Generate a concise executive summary of the following document. 
            Focus on key points, main findings, and actionable insights.
            Keep it under {max_length} characters.
            
            Document:
            {content}
            
            Executive Summary:"""
            
            response = openai.ChatCompletion.create(
                engine=OPENAI_GPT_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert at summarizing documents concisely."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Error generating summary: {e}")
            return ""
    
    def extract_topics(self, content: str, key_phrases: List[str]) -> List[Dict]:
        """Extract main topics using AI"""
        try:
            content = content[:3000] if len(content) > 3000 else content
            
            prompt = f"""Analyze this document and identify the top 5 main topics or themes.
            For each topic, provide a relevance score from 0 to 1.
            
            Document: {content}
            
            Key phrases: {', '.join(key_phrases[:10])}
            
            Return ONLY a JSON array in this format:
            [
              {{"topic": "Topic Name", "relevance": 0.95}},
              ...
            ]
            """
            
            response = openai.ChatCompletion.create(
                engine=OPENAI_GPT_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert at topic extraction. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.2
            )
            
            result = response.choices[0].message.content.strip()
            # Parse JSON response
            topics = json.loads(result)
            return topics if isinstance(topics, list) else []
        
        except Exception as e:
            print(f"Error extracting topics: {e}")
            return []
    
    def extract_action_items(self, content: str) -> List[str]:
        """Extract action items or recommendations"""
        try:
            content = content[:3000] if len(content) > 3000 else content
            
            prompt = f"""Extract any action items, recommendations, or next steps from this document.
            Return as a simple list, one item per line. If none found, return an empty list.
            
            Document: {content}
            
            Action Items:"""
            
            response = openai.ChatCompletion.create(
                engine=OPENAI_GPT_MODEL,
                messages=[
                    {"role": "system", "content": "You extract actionable items from documents."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            result = response.choices[0].message.content.strip()
            # Split by newlines and clean up
            items = [line.strip().lstrip('- ').lstrip('* ') 
                    for line in result.split('\n') 
                    if line.strip()]
            return items[:10]  # Limit to 10 items
        
        except Exception as e:
            print(f"Error extracting action items: {e}")
            return []
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for semantic search"""
        try:
            # Truncate text if needed
            text = text[:8000] if len(text) > 8000 else text
            
            response = openai.Embedding.create(
                engine=OPENAI_EMBEDDING_MODEL,
                input=text
            )
            
            embedding = response.data[0].embedding
            return embedding
        
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []
    
    def classify_document_type(self, title: str, content: str) -> str:
        """Classify document type"""
        try:
            content_sample = content[:1000] if len(content) > 1000 else content
            
            prompt = f"""Classify this document into ONE of these categories:
            - report
            - article
            - whitepaper
            - case_study
            - blog
            - documentation
            
            Title: {title}
            Content: {content_sample}
            
            Return ONLY the category name, nothing else."""
            
            response = openai.ChatCompletion.create(
                engine=OPENAI_GPT_MODEL,
                messages=[
                    {"role": "system", "content": "You classify documents accurately."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.1
            )
            
            doc_type = response.choices[0].message.content.strip().lower()
            valid_types = ["report", "article", "whitepaper", "case_study", "blog", "documentation"]
            
            return doc_type if doc_type in valid_types else "article"
        
        except Exception as e:
            print(f"Error classifying document: {e}")
            return "article"


def calculate_relevance_score(word_count: int, entity_count: int, keyphrase_count: int) -> float:
    """Calculate relevance score based on content metrics"""
    # Simple scoring algorithm
    score = 0.0
    
    # Word count contribution (normalize to 0-1)
    if word_count >= 1000:
        score += 0.4
    elif word_count >= 500:
        score += 0.3
    elif word_count >= 200:
        score += 0.2
    else:
        score += 0.1
    
    # Entity count contribution
    if entity_count >= 10:
        score += 0.3
    elif entity_count >= 5:
        score += 0.2
    else:
        score += 0.1
    
    # Key phrase contribution
    if keyphrase_count >= 10:
        score += 0.3
    elif keyphrase_count >= 5:
        score += 0.2
    else:
        score += 0.1
    
    return min(score, 1.0)


def optimize_for_search(content: str, title: str, key_phrases: List[str]) -> Dict:
    """Optimize content for search indexing"""
    # Create searchable content with boosted terms
    searchable = f"{title} {title} {' '.join(key_phrases)} {content}"
    
    # Extract search keywords (unique key phrases + title words)
    keywords = list(set(key_phrases + title.lower().split()))
    keywords = [k for k in keywords if len(k) > 2][:50]  # Top 50 keywords
    
    return {
        "searchable_content": searchable[:10000],  # Limit size
        "search_keywords": keywords,
        "search_score_boost": 1.0
    }


def process_silver_to_gold():
    """Main processing function"""
    print("Starting Silver to Gold transformation with AI enrichment...")
    
    # Read silver data
    silver_path = f"abfss://{CONTAINER_SILVER}@{STORAGE_ACCOUNT}.dfs.core.windows.net/tables/"
    silver_df = spark.read.parquet(silver_path)
    
    print(f"Loaded {silver_df.count()} records from silver layer")
    
    # Initialize AI enricher
    enricher = AIEnricher(OPENAI_ENDPOINT, OPENAI_KEY)
    
    # Collect documents for enrichment (process in batches)
    documents = silver_df.collect()
    enriched_data = []
    
    total = len(documents)
    for idx, doc in enumerate(documents):
        print(f"Processing document {idx + 1}/{total}: {doc.document_id}")
        
        try:
            # Generate AI enrichments
            executive_summary = enricher.generate_executive_summary(doc.content)
            topics = enricher.extract_topics(doc.content, doc.key_phrases or [])
            action_items = enricher.extract_action_items(doc.content)
            embedding = enricher.generate_embedding(doc.content)
            doc_type = enricher.classify_document_type(doc.title, doc.content)
            
            # Calculate metrics
            entity_count = len(doc.entities) if doc.entities else 0
            keyphrase_count = len(doc.key_phrases) if doc.key_phrases else 0
            relevance_score = calculate_relevance_score(
                doc.word_count,
                entity_count,
                keyphrase_count
            )
            
            # Optimize for search
            search_metadata = optimize_for_search(
                doc.content,
                doc.title,
                doc.key_phrases or []
            )
            
            # Build enriched record
            enriched = {
                "document_id": doc.document_id,
                "title": doc.title,
                "content": doc.content,
                "executive_summary": executive_summary,
                "business_metadata": {
                    "document_type": doc_type,
                    "relevance_score": relevance_score,
                    "priority": "high" if relevance_score > 0.8 else "medium" if relevance_score > 0.5 else "low"
                },
                "ai_insights": {
                    "main_topics": topics,
                    "action_items": action_items,
                    "related_concepts": doc.key_phrases[:20] if doc.key_phrases else []
                },
                "embeddings": {
                    "embedding_model": "text-embedding-ada-002",
                    "embedding_vector": embedding,
                    "embedding_dimension": len(embedding)
                },
                "search_metadata": search_metadata,
                "analytics": {
                    "view_count": 0,
                    "search_appearances": 0,
                    "avg_relevance_score": 0.0,
                    "last_accessed": None
                },
                "source_lineage": {
                    "bronze_path": f"{CONTAINER_BRONZE}/tables/{doc.document_id}",
                    "silver_path": f"{CONTAINER_SILVER}/tables/{doc.document_id}",
                    "gold_path": f"{CONTAINER_GOLD}/aggregates/{doc.document_id}",
                    "processing_pipeline": "bronze-silver-gold"
                },
                "enrichment_timestamp": str(current_timestamp()),
                "schema_version": "1.0"
            }
            
            enriched_data.append(enriched)
            
            # Save embedding separately for vector search
            embedding_record = {
                "document_id": doc.document_id,
                "embedding": embedding,
                "model": "text-embedding-ada-002",
                "dimension": len(embedding),
                "created_at": str(current_timestamp())
            }
            
            # Write embedding to AI container
            embedding_path = f"abfss://{CONTAINER_AI}@{STORAGE_ACCOUNT}.dfs.core.windows.net/embeddings/{doc.document_id}.json"
            # In production, would use proper file write
            
        except Exception as e:
            print(f"Error processing document {doc.document_id}: {e}")
            continue
    
    # Create DataFrame from enriched data
    gold_df = spark.createDataFrame(enriched_data)
    
    # Write to gold layer
    gold_path = f"abfss://{CONTAINER_GOLD}@{STORAGE_ACCOUNT}.dfs.core.windows.net/aggregates/"
    
    gold_df.write \
        .mode("overwrite") \
        .partitionBy("business_metadata.document_type") \
        .parquet(gold_path)
    
    print(f"Successfully wrote {gold_df.count()} records to gold layer")
    
    # Log statistics
    print("\nEnrichment Statistics:")
    print(f"- Total documents enriched: {len(enriched_data)}")
    print(f"- Documents with embeddings: {sum(1 for d in enriched_data if d['embeddings']['embedding_vector'])}")
    print(f"- Avg topics per document: {sum(len(d['ai_insights']['main_topics']) for d in enriched_data) / len(enriched_data):.2f}")
    print(f"- Avg action items: {sum(len(d['ai_insights']['action_items']) for d in enriched_data) / len(enriched_data):.2f}")


if __name__ == "__main__":
    try:
        process_silver_to_gold()
        print("\n✅ Silver to Gold transformation completed successfully")
    except Exception as e:
        print(f"\n❌ Error in Silver to Gold transformation: {e}")
        sys.exit(1)