"""
Search Index Builder
Builds and updates Azure AI Search indexes from gold layer data
"""
from ast import expr
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    VectorSearchAlgorithmConfiguration
)
from pyspark.sql import SparkSession
from typing import List, Dict
import sys
from functions.shared.constants import CONTAINER_GOLD

# Initialize Spark
spark = SparkSession.builder \
    .appName("Build_Search_Index") \
    .getOrCreate()

# Configuration
STORAGE_ACCOUNT = spark.conf.get("spark.storage.account")
SEARCH_ENDPOINT = spark.conf.get("spark.search.endpoint")
SEARCH_KEY = spark.conf.get("spark.search.key")
INDEX_NAME = "documents-index"


class SearchIndexBuilder:
    """Builds and manages Azure AI Search indexes"""
    
    def __init__(self, endpoint: str, api_key: str):
        self.credential = AzureKeyCredential(api_key)
        self.index_client = SearchIndexClient(endpoint, self.credential)
        self.endpoint = endpoint
    
    def create_index_schema(self) -> SearchIndex:
        """Create the search index schema"""
        
        fields = [
            SimpleField(
                name="document_id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True
            ),
            SearchableField(
                name="title",
                type=SearchFieldDataType.String,
                searchable=True,
                filterable=True,
                sortable=True,
                analyzer_name="en.microsoft"
            ),
            SearchableField(
                name="content",
                type=SearchFieldDataType.String,
                searchable=True,
                analyzer_name="en.microsoft"
            ),
            SearchableField(
                name="executive_summary",
                type=SearchFieldDataType.String,
                searchable=True,
                analyzer_name="en.microsoft"
            ),
            SearchableField(
                name="searchable_content",
                type=SearchFieldDataType.String,
                searchable=True,
                analyzer_name="en.microsoft"
            ),
            SearchableField(
                name="search_keywords",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                searchable=True,
                filterable=True,
                facetable=True
            ),
            SimpleField(
                name="document_type",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,
                sortable=True
            ),
            SimpleField(
                name="priority",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True
            ),
            SimpleField(
                name="relevance_score",
                type=SearchFieldDataType.Double,
                filterable=True,
                sortable=True
            ),
            SearchableField(
                name="main_topics",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                searchable=True,
                filterable=True,
                facetable=True
            ),
            SearchableField(
                name="action_items",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                searchable=True
            ),
            SimpleField(
                name="enrichment_timestamp",
                type=SearchFieldDataType.DateTimeOffset,
                filterable=True,
                sortable=True
            ),
            # Vector field for semantic search
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,  # Ada-002 dimension
                vector_search_configuration="vector-config"
            )
        ]
        
        # Configure vector search
        vector_search = VectorSearch(
            algorithm_configurations=[
                VectorSearchAlgorithmConfiguration(
                    name="vector-config",
                    kind="hnsw",
                    parameters={
                        "m": 4,
                        "efConstruction": 400,
                        "efSearch": 500,
                        "metric": "cosine"
                    }
                )
            ]
        )
        
        index = SearchIndex(
            name=INDEX_NAME,
            fields=fields,
            vector_search=vector_search
        )
        
        return index
    
    def create_or_update_index(self):
        """Create or update the search index"""
        try:
            index = self.create_index_schema()
            result = self.index_client.create_or_update_index(index)
            print(f"Index '{result.name}' created/updated successfully")
            return result
        except Exception as e:
            print(f"Error creating index: {e}")
            raise
    
    def upload_documents(self, documents: List[Dict]):
        """Upload documents to search index"""
        try:
            search_client = SearchClient(
                endpoint=self.endpoint,
                index_name=INDEX_NAME,
                credential=self.credential
            )
            
            # Upload in batches
            batch_size = 1000
            total_uploaded = 0
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                result = search_client.upload_documents(documents=batch)
                
                succeeded = sum(1 for r in result if r.succeeded)
                total_uploaded += succeeded
                
                print(f"Uploaded batch {i // batch_size + 1}: {succeeded}/{len(batch)} documents")
            
            print(f"Total documents uploaded: {total_uploaded}/{len(documents)}")
            return total_uploaded
        
        except Exception as e:
            print(f"Error uploading documents: {e}")
            raise
    
    def delete_index(self):
        """Delete the search index"""
        try:
            self.index_client.delete_index(INDEX_NAME)
            print(f"Index '{INDEX_NAME}' deleted")
        except Exception as e:
            print(f"Error deleting index: {e}")


def transform_gold_to_search_doc(row) -> Dict:
    """Transform gold layer document to search document format"""
    return {
        "document_id": row.document_id,
        "title": row.title,
        "content": row.content[:50000],  # Limit size
        "executive_summary": row.executive_summary or "",
        "searchable_content": row.search_metadata.get("searchable_content", "")[:50000],
        "search_keywords": row.search_metadata.get("search_keywords", [])[:100],
        "document_type": row.business_metadata.get("document_type", "article"),
        "priority": row.business_metadata.get("priority", "medium"),
        "relevance_score": row.business_metadata.get("relevance_score", 0.0),
        "main_topics": [t["topic"] for t in row.ai_insights.get("main_topics", [])[:20]],
        "action_items": row.ai_insights.get("action_items", [])[:20],
        "enrichment_timestamp": row.enrichment_timestamp,
        "content_vector": row.embeddings.get("embedding_vector", [])
    }


def build_search_index():
    """Main function to build search index"""
    print("Starting search index build...")
    
    # Initialize index builder
    builder = SearchIndexBuilder(SEARCH_ENDPOINT, SEARCH_KEY)
    
    # Create or update index
    print("Creating/updating search index schema...")
    builder.create_or_update_index()
    
    # Read gold layer data
    gold_path = f"abfss://{CONTAINER_GOLD}@{STORAGE_ACCOUNT}.dfs.core.windows.net/aggregates/"
    gold_df = spark.read.parquet(gold_path)
    
    print(f"Loaded {gold_df.count()} documents from gold layer")
    
    # Transform to search documents
    documents = []
    for row in gold_df.collect():
        try:
            search_doc = transform_gold_to_search_doc(row)
            documents.append(search_doc)
        except Exception as e:
            print(f"Error transforming document {row.document_id}: {e}")
            continue
    
    print(f"Transformed {len(documents)} documents for indexing")
    
    # Upload to search index
    print("Uploading documents to search index...")
    uploaded_count = builder.upload_documents(documents)
    
    print(f"\n✅ Search index build completed successfully")
    print(f"Total documents indexed: {uploaded_count}")
    
    return uploaded_count


def incremental_update(document_ids: List[str] = None):
    """Incrementally update specific documents in search index"""
    print("Starting incremental search index update...")
    
    builder = SearchIndexBuilder(SEARCH_ENDPOINT, SEARCH_KEY)
    
    # Read gold layer data
    gold_path = f"abfss://{CONTAINER_GOLD}@{STORAGE_ACCOUNT}.dfs.core.windows.net/aggregates/"
    gold_df = spark.read.parquet(gold_path)
    
    # Filter if specific document IDs provided
    if document_ids:
        gold_df = gold_df.filter(gold_df.document_id.isin(document_ids))
    else:
        # Get recently updated documents (last 24 hours)
        from pyspark.sql.functions import col, current_timestamp
        gold_df = gold_df.filter(
            col("enrichment_timestamp") >= current_timestamp() - expr("INTERVAL 1 DAY")
        )
    
    print(f"Updating {gold_df.count()} documents in search index")
    
    # Transform and upload
    documents = [transform_gold_to_search_doc(row) for row in gold_df.collect()]
    uploaded_count = builder.upload_documents(documents)
    
    print(f"✅ Incremental update completed: {uploaded_count} documents updated")
    return uploaded_count


if __name__ == "__main__":
    try:
        # Check if incremental update mode
        if len(sys.argv) > 1 and sys.argv[1] == "--incremental":
            incremental_update()
        else:
            build_search_index()
    except Exception as e:
        print(f"\n❌ Error in search index build: {e}")
        sys.exit(1)