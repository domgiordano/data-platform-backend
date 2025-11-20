# data-platform-backend

Data processing backend for Azure Data Platform.
[Project Documentation Repo](https://github.com/domgiordano/data-platform-meta/tree/master)

## Overview

Contains all data processing components:
- Synapse notebooks for ETL/ELT
- ADF pipeline definitions
- Azure Functions for event processing
- AI enrichment & search indexing
- Data quality & validation

## Components

### Notebooks
- **Bronze Layer**: Raw data validation & extraction
- **Silver Layer**: Cleansing & standardization
- **Gold Layer**: Business aggregations
- **AI Processing**: Embeddings & enrichment

### Pipelines
- Document ingestion from web sources
- Batch processing orchestration
- Real-time event handling

### Functions
- Document parsing triggers
- Event Hub processors
- API endpoints

## Setup

### Prerequisites
```bash
python >= 3.9
Azure CLI
```

### Local Development
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Testing
```bash
pytest tests/unit
pytest tests/integration --env=dev
```

## Deployment

GitHub Actions automatically deploys on push:
- `develop` → Dev environment
- `staging` → Staging environment
- `main` → Production (manual approval)

### Manual Deployment
```bash
# Deploy notebooks
python scripts/deploy_notebooks.py --env dev

# Deploy pipelines
python scripts/deploy_pipelines.py --env dev

# Deploy functions
python scripts/deploy_functions.py --env dev
```

## Infrastructure Dependencies

Fetches infrastructure details from Terraform Cloud:
- Synapse workspace for notebooks
- ADF for pipelines
- Storage accounts for data
- AI services endpoints

## Data Flow

```
Raw → Bronze → Silver → Gold
         ↓        ↓
   AI Enrichment  Search Index
```
