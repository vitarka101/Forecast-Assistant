# Retail Router-Executor Forecasting Service

This project is a FastAPI-based retail forecasting system built around a router-executor pattern:

- `stock_code` is the product identifier
- `customer_id` is the customer identifier
- each entity is assigned to a cluster
- each cluster owns a forecasting model
- the API exposes direct forecast endpoints and a natural-language agent layer
- the included UI is a chat-style forecasting workspace

The project uses the Online Retail workbook in [`online_retail.xlsx`](./online_retail.xlsx), trains product and customer forecasting pipelines, stores cluster metadata in a database, saves weekly snapshots and model artifacts locally, and serves both an API and a browser UI.

## What This Project Does

You can ask questions like:

- `Forecast product 85123A for the next 4 weeks`
- `Forecast customer 14606 for the next 4 weeks`
- `Compare product 85123A vs product 85099B over the next 4 weeks`
- `Tell me which customers in the declining cluster have a declining forecast`

The system then:

1. identifies whether the request is about a product or a customer
2. maps the entity to a cluster
3. loads that cluster's trained model
4. generates a forecast from historical weekly data
5. returns both structured output and a human-readable summary

## Current Architecture

### 1. Data ingestion

The Excel workbook is loaded from [`online_retail.xlsx`](./online_retail.xlsx).

Implemented in:

- [`app/services/data_loader.py`](./app/services/data_loader.py)

Responsibilities:

- reads all sheets from the workbook
- normalizes column names
- normalizes `stock_code` and `customer_id`
- computes issue flags for dirty rows
- keeps all rows instead of deleting them
- marks whether a row is valid for product modeling or customer modeling
- sets invalid row contribution to zero for modeling purposes

### 2. Weekly feature generation

Implemented in:

- [`app/services/features.py`](./app/services/features.py)

For both products and customers, the pipeline creates weekly time series with:

- `revenue`
- `quantity`
- `invoice_count`
- `counterparties`
- `avg_unit_price`

Missing weeks are filled with zeros so each entity has a continuous weekly series.

### 3. Shape-based clustering

Implemented in:

- [`app/services/clustering.py`](./app/services/clustering.py)

Current clustering strategy:

- uses normalized weekly series so shape matters more than scale
- computes DTW distance between time series
- runs agglomerative clustering on the DTW distance matrix
- chooses a data-driven cluster count rather than blindly trusting the requested count
- uses a balanced assignment step so one or two clusters do not absorb most entities

This is designed to produce more diverse clusters than naive magnitude-based grouping.

### 4. Entity catalog and data quality handling

Implemented in:

- [`app/services/entity_catalog.py`](./app/services/entity_catalog.py)

Each product or customer is assigned a status:

- `ok`
- `sparse_history`
- `data_issue`

The catalog stores:

- issue summary
- issue codes
- active weeks
- valid transaction count
- cluster metadata
- nearest peers for sparse-history fallback

Important behavior:

- rows with issues are retained in the source load
- if an entity only survives through bad data, its forecast is forced to `0`
- if an entity has too little clean history, the forecast is blended from similar peers

### 5. Forecast model training

Implemented in:

- [`app/services/forecasting.py`](./app/services/forecasting.py)
- [`app/services/pipeline.py`](./app/services/pipeline.py)

For each cluster, the system trains one model per entity type and target metric.

Current forecasting behavior:

- primary candidates are Gradient Boosting models
- a Random Forest candidate is considered for larger training sets
- the best candidate is selected by temporal validation
- `DummyRegressor` fallback is used only when there is not enough training data
- multi-step forecasts are generated recursively

### 6. Router / agent layer

Implemented in:

- [`app/services/router.py`](./app/services/router.py)
- [`app/services/llm/base.py`](./app/services/llm/base.py)
- [`app/services/llm/ollama.py`](./app/services/llm/ollama.py)

The router layer does not do forecasting math.

It only decides:

- forecast vs compare vs scenario vs alerts
- product vs customer
- which IDs were mentioned
- whether a price scenario was requested

Then the local forecast service runs the actual cluster model.

Available router providers:

- `heuristic`
- `ollama`

## Project Layout

```text
.
├── app
│   ├── api                FastAPI routes
│   ├── core               settings and configuration
│   ├── db                 SQLAlchemy engine and session
│   ├── models             database tables
│   ├── prompts            system prompt for LLM routing
│   ├── repositories       DB read/write helpers
│   ├── schemas            request and response models
│   ├── services           ingestion, clustering, forecasting, routing
│   └── static             browser UI
├── artifacts              generated weekly snapshots and model files
├── scripts                command-line training helper
├── docker-compose.yml
├── Dockerfile
├── online_retail.xlsx
├── requirements.txt
└── README.md
```

## Tech Stack

- API: FastAPI
- DB layer: SQLAlchemy
- Docker database: PostgreSQL 16
- Local fallback database: SQLite
- Data processing: pandas, numpy
- Modeling: scikit-learn
- Workbook reading: openpyxl
- LLM router option: Ollama
- UI: static HTML/CSS/JS served by FastAPI

## Dataset and Identifiers

Source workbook:

- [`online_retail.xlsx`](./online_retail.xlsx)

Entity IDs used throughout the app:

- product ID = `stock_code`
- customer ID = `customer_id`

Examples of known working entities:

- product: `85123A`
- product: `22774`
- customer: `14606`
- customer: `17841`

## Data Cleaning Rules

The pipeline preserves all raw rows, but not every row contributes to model training.

Current issue flags include:

- duplicate rows
- cancelled invoices
- non-positive quantity
- non-positive price
- missing invoice date
- missing stock code
- missing customer ID

Modeling behavior:

- product weekly metrics use rows valid for product forecasting
- customer weekly metrics use rows valid for customer forecasting
- invalid rows contribute `0` to modeling outputs
- issue details are stored up front in the entity catalog so the API does not need to recalculate them on each request

This is why some entities return:

- a normal forecast
- a sparse-history fallback forecast
- a hard zero forecast with a data-quality explanation

## What Gets Stored

### Database tables

Created automatically on startup by [`app/db/session.py`](./app/db/session.py):

- `cluster_assignments`
- `cluster_profiles`
- `forecast_model_records`
- `entity_catalog_records`

### Artifact files

Saved under [`artifacts`](./artifacts):

- `product_weekly_metrics.csv`
- `customer_weekly_metrics.csv`
- `artifacts/models/product/*.joblib`
- `artifacts/models/customer/*.joblib`

## API Surface

Base routes are registered in:

- [`app/api/routes.py`](./app/api/routes.py)

### Health and UI

- `GET /health`
- `GET /`
- `GET /docs`

### Training

- `POST /api/v1/pipeline/train`

### Forecasting

- `POST /api/v1/forecasts/entity`
- `POST /api/v1/forecasts/compare`
- `POST /api/v1/alerts/declining`

### Supporting data

- `GET /api/v1/history/entity`
- `GET /api/v1/context/entity`

### Agent layer

- `POST /api/v1/agent/query`

## Request Examples

### Train the pipeline

```bash
curl -X POST http://localhost:8001/api/v1/pipeline/train \
  -H "Content-Type: application/json" \
  -d '{}'
```

Smaller training run for quick validation:

```bash
curl -X POST http://localhost:8001/api/v1/pipeline/train \
  -H "Content-Type: application/json" \
  -d '{
    "product_clusters": 4,
    "customer_clusters": 4,
    "min_history_weeks": 8,
    "max_entities_for_dtw": 30,
    "lag_weeks": 8
  }'
```

### Forecast one product

```bash
curl -X POST http://localhost:8001/api/v1/forecasts/entity \
  -H "Content-Type: application/json" \
  -d '{
    "entity_type": "product",
    "entity_id": "85123A",
    "horizon_weeks": 4,
    "target_metric": "revenue"
  }'
```

### Forecast one customer

```bash
curl -X POST http://localhost:8001/api/v1/forecasts/entity \
  -H "Content-Type: application/json" \
  -d '{
    "entity_type": "customer",
    "entity_id": "14606",
    "horizon_weeks": 4,
    "target_metric": "revenue"
  }'
```

### Compare two products

```bash
curl -X POST http://localhost:8001/api/v1/forecasts/compare \
  -H "Content-Type: application/json" \
  -d '{
    "entity_type": "product",
    "entity_ids": ["85123A", "85099B"],
    "horizon_weeks": 4,
    "target_metric": "revenue"
  }'
```

### Declining alerts

```bash
curl -X POST http://localhost:8001/api/v1/alerts/declining \
  -H "Content-Type: application/json" \
  -d '{
    "entity_type": "customer",
    "cluster_label_contains": "declining",
    "horizon_weeks": 4,
    "target_metric": "revenue",
    "declining_threshold_pct": 0.1,
    "top_k": 10
  }'
```

### History endpoint

```bash
curl "http://localhost:8001/api/v1/history/entity?entity_type=product&entity_id=85123A&lookback_weeks=4"
```

### Entity context endpoint

```bash
curl "http://localhost:8001/api/v1/context/entity?entity_type=product&entity_id=22774"
```

### Agent query

```bash
curl -X POST http://localhost:8001/api/v1/agent/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Forecast product 85123A for the next 4 weeks",
    "target_metric": "revenue"
  }'
```

## UI

The browser UI is served from the FastAPI app.

Open:

- `http://localhost:8001/`

What the UI shows:

- chat-based query input
- quick prompt cards before the first message
- actuals + forecast chart
- hover tooltip on chart points
- recent timeline table
- entity context and forecast summary

The left chat panel stays pinned, while the right context panel scrolls.

## How To Run

### Option 1: Docker (recommended)

### Prerequisites

- Docker Desktop
- `docker compose`

### Start the stack

From the project root:

```bash
cd /Users/ayushkumar/Desktop/Columbia/Forecasting/Project/Retail/Deliverable_2
cp .env.example .env
docker compose up --build
```

If you are not using Ollama yet, change this in `.env` before starting:

```env
LLM_PROVIDER=heuristic
LLM_MODEL=
```

The default Docker port mapping is:

- host `8001` -> container `8000`

So the app is available at:

- `http://localhost:8001/`
- `http://localhost:8001/docs`
- `http://localhost:8001/health`

### Train after startup

In a second terminal:

```bash
curl -X POST http://localhost:8001/api/v1/pipeline/train \
  -H "Content-Type: application/json" \
  -d '{}'
```

### Rebuild after code changes

```bash
docker compose up -d --build api
```

### Stop the stack

```bash
docker compose down
```

### Option 2: Local Python

### Prerequisites

- Python 3.11+
- local virtual environment support

### Setup

```bash
cd /Users/ayushkumar/Desktop/Columbia/Forecasting/Project/Retail/Deliverable_2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run training locally

```bash
python3 scripts/train_pipeline.py
```

### Start the API

```bash
uvicorn app.main:app --reload --port 8000
```

Then open:

- `http://localhost:8000/`
- `http://localhost:8000/docs`

## Environment Variables

Configured in:

- [`.env`](./.env)
- [`.env.example`](./.env.example)
- [`app/core/config.py`](./app/core/config.py)

Common settings:

```env
POSTGRES_DB=retail_router
POSTGRES_USER=retail
POSTGRES_PASSWORD=retail
DATABASE_URL=postgresql+psycopg2://retail:retail@db:5432/retail_router
RAW_DATA_PATH=/app/online_retail.xlsx
ARTIFACTS_DIR=/app/artifacts
LLM_PROVIDER=heuristic
LLM_MODEL=
```

Notes:

- use `LLM_PROVIDER=heuristic` if you want the app to run without Ollama
- inside Docker on macOS, `host.docker.internal` is the expected Ollama host
- outside Docker, `OLLAMA_BASE_URL=http://localhost:11434`
- the code lowercases `LLM_PROVIDER`, so `ollama` and `Ollama` are both accepted

## Ollama Setup

The forecasting engine is local and deterministic. Ollama is only used for routing natural-language requests.

### What Ollama does

Ollama decides:

- intent
- entity type
- entity IDs
- scenario multiplier
- cluster label hints

Ollama does not:

- train models
- compute forecasts
- replace the cluster models

### Start Ollama

Example:

```bash
ollama pull llama3.1:8b
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

### Docker + Ollama

Use:

```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

Then rebuild the API:

```bash
docker compose up -d --build api
```

### Local Python + Ollama

Use:

```env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434
```

## Training Pipeline Details

The training pipeline is implemented in:

- [`app/services/pipeline.py`](./app/services/pipeline.py)
- [`scripts/train_pipeline.py`](./scripts/train_pipeline.py)

Training steps:

1. load workbook
2. normalize IDs and create row-quality flags
3. build weekly product metrics
4. build weekly customer metrics
5. save weekly snapshots under `artifacts/`
6. cluster entities using DTW + agglomerative clustering
7. save cluster assignments and profiles to the database
8. build the entity catalog with issue metadata and peer fallback metadata
9. train one forecast model per cluster
10. save model records to the database and `.joblib` files under `artifacts/models/`

## Forecast Semantics

### Product forecasts

Product forecasts predict future weekly product behavior for a `stock_code`.

Typical interpretation:

- expected future revenue or quantity
- recent baseline versus near-term forecast
- cluster behavior
- whether the product is normal, sparse, or data-constrained

### Customer forecasts

Customer forecasts predict future weekly customer purchase behavior for a `customer_id`.

Typical interpretation:

- expected future spend
- whether the customer is trending up or down
- whether the customer has enough clean history for a direct cluster forecast

## Example Working Entities

These have been verified in the current trained setup.

### Products

- `85123A`
  - label: `WHITE HANGING HEART T-LIGHT HOLDER`
- `22774`
  - label: `RED DRAWER KNOB ACRYLIC EDWARDIAN`

### Customers

- `14606`
- `17841`
- `15311`
- `14911`

## Troubleshooting

### `{"detail":"Not Found"}`

Usually means you are hitting the wrong container or port.

Check:

- `http://localhost:8001/health`
- `docker compose ps`

### `Bind for 0.0.0.0:8000 failed: port is already allocated`

The project uses host port `8001` by default now. If you changed it back to `8000`, free that port or remap it.

### `Failed to reach Ollama at http://host.docker.internal:11434`

Check:

1. Ollama is actually running
2. the model exists
3. the API container can reach the Ollama host
4. `OLLAMA_BASE_URL` matches your runtime

Useful checks:

```bash
curl http://localhost:11434/api/tags
docker compose exec api python -c "import urllib.request; print(urllib.request.urlopen('http://host.docker.internal:11434/api/tags').read().decode())"
```

### `No cluster mapping found for ...`

You need to run training first:

```bash
curl -X POST http://localhost:8001/api/v1/pipeline/train \
  -H "Content-Type: application/json" \
  -d '{}'
```

### `No weekly history found for product 22774`

This was caused by numeric `entity_id` type coercion in the weekly snapshot loader and has already been fixed in the current code.

### UI looks stale after a change

Hard refresh the browser after rebuilding the API container.

## Known Limitations

- no Alembic migrations yet; tables are created with `create_all`
- no automated test suite yet
- DTW is pure Python and will be slow at larger scale
- training currently runs in-process, not as a background job
- alerts currently depend on cluster filtering instead of a dedicated risk model
- natural-language routing is only as good as the selected provider and prompt

## Main Files To Read First

If you are new to this repository, start here:

- [`app/main.py`](./app/main.py)
- [`app/api/routes.py`](./app/api/routes.py)
- [`app/services/pipeline.py`](./app/services/pipeline.py)
- [`app/services/forecasting.py`](./app/services/forecasting.py)
- [`app/services/clustering.py`](./app/services/clustering.py)
- [`app/services/entity_catalog.py`](./app/services/entity_catalog.py)
- [`app/services/router.py`](./app/services/router.py)

## Current Status

This repository is a working prototype with:

- product forecasting
- customer forecasting
- compare and alert flows
- a chat-style UI
- optional Ollama routing
- Dockerized local deployment

The forecasting engine is live. The agent layer is usable and can be backed by Ollama, but it is still a lightweight router over local cluster models rather than a fully autonomous planning agent.
