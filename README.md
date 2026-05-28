# Pitch Vision Analytics

Pitch Vision Analytics is an IPL-focused AI/ML analytics engine built on ball-by-ball data. It combines a FastAPI backend, a React + TypeScript frontend, and a reproducible Python data pipeline to deliver tactical insights, predictive models, and hackathon-ready cricket intelligence.

## Highlights

- Team-wise IPL analytics dashboard
- Live win prediction for first and second innings
- Batter vs bowler head-to-head analysis
- Clutch Mode with Pressure-Resilience Index (PRI)
- Captain AI with phase-aware bowling recommendations
- Impact Lab for tactical substitution timing
- Data Proof outputs for cleaning and audit traceability

## Tech Stack

- Frontend: `React`, `TypeScript`, `Vite`
- Backend: `FastAPI`, `Python`
- Data and ML: `Pandas`, `NumPy`, model artifacts, Parquet marts
- Tactical assistant: `Ollama` with `TinyLlama` for local grounded generation
- Deployment: `Docker Compose`

## Project Structure

```text
.
|-- apps/
|   |-- api/                 # FastAPI backend
|   `-- web/                 # React + TypeScript frontend
|-- ml/
|   `-- scripts/             # Data cleaning, training, artifact builders
|-- data/
|   |-- raw/                 # Source dataset and curated metadata
|   `-- processed/           # Generated Parquet outputs
|-- artifacts/
|   |-- models/              # Trained model files
|   |-- reports/             # Dashboard summary and audit proof
|   `-- vectorstore/         # Retrieval artifacts for Captain AI
|-- tests/                   # API and pipeline tests
|-- docker-compose.yml
`-- README.md
```

## Features

### Dashboard

- Toss impact by team
- Bat-first vs bowl-first win splits
- 200+ chase success and 150-or-lower defense success
- Phase-wise winner vs loser scoring comparison
- Top batters and top bowlers leaderboard
- Surprising finding generated from the dataset

### Win Predictor

- Live innings-state prediction
- Works for both first innings and chase situations
- Returns calibrated win probability and confidence band

### Matchups

- Batter vs bowler head-to-head
- Partnership synergy lookup

### Clutch Mode

- Pressure-only leaderboard
- PRI is computed from high-pressure match states
- Eligibility currently requires at least `50` pressure innings

### Captain AI

- Team-scoped player selection with all-player fallback
- Next-over bowling planning with legal overs remaining
- Phase-aware, sample-aware, non-consecutive bowling suggestions

### Impact Lab

- Role-based substitution timing simulation
- Estimates best over band for introducing impact roles

## Running the Project

### Option 1: Docker Compose

Start the app from the project root:

```powershell
$env:DOCKER_BUILDKIT='0'
$env:COMPOSE_DOCKER_CLI_BUILD='0'
docker compose up --build -d
```

Open:

- Frontend: `http://localhost:4173`
- API docs: `http://localhost:8000/docs`

To stop the stack:

```powershell
docker compose down
```

Optional LLM services:

```powershell
docker compose --profile llm up -d
```

### Option 2: Local Development

Install backend dependencies:

```powershell
python -m pip install -r apps/api/requirements.txt
```

Install frontend dependencies:

```powershell
npm install
```

Build processed data and model artifacts:

```powershell
python ml/scripts/build_all.py
```

Run the API:

```powershell
npm run api:dev
```

Run the frontend in another terminal:

```powershell
npm run web:dev
```

## Available Scripts

```bash
npm run web:dev
npm run web:build
npm run api:dev
npm run pipeline
npm test
```

## Data Pipeline

The pipeline is centered around:

- `ml/scripts/clean_ipl_data.py`
- `ml/scripts/train_models.py`
- `ml/scripts/build_all.py`

It performs:

- dataset cleaning and normalization
- derived match-state feature generation
- leaderboard and dashboard mart creation
- pressure tables for PRI
- model artifact generation
- report generation for audit and proof

## Testing

Run the full test suite:

```powershell
pytest
```

## Notes

- The repository ignores generated Parquet and model binaries by default.
- Captain AI is grounded on processed cricket data and rule-based constraints; it is not a free-form chatbot.
- Some frontend build warnings about chunk size may still appear, but they do not block production builds.
