# Pitch Vision Analytics

Pitch Vision Analytics is an IPL-focused AI/ML analytics engine built on ball-by-ball data. It combines a FastAPI backend, a React + TypeScript frontend, and a reproducible Python data pipeline to deliver tactical insights, predictive models, and hackathon-ready cricket intelligence.

## Highlights

- Team-wise IPL analytics dashboard
- Live win prediction for first and second innings
- Batter vs bowler head-to-head analysis
- Clutch Mode with Pressure-Resilience Index (PRI)
- Captain AI with phase-aware bowling recommendations
- Impact Lab for tactical substitution timing
- Player Styles: batter clustering with a 2-D style map
- Chase Coach: a reinforcement-learning agent for chase strategy across all 20 overs
- Data Proof outputs for cleaning, audit and model traceability

## Tech Stack

- Frontend: `React`, `TypeScript`, `Vite`
- Backend: `FastAPI`, `Python`
- Data and ML: `Pandas`, `NumPy`, `scikit-learn`, `XGBoost`, `Matplotlib`, `Seaborn`, Parquet marts
- Tactical assistant (optional): any `Ollama` model already installed locally
- Runs fully locally: no cloud or container setup needed

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
- Visual Insights gallery (Seaborn charts of phases, toss, correlations, season trends)
- Freak Innings detector for the most unusual innings in IPL history

### Win Predictor

- Live innings-state prediction
- Works for both first innings and chase situations
- Returns calibrated win probability and confidence band
- First innings: projected final score from Linear Regression and SVR
- Chase: second opinions from Decision Tree, SVM, AdaBoost and XGBoost, plus the decision-tree path behind the call

### Matchups

- Batter vs bowler head-to-head
- Partnership synergy lookup

### Clutch Mode

- Pressure-only leaderboard
- PRI is computed from high-pressure match states
- Eligibility currently requires at least `50` pressure innings

### Player Styles

- Batters with 300+ balls grouped into 4 styles (Power-hitter, Anchor, Accumulator, Grinder)
- 2-D style map of every batter, with an elbow chart explaining the choice of 4 groups

### Chase Coach (in Clutch Mode)

- Enter exact balls left (1-120), runs needed and wickets in hand
- The coach recommends Defend, Rotate or Attack with a win chance for each, flags close calls and rarely seen situations, and shows its learning curve on fixed test chases

### Captain AI

- Team-scoped player selection with all-player fallback
- Next-over bowling planning with legal overs remaining
- Phase-aware, sample-aware, non-consecutive bowling suggestions

### Data Proof

- Cleaning lineage and audit summary
- Chase model leaderboard, before/after hyperparameter tuning and score-model error metrics
- Readable decision-tree rules

### Impact Lab

- Role-based substitution timing simulation
- Estimates best over band for introducing impact roles

## Running the Project

Everything runs on your machine.

Create a virtual environment and install backend dependencies:

```powershell
python -m venv .venv313
.\.venv313\Scripts\python -m pip install -r apps/api/requirements.txt
```

Install frontend dependencies:

```powershell
npm install
```

Build processed data, models, charts and ML insights (about 1–2 minutes):

```powershell
.\.venv313\Scripts\python ml/scripts/build_all.py
```

Run the API (with the virtual environment activated):

```powershell
npm run api:dev
```

Run the frontend in another terminal:

```powershell
npm run web:dev
```

Open:

- Frontend: `http://localhost:4173`
- API docs: `http://localhost:8000/docs`

### Optional: Captain AI with Ollama

If Ollama is running locally, Captain AI automatically uses a model that is already installed (it prefers `tinyllama`, then `llama3.2:1b`, then `qwen2.5:3b`, otherwise the first installed model). It never downloads a model. Set `OLLAMA_MODEL` to force a specific installed model. Without Ollama, Captain AI falls back to its rule-based answer.

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
- `ml/scripts/train_insights.py`
- `ml/scripts/build_all.py`

It performs:

- dataset cleaning and normalization
- derived match-state feature generation
- leaderboard and dashboard mart creation
- pressure tables for PRI
- model artifact generation
- report generation for audit and proof
- charts, score/chase models, clustering, anomaly detection and the RL chase coach (`artifacts/reports/model_insights.json`)

## Machine Learning Techniques

| Feature in the app | Technique |
|---|---|
| Visual Insights (Dashboard) | NumPy, Pandas, Matplotlib, Seaborn |
| Projected Final Score (Win Predictor) | Simple and Multiple Linear Regression |
| Live Win Predictor | Logistic Regression with ROC-AUC / Brier evaluation |
| Decision Path + rules (Win Predictor, Data Proof) | Decision Tree |
| SVM second opinion, SVR projected score, Freak Innings | SVC, SVR, One-Class SVM (outlier detection) |
| Second opinions + model leaderboard | AdaBoost, XGBoost |
| Batter Styles (Player Styles) | K-Means Clustering (elbow method) |
| Style Map (Player Styles) | PCA (2 components) |
| Chase Coach (Clutch Mode) | Reinforcement Learning (tabular Q-learning) |
| Before vs after tuning (Data Proof) | GridSearchCV, RandomizedSearchCV |

## Testing

Run the full test suite:

```powershell
pytest
```

## Notes

- The repository ignores generated Parquet and model binaries by default.
- Captain AI is grounded on processed cricket data and rule-based constraints; it is not a free-form chatbot.
- Some frontend build warnings about chunk size may still appear, but they do not block production builds.
