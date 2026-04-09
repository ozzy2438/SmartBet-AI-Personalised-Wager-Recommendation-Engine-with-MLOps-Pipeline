# SmartBet AI

Production-style stage-1 build of a two-tower recommendation engine for personalised sports betting market suggestions. The project includes synthetic data generation, feature engineering, PyTorch training, evaluation, drift checks, a lightweight MLOps agent, a FastAPI serving layer, a Streamlit dashboard, Docker support, and a GitHub Actions smoke pipeline.

## Project Layout

```text
smartbet_ai-sports/
├── api/
├── configs/
├── dashboard/
├── data/
├── models/
├── src/
│   ├── smartbet_ai/
│   ├── data_generation.py
│   ├── data_validation.py
│   ├── feature_engineering.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── drift_check.py
│   ├── mlops_agent.py
│   └── register_model.py
├── .github/workflows/mlops_pipeline.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## End-to-End Run

```bash
python src/data_generation.py
python src/data_validation.py
python src/feature_engineering.py
python src/train.py
python src/evaluate.py
python src/drift_check.py
python src/register_model.py
```

## Serve the Model

```bash
uvicorn api.serve:app --reload
```

Endpoints:

- `GET /health`
- `GET /model-info`
- `POST /recommend`

Example request:

```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "top_k": 5, "sport_filter": "NBA", "exclude_live": true}'
```

## Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard reads local training artifacts and can preview recommendations from the saved model when a checkpoint exists.

## Runtime Overrides

The config files remain the default interface, but lightweight smoke runs can override the largest values via environment variables:

- `SMARTBET_TRAINING_EPOCHS`
- `SMARTBET_BATCH_SIZE`
- `SMARTBET_N_USERS`
- `SMARTBET_N_MARKETS`
- `SMARTBET_N_INTERACTIONS`
- `SMARTBET_N_NEGATIVES`

## Notes

- Synthetic data is the default source in stage 1.
- The MLOps agent uses OpenAI routing when `OPENAI_API_KEY` is present and otherwise falls back to a deterministic local router.
- Model registration writes to a local registry file and attempts MLflow logging when MLflow is available.
