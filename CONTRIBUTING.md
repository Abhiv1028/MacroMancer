# Contributing to Macromancer

Thanks for taking a look! This is a personal/portfolio project, but issues and
PRs are welcome.

## Development setup

```bash
git clone https://github.com/Abhiv1028/MacroMancer.git && cd MacroMancer
make setup                # venv + deps  (or: python -m venv venv && pip install -r requirements.txt)
source venv/bin/activate
make test                 # run the suite (185 tests)
```

macOS note: XGBoost needs the OpenMP runtime — `brew install libomp` (Linux uses
`libgomp1`, already handled in the Dockerfile and CI).

## Handy commands (see `make help`)

| command | does |
|---|---|
| `make run` / `make ui` | start the backend / the Streamlit dashboard |
| `make test` / `make cov` | tests / tests with coverage |
| `make eval` | evaluate the recommender vs baselines |
| `make train` | retrain the XGBoost model |
| `make lint` | pyflakes |
| `make docker` | build + run the full stack in one container |

## Guidelines

- **Tests:** add/adjust tests for any behavior change; keep CI green
  (`pytest` + 75% coverage gate). External services (Ollama, OCR, Nutritionix,
  OSM) must be mocked in tests.
- **Style:** type hints + docstrings on new functions; keep imports clean
  (`make lint` should pass).
- **Scope:** backend routes live under `backend/routes/`, business logic under
  `backend/services/`; the Streamlit UI only talks to the API via
  `frontend/api_client.py`.
- **Commits/PRs:** small and focused, with a clear description of what and why.

## Good first issues

- Add Alembic migrations (currently `create_all`).
- Per-user RL bandit weights (currently a single global bandit).
- Swap `print` calls for structured logging.
- Add `GET` pagination and richer filters to the list endpoints.
