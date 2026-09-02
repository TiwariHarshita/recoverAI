# RecoverAI

RecoverAI is a Python backend and simulation project for revenue-recovery
workflows. The repository currently includes a FastAPI webhook service,
domain and policy logic, synthetic simulations, ML experiments, PostgreSQL
persistence code, and Razorpay Test Mode adapters.

## Backend setup

Python 3.11 or newer and Docker with Compose are required.

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
cp .env.example .env
```

The application reads environment variables from the process environment; it
does not implicitly load `.env`. Export the values you need before running the
database initializer, API, or Razorpay smoke test. For example:

```bash
set -a
source .env
set +a
```

The example Razorpay values are placeholders. Keep test credentials and
webhook secrets in the untracked `.env` file and use Test Mode keys only.

## PostgreSQL

Start the local PostgreSQL service from `backend/`:

```bash
docker compose -f docker-compose.postgres.yml up -d
recoverai-init-db
```

The Compose service exposes PostgreSQL on `localhost:5433`, matching the
canonical `DATABASE_URL` in `.env.example` and the application's development
default.

## Run the API

From `backend/` with the virtual environment active:

```bash
uvicorn app.main:app --reload
```

The health endpoint is available at `http://127.0.0.1:8000/health`.

## Run tests

From `backend/`:

```bash
pytest
```

Pytest configuration and import paths are defined in `pyproject.toml`; no
manual `PYTHONPATH` value is required.
