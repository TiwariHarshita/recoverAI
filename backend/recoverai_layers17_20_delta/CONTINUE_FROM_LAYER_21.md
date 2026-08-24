# RecoverAI checkpoint: Layers 17–20

This is a DELTA-ONLY source bundle. It intentionally contains only files added or modified during Layers 17–20, not the frozen Layers 1–16 source tree.

## Frozen layers in this bundle

- Layer 17 — Batch Policy Evaluation ✅
  - CatBoost + ERV vs Logistic + ERV vs rules-first baseline
  - Layer tests: 8/8 passed
  - Full suite after L17: 188/188 passed
  - Fresh evaluation run completed successfully.
  - Observed 200-case sample: CatBoost+ERV clearly beat rules-first (+29.06% net value), while comparison with Logistic+ERV was inconclusive on that sample.

- Layer 18 — PostgreSQL Persistence ✅
  - SQLAlchemy persistence for customers, payments, subscriptions, invoices, recovery_cases, recovery_actions, merchant_policies, audit_events.
  - Layer tests: 9/9 passed
  - Full suite after L18: 197/197 passed
  - Local project PostgreSQL is on host port 5433 because host port 5432 is already occupied.
  - Working DATABASE_URL: postgresql+psycopg://recoverai:recoverai@localhost:5433/recoverai

- Layer 19 — Razorpay Test Mode Integration ✅
  - Test-mode-only client boundary; live keys fail closed.
  - Payment fetch/list, subscription fetch, invoice fetch, Payment Link create/fetch/cancel, typed errors and secret redaction.
  - Layer tests: 14/14 passed
  - Full suite after L19: 211/211 passed
  - Real ₹100 Payment Link creation intentionally deferred until final E2E testing.
  - Razorpay account/API keys are not yet required for continuing development.

- Layer 20 — Razorpay Webhook Verification + Normalization + Processing ✅
  - Raw-body HMAC verification, x-razorpay-event-id idempotency/dedup, normalization, persistence, recovery-case updates, recovered-revenue handling, audit events.
  - Added webhook_events persistence table.
  - Layer tests: 16/16 passed
  - Full suite after L20: 227/227 passed
  - FastAPI/Uvicorn local health endpoint was tested successfully.

## Current frozen architecture contract

Earlier Layers 1–16 remain frozen and are NOT included here. Continue from the existing repository plus this delta bundle.

Key rules to preserve:
- deterministic diagnosis
- deterministic candidate generation
- deterministic merchant policy / guardrails
- ML only estimates recovery probability/ranking
- ERV/economic math chooses among allowed actions
- money-moving/external actions stay bounded, explainable, gated and auditable
- Razorpay integration remains Test Mode only during development
- webhook processing must remain signature-verified and idempotent
- PostgreSQL host port for this local setup is 5433

## Next layer

Layer 21 — Recovery Action Execution

Build it on top of Layer 20. Requirements already agreed before the chat stopped:
- idempotent execution
- policy-gated execution
- Razorpay-backed execution where an action requires it
- auditable state transitions/results
- preserve all frozen contracts
- add dedicated Layer 21 tests and run full regression before freezing

## Useful verification commands

```bash
export DATABASE_URL="postgresql+psycopg://recoverai:recoverai@localhost:5433/recoverai"
python -m app.db.init_db
pytest tests/test_postgres_persistence.py -v
pytest tests/test_razorpay_integration.py -v
pytest tests/test_razorpay_webhooks.py -v
pytest -q
```

Expected frozen full-suite count after Layer 20: 227 passed.
