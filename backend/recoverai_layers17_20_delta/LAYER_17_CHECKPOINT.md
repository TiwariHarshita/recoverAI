# RecoverAI Layer 17 — Batch Policy Evaluation

Status: COMPLETE / TESTED / FROZEN

Files:
- simulator/evaluation.py
- experiments/evaluate_policies.py
- tests/test_batch_evaluation.py

Policies compared:
1. CatBoost + ERV
2. Logistic Regression + ERV
3. Deterministic rules-first baseline

Evaluation contract:
- fresh unseen synthetic cases
- same cases for all policies
- repeated simulator rollouts
- policy gates preserved before ML
- latent simulator probability never used for selection
- approval requirement tracked explicitly
- primary metrics: recovered amount, recovered amount rate, expected net recovery value
- paired comparisons + bootstrap 95% confidence intervals

Validation:
- Layer 17 tests: 8 passed
- Full backend regression: 188 passed
- Real trained-model evaluation script: completed successfully

Next layer: Layer 18 — PostgreSQL persistence
