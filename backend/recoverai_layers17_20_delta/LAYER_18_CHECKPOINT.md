# Layer 18 — PostgreSQL Persistence

Status: COMPLETE / FROZEN

Added:
- SQLAlchemy 2.x database/session infrastructure
- PostgreSQL `DATABASE_URL` configuration
- Persistent tables for customers, payments, subscriptions, invoices, recovery cases, recovery actions, merchant policies, and audit events
- Domain ↔ persistence repositories
- Case/action/audit history queries
- Transaction commit/rollback context manager
- PostgreSQL Docker Compose config
- Layer 18 persistence test suite

Validation:
- Layer 18 tests: 9 passed
- Full backend suite: 197 passed

Next layer: Razorpay Test Mode integration.
