from __future__ import annotations

import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


DEFAULT_DATABASE_URL = "postgresql+psycopg://recoverai:recoverai@localhost:5433/recoverai"


class Base(DeclarativeBase):
    pass


def get_database_url() -> str:
    return os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


def build_engine(database_url: str | None = None, *, echo: bool = False) -> Engine:
    url = database_url or get_database_url()
    kwargs: dict[str, object] = {
        "echo": echo,
        "pool_pre_ping": True,
    }
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kwargs)


def build_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return build_engine()


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    return build_session_factory(get_engine())


@contextmanager
def session_scope(session_factory: sessionmaker[Session] | None = None) -> Iterator[Session]:
    factory = session_factory or get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_schema(engine: Engine | None = None) -> None:
    # Import registers all mapped tables on Base.metadata.
    from app.db import models as _models  # noqa: F401

    Base.metadata.create_all(bind=engine or get_engine())


def drop_schema(engine: Engine | None = None) -> None:
    from app.db import models as _models  # noqa: F401

    Base.metadata.drop_all(bind=engine or get_engine())
