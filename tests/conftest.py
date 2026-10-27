import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine

from crisk.config import get_settings
from tests import ROOT


@pytest.fixture
def pg_url():
    url = get_settings().test_database_url
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    return url


@pytest.fixture
def migrated(pg_url):
    """A test database migrated to head, torn back down afterwards."""
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.attributes["url"] = pg_url
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    engine = create_engine(pg_url)
    yield engine
    engine.dispose()
    command.downgrade(cfg, "base")
