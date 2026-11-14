"""Pytest fixtures and environment bootstrap."""

import os
import pytest
from httpx import ASGITransport, AsyncClient

os.environ["APP_ENV"] = "testing"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["DEEPSEEK_API_KEY"] = ""
os.environ["LLM_API_KEY"] = ""
os.environ["LLM_BASE_URL"] = ""
os.environ["LLM_MODEL"] = ""

from policy_rag.api.main import create_app
from policy_rag.config import Settings
from policy_rag.service import PolicyContainer


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        app_env="testing",
        app_name="policy-rag-test",
    )


@pytest.fixture
def test_container(test_settings: Settings) -> PolicyContainer:
    return PolicyContainer(settings=test_settings)


@pytest.fixture
def app(test_container: PolicyContainer):
    return create_app(test_container)


@pytest.fixture
async def client(app) -> AsyncClient:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
        yield ac
