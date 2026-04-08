import pytest
from unittest.mock import MagicMock

@pytest.fixture(autouse=True)
def mock_embeddings(monkeypatch):
    """Automatically mock embeddings for all tests to prevent model downloads."""
    mock_emb = MagicMock()
    # Mock the lazy loading function in main
    monkeypatch.setattr("main.get_embeddings", lambda: mock_emb)
    return mock_emb

@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    """Ensure environment variables are set for tests."""
    monkeypatch.setenv("GROQ_API_KEY", "test_key")
