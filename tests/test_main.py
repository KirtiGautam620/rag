import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from main import app

client = TestClient(app)

def test_read_root():
    """Verify that the landing page loads correctly."""
    response = client.get("/")
    assert response.status_code == 200
    assert "AI RAG Assistant" in response.text

def test_list_documents_empty(tmp_path):
    """Verify that the documents list returns correctly even if empty."""
    with patch("main.UPLOAD_DIR", tmp_path):
        response = client.get("/documents")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

def test_upload_invalid_file():
    """Verify that unsupported file types are rejected."""
    response = client.post(
        "/upload",
        files={"file": ("test.exe", b"invalid content", "application/x-msdownload")}
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]

@patch("main.vector_store", None)
def test_query_no_document():
    """Verify that querying fails if no document is uploaded."""
    response = client.post(
        "/query",
        json={"question": "What is RAG?"}
    )
    assert response.status_code == 400
    assert "Please upload a document first" in response.json()["detail"]

def test_clear_history():
    """Verify that history clearing endpoint works."""
    with patch("main.memory.clear") as mock_clear:
        response = client.delete("/history")
        assert response.status_code == 200
        assert response.json()["message"] == "Conversation history cleared."
        mock_clear.assert_called_once()
