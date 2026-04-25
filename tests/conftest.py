from __future__ import annotations
import pytest

from llm_values.clients.mock import MockChatClient


@pytest.fixture
def mock_client():
    return MockChatClient()
