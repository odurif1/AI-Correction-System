import pytest


@pytest.fixture
def provider():
    from src.ai.gemini_provider import GeminiProvider

    return GeminiProvider(api_key="dummy_key", mock_mode=True)


def test_imports():
    from src.ai.gemini_provider import GeminiProvider

    assert GeminiProvider is not None


def test_provider_creation():
    from src.ai.gemini_provider import GeminiProvider

    provider = GeminiProvider(api_key="dummy_key", mock_mode=True)
    assert provider is not None


def test_text_call(provider):
    result = provider.call_text("Test prompt")
    assert isinstance(result, str)


def test_embedding(provider):
    result = provider.get_embedding("Test text")
    assert isinstance(result, list)
