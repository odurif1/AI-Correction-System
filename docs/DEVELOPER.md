# Developer Guide

This guide helps developers set up and contribute to the AI Correction System.

## Development Setup

### Prerequisites

- Python 3.11+
- Poetry or pip
- API keys for LLM providers (Gemini, OpenAI)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Correction

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your API keys
```

### Environment Variables

Create a `.env` file with:

```env
# Provider API Keys
AI_CORRECTION_GEMINI_API_KEY=your-gemini-key
AI_CORRECTION_OPENAI_API_KEY=your-openai-key

# Primary Provider
AI_CORRECTION_PROVIDER=gemini
AI_CORRECTION_MODEL=gemini-2.5-flash

# Dual LLM Mode
AI_CORRECTION_COMPARISON_MODE=true
AI_CORRECTION_LLM1_PROVIDER=gemini
AI_CORRECTION_LLM1_MODEL=gemini-2.5-flash
AI_CORRECTION_LLM2_PROVIDER=openai
AI_CORRECTION_LLM2_MODEL=gpt-4o

# API Authentication (optional)
AI_CORRECTION_API_KEY=your-api-key
```

## Project Structure

```
src/
├── ai/                    # AI provider implementations
│   ├── base_provider.py   # Abstract base class
│   ├── gemini_provider.py # Google Gemini
│   ├── openai_provider.py # OpenAI
│   ├── comparison_provider.py  # Dual LLM
│   └── batch_grader.py    # Batch grading logic
├── api/                   # FastAPI application
│   ├── app.py            # Main application
│   ├── schemas.py        # Request/response models
│   └── websocket.py      # WebSocket handling
├── config/               # Configuration
│   ├── settings.py       # Environment settings
│   └── constants.py      # System constants
├── core/                 # Core business logic
│   ├── models.py         # Pydantic data models
│   ├── session.py        # Session orchestration
│   ├── exceptions.py     # Custom exceptions
│   └── workflow_state.py # Workflow state machine
├── export/               # Export functionality
│   ├── pdf_annotator.py  # PDF annotation
│   ├── annotation_service.py  # Coordinate detection
│   └── analytics.py      # Data export
├── grading/              # Grading logic
│   └── grader.py         # Intelligent grader
├── interaction/          # User interaction
│   └── cli.py            # CLI interface
├── prompts/              # LLM prompts
│   ├── grading.py        # Grading prompts
│   ├── batch.py          # Batch prompts
│   ├── common.py         # Shared prompts
│   └── analysis.py       # Analysis prompts
├── storage/              # Data persistence
│   ├── session_store.py  # Session storage
│   └── file_store.py     # File operations
└── utils/                # Utilities
    ├── confidence.py     # Confidence helpers
    └── json_extractor.py # JSON parsing
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_grader.py -v
```

## Running the Application

### CLI Mode

```bash
# Grade PDFs
python -m src.main correct dual batch path/to/pdfs/ --debug --auto-confirm

# List sessions
python -m src.main list

# Export results
python -m src.main export <session_id> --format csv
```

### API Mode

```bash
# Start server
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

# Access API docs
open http://localhost:8000/docs
```

## Adding a New AI Provider

1. **Create the provider class** in `src/ai/`:

```python
from ai.base_provider import BaseProvider

class MyProvider(BaseProvider):
    def __init__(self, model: str, api_key: str):
        super().__init__()
        self.model = model
        self.client = MyClient(api_key)

    def call_vision(self, prompt, image_path=None, **kwargs) -> str:
        # Implement vision API call
        pass

    def call_text(self, prompt, system_prompt=None, **kwargs) -> str:
        # Implement text API call
        pass

    def get_embedding(self, text: str) -> list:
        # Implement embedding
        pass

    def get_embeddings(self, texts: list) -> list:
        # Implement batch embedding
        pass
```

2. **Register in provider factory** (`src/ai/provider_factory.py`):

```python
def create_ai_provider(provider_name: str, **kwargs):
    if provider_name == "myprovider":
        return MyProvider(**kwargs)
    # ... existing providers
```

3. **Add configuration** in `src/config/settings.py`:

```python
myprovider_api_key: str = Field(default="", alias="AI_CORRECTION_MYPROVIDER_API_KEY")
myprovider_model: str = Field(default="myprovider-1.0")
```

## Code Style

- Use type hints for all function parameters and returns
- Follow PEP 8 naming conventions
- Keep functions under 50 lines when possible
- Use Pydantic models for data validation
- Add docstrings for public functions and classes

## Debugging

### Enable Debug Capture

```bash
export AI_CORRECTION_DEBUG_CAPTURE=true
python -m src.main correct ...
```

Debug output is saved to `debug_output/` directory.

### Verbose Logging

```bash
python -m src.main correct ... --debug
```

## Common Issues

### Import Errors

Make sure you're running from the project root with the virtual environment activated:

```bash
cd Correction
source .venv/bin/activate
python -m src.main ...
```

### API Key Issues

Check that environment variables are set correctly:

```bash
python -c "from config.settings import get_settings; s=get_settings(); print(s.gemini_api_key[:10])"
```

### Memory Issues with Large PDFs

The image cache is limited to 128 items. For large batches:

```python
from ai.base_provider import clear_image_cache
clear_image_cache()  # Call periodically
```
