# Using Gemini Provider with AI Correction System

## Overview

The AI Correction System now supports Google's Gemini models as an alternative to OpenAI. The Gemini provider implements the same interface as the OpenAI provider, making it easy to switch between them.

## Installation

1. Install the required dependencies:
```bash
pip install google-generativeai
```

2. Set your Gemini API key:
```bash
export AI_CORRECTION_GEMINI_API_KEY="your-gemini-api-key"
```

Or add to your `.env` file:
```
AI_CORRECTION_GEMINI_API_KEY=your-gemini-api-key
```

## Configuration

### Available Models

- **gemini-2.5-flash-exp** (default): Fast, efficient model for most tasks
- **gemini-2.5-pro**: Higher quality model for complex tasks
- **text-embedding-004**: Embedding model for clustering and similarity

### Configuration Options

In your `.env` file or environment variables:

```bash
# Use Gemini as the AI provider
AI_CORRECTION_AI_PROVIDER=gemini

# Optional: Specify custom models
AI_CORRECTION_GEMINI_MODEL=gemini-2.5-flash-exp
AI_CORRECTION_GEMINI_VISION_MODEL=gemini-2.5-flash-exp
AI_CORRECTION_GEMINI_EMBEDDING_MODEL=text-embedding-004
```

## Usage

### Direct Usage

```python
from src.ai import GeminiProvider

# Create provider
provider = GeminiProvider()

# Grade with vision
result = provider.grade_with_vision(
    question_text="What is 2 + 2?",
    criteria="Correct answer gets full points",
    image_path="student_answer.png",
    max_points=5.0
)

print(f"Grade: {result['grade']}")
print(f"Confidence: {result['confidence']}")
print(f"Rationale: {result['rationale']}")
```

### Using the Factory (Recommended)

```python
from src.ai import create_ai_provider

# Auto-detects provider based on available API keys
provider = create_ai_provider()

# Or explicitly specify
provider = create_ai_provider(provider_type="gemini")
```

### In Session Orchestrator

The session orchestrator will automatically use the configured provider:

```python
from src.core.session import GradingSessionOrchestrator

orchestrator = GradingSessionOrchestrator(
    pdf_paths=["copy1.pdf", "copy2.pdf"]
)

# Uses provider from settings (OpenAI or Gemini)
result = await orchestrator.run()
```

## Switching Between Providers

### Method 1: Environment Variable

Set the provider in `.env`:
```
AI_CORRECTION_AI_PROVIDER=gemini
```

### Method 2: Programmatic

```python
from src.ai import create_ai_provider

# Use Gemini
gemini_provider = create_ai_provider("gemini")

# Use OpenAI
openai_provider = create_ai_provider("openai")
```

## API Compatibility

The Gemini provider implements the same interface as OpenAI:

- `call_vision(prompt, image_path, image_bytes, pil_image)` - Grade with images
- `call_text(prompt, system_prompt)` - Text analysis
- `get_embedding(text)` - Get text embeddings
- `get_embeddings(texts)` - Batch embeddings
- `grade_with_vision()` - Grading with vision
- `extract_rule()` - Extract grading rules
- `analyze_cross_copy()` - Cross-copy analysis
- `generate_feedback()` - Student feedback

## Model Selection

### Gemini 2.5 Flash (Default)

- Best for: Speed and cost efficiency
- Use cases: Quick grading, large batches, development
- Good quality for most handwritten answers

```python
provider = GeminiProvider(model="gemini-2.5-flash-exp")
```

### Gemini 2.5 Pro

- Best for: Maximum accuracy
- Use cases: Complex questions, important exams, edge cases
- Better at: Ambiguous handwriting, nuanced understanding

```python
provider = GeminiProvider(model="gemini-2.5-pro")
```

## Performance Considerations

### Speed

- Gemini Flash is typically faster than GPT-4o
- Good for real-time or batch processing
- Pro model is slower but more accurate

### Cost

- Check Google AI pricing for current rates
- Flash model is more cost-effective for large batches
- Pro model for critical evaluations

### Quality

- Both models handle handwritten text well
- Pro model better for:
  - Mathematical notation
  - Diagrams and graphs
  - Ambiguous or messy handwriting

## Example: Complete Grading Workflow

```python
from src.ai import create_ai_provider
from src.core.session import GradingSessionOrchestrator

# 1. Create provider (auto-detects from settings)
provider = create_ai_provider()

# 2. Create session
orchestrator = GradingSessionOrchestrator(
    pdf_paths=["student_copies/*.pdf"]
)

# 3. Run grading (uses configured provider)
result = await orchestrator.run()

print(f"Graded {result['graded_count']} copies")
print(f"Average score: {result['average_score']}")
```

## Troubleshooting

### Module Not Found Error

```
ModuleNotFoundError: No module named 'google.generativeai'
```

**Solution**: Install the dependency:
```bash
pip install google-generativeai
```

### API Key Not Found

```
ValueError: Gemini API key is required
```

**Solution**: Set the API key:
```bash
export AI_CORRECTION_GEMINI_API_KEY="your-key"
```

### Model Not Available

If you encounter issues with `gemini-2.5-flash-exp`, try:
```bash
export AI_CORRECTION_GEMINI_MODEL=gemini-2.5-pro
```

## Migration from OpenAI

Switching from OpenAI to Gemini is straightforward:

1. Install Gemini package and set API key
2. Change provider in settings or code
3. No code changes needed - interface is identical

The system will automatically use the new provider for all AI operations.

## Testing

To test without using API credits, use mock mode:

```python
provider = GeminiProvider(mock_mode=True)
```

This returns mock responses without making API calls.
