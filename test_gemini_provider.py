#!/usr/bin/env python3
"""
Test script for Gemini provider.

Usage:
    # With real API (requires GEMINI_API_KEY)
    AI_CORRECTION_GEMINI_API_KEY=your-key python test_gemini_provider.py

    # With mock mode (no API key needed)
    python test_gemini_provider.py --mock
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_imports():
    """Test that all imports work."""
    print("Testing imports...")

    try:
        from src.ai import GeminiProvider, create_ai_provider
        print("✓ GeminiProvider imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        print("  Note: google-generativeai package not installed")
        return False


def test_provider_creation(mock_mode=False):
    """Test provider instantiation."""
    print("\nTesting provider creation...")

    try:
        from src.ai import GeminiProvider

        provider = GeminiProvider(mock_mode=mock_mode)
        print(f"✓ GeminiProvider created (mock={mock_mode})")
        print(f"  Model: {provider.model}")
        print(f"  Vision Model: {provider.vision_model}")
        print(f"  Embedding Model: {provider.embedding_model}")
        return provider
    except Exception as e:
        print(f"✗ Provider creation failed: {e}")
        return None


def test_text_call(provider):
    """Test text API call."""
    print("\nTesting text call...")

    try:
        response = provider.call_text(
            prompt="What is 2 + 2?",
            system_prompt="You are a helpful assistant."
        )
        print(f"✓ Text call successful")
        print(f"  Response: {response[:100]}...")
        return True
    except Exception as e:
        print(f"✗ Text call failed: {e}")
        return False


def test_embedding(provider):
    """Test embedding generation."""
    print("\nTesting embedding generation...")

    try:
        embedding = provider.get_embedding("Test text for embedding")
        print(f"✓ Embedding generated")
        print(f"  Dimension: {len(embedding)}")
        return True
    except Exception as e:
        print(f"✗ Embedding generation failed: {e}")
        return False


def test_factory():
    """Test provider factory."""
    print("\nTesting provider factory...")

    try:
        from src.ai import create_ai_provider, get_available_providers

        # Check available providers
        available = get_available_providers()
        print(f"  Available providers: {available}")

        # Try to create Gemini provider
        try:
            provider = create_ai_provider("gemini", mock_mode=True)
            print(f"✓ Factory created Gemini provider")
        except ValueError as e:
            print(f"  Factory error (expected if no API key): {e}")
            return False

        return True
    except Exception as e:
        print(f"✗ Factory test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Gemini provider")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock mode (no API calls)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Gemini Provider Test Suite")
    print("=" * 60)

    # Check for API key
    api_key = os.environ.get("AI_CORRECTION_GEMINI_API_KEY")
    if not args.mock and not api_key:
        print("\n⚠ Warning: No GEMINI_API_KEY found")
        print("  Set AI_CORRECTION_GEMINI_API_KEY or use --mock")
        print("  Continuing with mock mode...\n")
        args.mock = True

    # Run tests
    tests = [
        ("Imports", test_imports),
        ("Provider Creation", lambda: test_provider_creation(args.mock)),
        ("Factory", test_factory),
    ]

    # Add API tests only if provider creation succeeded
    provider = None
    if args.mock or api_key:
        try:
            from src.ai import GeminiProvider
            provider = GeminiProvider(mock_mode=args.mock)
        except ImportError:
            print("\n⚠ Warning: GeminiProvider not available")
            print("  Install with: pip install google-generativeai")
            provider = None
        tests.extend([
            ("Text Call", lambda: test_text_call(provider)),
            ("Embedding", lambda: test_embedding(provider)),
        ])

    # Execute tests
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
