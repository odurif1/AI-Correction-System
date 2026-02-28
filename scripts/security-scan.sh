#!/bin/bash
set -e

echo "=== Security Scan: La Corrigeuse ==="
echo ""

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# pip-audit: Check for known vulnerabilities in dependencies
echo "1. Scanning Python dependencies for known vulnerabilities..."
if pip-audit --desc --format json; then
    echo -e "${GREEN}✓ No vulnerabilities found${NC}"
else
    echo -e "${RED}✗ Vulnerabilities detected${NC}"
    # Continue scan but flag for review
fi
echo ""

# bandit: Check Python code for security issues
echo "2. Running static security analysis on Python code..."
bandit -r src/ \
    --severity-level medium \
    --confidence-level medium \
    --format json \
    --output scripts/bandit-report.json || true

# Check exit code (bandit returns 1 if issues found)
if bandit -r src/ --severity-level medium --confidence-level medium; then
    echo -e "${GREEN}✓ No security issues found${NC}"
else
    echo -e "${YELLOW}⚠ Security issues detected - check bandit-report.json${NC}"
    # Don't fail on bandit issues (warnings only)
fi
echo ""

echo "=== Security Scan Complete ==="
