#!/usr/bin/env bash
# setup.sh — One-command local setup for the Telco Churn API
# Usage:  bash setup.sh
# Tested: Python 3.10, 3.11, 3.12

set -e  # exit immediately if any command fails

echo ""
echo "=========================================="
echo "  Telco Churn API — Local Setup"
echo "=========================================="
echo ""

# 1. Check Python version
PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
    echo "ERROR: Python not found. Install Python 3.10+ and try again."
    exit 1
fi

PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Using Python $PY_VERSION at $($PYTHON -c 'import sys; print(sys.executable)')"

MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")
if [ "$MAJOR" -lt 3 ] || { [ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]; }; then
    echo "ERROR: Python 3.10 or higher is required. You have $PY_VERSION."
    exit 1
fi

# 2. Create virtual environment
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    $PYTHON -m venv venv
    echo "Created: ./venv"
else
    echo "Virtual environment already exists at ./venv"
fi

# 3. Activate virtual environment
# Source the correct activation script for the OS
if [ -f "venv/Scripts/activate" ]; then
    # Windows (Git Bash / WSL path)
    source venv/Scripts/activate
else
    source venv/bin/activate
fi
echo "Activated: $(which python)"

# 4. Upgrade pip silently
echo ""
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# 5. Install dependencies
echo ""
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# 6. Verify every import from main.py works
echo ""
echo "Verifying imports..."
python -c "
import joblib, numpy, pandas, fastapi, pydantic, uvicorn, sklearn, xgboost
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator
print('  joblib       ', joblib.__version__)
print('  numpy        ', numpy.__version__)
print('  pandas       ', pandas.__version__)
print('  fastapi      ', fastapi.__version__)
print('  pydantic     ', pydantic.__version__)
print('  uvicorn      ', uvicorn.__version__)
print('  scikit-learn ', sklearn.__version__)
print('  xgboost      ', xgboost.__version__)
print()
print('All imports OK.')
"

# 7. Start the server
echo ""
echo "=========================================="
echo "  Setup complete. Starting server..."
echo "  API docs → http://localhost:8000/docs"
echo "  Press Ctrl+C to stop"
echo "=========================================="
echo ""
uvicorn main:app --reload --host 0.0.0.0 --port 8000
