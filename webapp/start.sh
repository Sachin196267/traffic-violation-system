#!/bin/bash
# Traffic Violation System Web Application — Startup Script

set -e

cd "$(dirname "$0")"

echo "🚦 ITMS Traffic Violation Web Application"
echo "=========================================="
echo ""

# Check if we're in the correct directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found. Run this script from the webapp/ directory."
    exit 1
fi

# Check for virtual environment
if [ -d "../.venv" ]; then
    source ../.venv/bin/activate
    echo "✓ Activated virtual environment: .venv"
elif [ -d "../traffic_env" ]; then
    source ../traffic_env/bin/activate
    echo "✓ Activated virtual environment: traffic_env"
else
    echo "⚠️  No virtual environment found. Creating..."
    python3 -m venv ../.venv
    source ../.venv/bin/activate
    echo "✓ Created and activated virtual environment: .venv"
fi

# Check if web application dependencies are already installed
if ! python -c "import fastapi, uvicorn, jwt, passlib, pydantic" 2>/dev/null; then
    echo "Installing web application dependencies..."
    pip install -r requirements.txt
    echo "✓ Dependencies installed"
else
    echo "✓ Dependencies verified"
fi

# Check if pipeline dependencies are installed
if ! python -c "import ultralytics, cv2, yaml, reportlab" 2>/dev/null; then
    echo "Installing missing pipeline dependencies..."
    pip install -r ../requirements.txt 2>/dev/null || true
fi

# Free port 8000 if already occupied by a previous process
if lsof -ti :8000 >/dev/null 2>&1; then
    echo "Releasing port 8000..."
    lsof -ti :8000 | xargs kill -9 2>/dev/null || true
    sleep 1
fi

echo ""
echo "=========================================="
echo "🚀 Starting FastAPI server on http://localhost:8000"
echo ""
echo "Available pages:"
echo "  • http://localhost:8000           → Login"
echo "  • http://localhost:8000/register  → Register"
echo "  • http://localhost:8000/dashboard → Video Upload & Dashboard"
echo "  • http://localhost:8000/history   → Job History"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

# Start uvicorn
exec uvicorn main:app --host 0.0.0.0 --port 8000 --reload
