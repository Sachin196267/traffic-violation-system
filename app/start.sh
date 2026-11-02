#!/bin/bash
# Start the platform: builds the web app if needed, then serves everything on http://127.0.0.1:8600
cd "$(dirname "$0")" || exit 1
[ -f web/dist/index.html ] || (cd web && npm install && npm run build)
cd server && exec ../../.venv/bin/uvicorn main:app --host 127.0.0.1 --port "${PORT:-8600}"
