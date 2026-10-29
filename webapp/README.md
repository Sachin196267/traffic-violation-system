# Traffic Violation System — Web Application

Full-stack web interface for the Mumbai ITMS Traffic Violation Detection & e-Challan system.

## Features

- **Authentication**: Secure JWT-based login and registration with bcrypt password hashing
- **Video Upload**: Upload local MP4/AVI/MOV files for processing
- **URL Download**: Submit YouTube or video URLs (yt-dlp download + processing)
- **Real-time Progress**: Live job status polling with progress bars
- **Results Dashboard**: 
  - Side-by-side composite video (camera + digital twin)
  - Generated PDF e-Challans with evidence gallery
  - XAI Grad-CAM heatmaps and severity scores
  - ANPR plate recognition results
- **Job History**: Per-user job tracking and results archive
- **Background Processing**: ThreadPoolExecutor-based async job queue

## Quick Start

### 1. Install Dependencies

```bash
cd webapp/
pip install -r requirements.txt
```

### 2. Start the Server

```bash
./start.sh
```

Or directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The application will start at **http://localhost:8000**

### 3. Use

1. Navigate to http://localhost:8000/register — create an account
2. Login at http://localhost:8000
3. Upload a video or submit a URL from the dashboard
4. View results at http://localhost:8000/history

## API Documentation

Available at **http://localhost:8000/docs** (Swagger UI)

## Directory Structure

```
webapp/
├── main.py                 # FastAPI entry point
├── auth.py                 # JWT authentication
├── database.py             # SQLite queries
├── jobs.py                 # Job queue management
├── pipeline_runner.py      # Pipeline wrapper
├── start.sh                # Startup script
├── static/                 # CSS + JS
├── templates/              # HTML pages
├── jobs/                   # Per-job output (runtime)
└── traffic_violations.db   # SQLite DB (runtime)
```

## Troubleshooting

**"Module not found" errors**: Run `pip install -r requirements.txt` from project root first, then `pip install -r webapp/requirements.txt`

**Database locked**: Use SQLite in WAL mode (default)

**Jobs stuck in processing**: Check terminal for Python exceptions — pipeline is CPU/GPU intensive