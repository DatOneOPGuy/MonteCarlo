#!/bin/bash
# Launcher script for Monte Carlo API and UI services

# Start FastAPI server in background
echo "Starting FastAPI server..."
uvicorn montecarlo.api:app --reload --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait a moment for API to start
sleep 2

# Start Streamlit UI
echo "Starting Streamlit UI..."
streamlit run ui/streamlit_app.py

# Cleanup: kill API when Streamlit exits
kill $API_PID 2>/dev/null

