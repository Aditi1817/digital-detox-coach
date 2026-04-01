#!/bin/bash
# Start OpenEnv API server in background
python openenv_api.py &
# Start Streamlit app
streamlit run app.py --server.port=8501 --server.address=0.0.0.0