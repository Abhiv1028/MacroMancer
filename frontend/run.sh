#!/bin/bash
# Launch the Macromancer Streamlit frontend.
# Backend base URL can be overridden: MACROMANCER_API_URL=http://host:8000 ./frontend/run.sh
cd "$(dirname "$0")/.."
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0
