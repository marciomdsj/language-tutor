#!/bin/bash
# Launch the Language Tutor web UI on port 8888
cd "$(dirname "$0")"
source .venv/bin/activate
python -m streamlit run src/language_tutor/app.py --server.port 8888
