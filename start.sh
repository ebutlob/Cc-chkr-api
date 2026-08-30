#!/bin/bash
exec uvicorn api_server:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 60
