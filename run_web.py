from __future__ import annotations

import uvicorn
from web.app import app

if __name__ == "__main__":
    print("Starting Bus Delay Predictor web app...")
    print("Open http://localhost:8000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8000)
