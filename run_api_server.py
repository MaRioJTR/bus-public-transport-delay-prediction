from __future__ import annotations

import uvicorn
from api_server import app

if __name__ == "__main__":
    print("starting server...")
    print("pop here ->  http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
