"""
main.py - Application entry point.
Run with:  python main.py
      or:  uvicorn main:app --reload
"""
import uvicorn
from app.api import app
from app.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level="info",
    )
