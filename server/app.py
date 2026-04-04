"""
Server entry point for OpenEnv Email Triage Environment.

This module provides the server entry point for multi-mode deployment.
It imports and configures the main FastAPI application.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the Python path so we can import from the root
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import the main FastAPI app
from app import app

def main():
    """Main entry point for the server."""
    import uvicorn
    
    port = int(os.getenv("PORT", "7860"))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"[Server] Starting OpenEnv Email Triage server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, reload=False)

if __name__ == "__main__":
    main()