#   run_api.py
import os
import uvicorn

if __name__ == "__main__":
    # Read the PORT environment variable (Cloud Run sets this to 8080)
    port = int(os.environ.get("PORT", 8080))
    
    # Make sure to bind to 0.0.0.0 to listen on all interfaces
    uvicorn.run("api.app:app", host="0.0.0.0", port=port, reload=False)