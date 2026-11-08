import os
import uvicorn
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import yaml
import logging

from .service import get_model_service, model_service
from .endpoints import router as api_router

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

config_path = Path(__file__).parent.parent.parent / 'config' / 'config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

app = FastAPI(
    title="GNSS Error Prediction API",
    description="API for predicting GNSS satellite clock and ephemeris errors using deep learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    """Initialize the application."""
    try:
        model_path = Path(config['api']['model_path'])
        processor_path = Path(config['api']['processor_path'])
        
        if not model_path.exists() or not processor_path.exists():
            logger.warning(
                "Model or processor not found. "
                f"Model path: {model_path}, exists: {model_path.exists()}\n"
                f"Processor path: {processor_path}, exists: {processor_path.exists()}"
            )
            return
            
        model_service.load_model(model_path, processor_path)
        logger.info("Model and processor loaded successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}", exc_info=True)
        raise

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "GNSS Error Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }

def run():
    """Run the API server."""
    uvicorn.run(
        "src.api.main:app",
        host=config['api']['host'],
        port=config['api']['port'],
        reload=config['api'].get('debug', False),
        workers=config['api'].get('workers', 1)
    )

if __name__ == "__main__":
    run()