from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import logging

from .service import get_model_service, ModelService

logger = logging.getLogger(__name__)
router = APIRouter()

class PredictionRequest(BaseModel):
    """Request model for predictions."""
    sequence: List[Dict[str, float]] = Field(
        ...,
        description="List of input feature dictionaries. Each dictionary should contain the same set of keys."
    )
    prediction_horizon: int = Field(
        default=15,
        description="Prediction horizon in minutes. Must be one of the configured horizons."
    )

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: Dict[str, float]
    uncertainty: Dict[str, float]
    metadata: Dict[str, Any]

@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    model_service: ModelService = Depends(get_model_service)
):
    """
    Make predictions for GNSS errors.
    
    This endpoint accepts a sequence of GNSS measurements and returns
    predictions for the specified time horizon.
    
    Example request:
    ```json
    {
        "sequence": [
            {"clock_error": 0.1, "x_error": 0.2, "y_error": 0.3, "z_error": 0.4},
            {"clock_error": 0.15, "x_error": 0.25, "y_error": 0.35, "z_error": 0.45}
        ],
        "prediction_horizon": 15
    }
    ```
    """
    try:
        if not model_service.is_ready():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not ready"
            )
        
        sequence_length = len(request.sequence)
        if sequence_length < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Sequence must contain at least one item"
            )
        
        first_item_keys = set(request.sequence[0].keys())
        for i, item in enumerate(request.sequence[1:], 1):
            if set(item.keys()) != first_item_keys:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"All items in sequence must have the same keys. "
                          f"Item 0 has keys: {sorted(first_item_keys)}, "
                          f"but item {i} has keys: {sorted(item.keys())}"
                )
        
        result = await model_service.predict(
            request.sequence,
            request.prediction_horizon
        )
        
        return result
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making prediction: {str(e)}"
        )

@router.get("/health")
async def health_check(model_service: ModelService = Depends(get_model_service)):
    """
    Health check endpoint.
    
    Returns the status of the API and whether the model is ready.
    """
    return {
        "status": "healthy",
        "model_ready": model_service.is_ready(),
        "device": str(model_service.device) if model_service.is_ready() else None
    }

@router.get("/info")
async def model_info(model_service: ModelService = Depends(get_model_service)):
    """
    Get information about the loaded model.
    
    Returns model configuration and statistics.
    """
    if not model_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready"
        )
    
    return {
        "model_type": "MultiHorizonTransformer",
        "input_features": model_service.processor.target_columns if model_service.processor else None,
        "target_features": model_service.processor.target_columns if model_service.processor else None,
        "device": str(model_service.device),
        "prediction_horizons": model_service.model.config.get('prediction_horizons', []) if model_service.model else None
    }
