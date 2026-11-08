import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import joblib

from src.models.advanced_model import MultiHorizonTransformer, load_model
from src.data.advanced_processor import AdvancedGNSSProcessor

logger = logging.getLogger(__name__)

class ModelService:
    """
    Service class for handling model predictions.
    """
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = None
        self.ready = False
    
    def load_model(self, model_path: str, processor_path: str) -> None:
        """
        Load the trained model and processor.
        
        Args:
            model_path: Path to the trained model checkpoint
            processor_path: Path to the saved processor
        """
        try:
            logger.info(f"Loading model from {model_path}")
            
            self.model = load_model(model_path)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Loading processor from {processor_path}")
            self.processor = joblib.load(processor_path)
            
            self.ready = True
            logger.info("Model and processor loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.ready = False
            raise
    
    async def predict(
        self, 
        input_data: List[Dict[str, float]],
        prediction_horizon: int = 15
    ) -> Dict[str, Any]:
        """
        Make predictions for the given input data.
        
        Args:
            input_data: List of dictionaries containing input features
            prediction_horizon: Prediction horizon in minutes
            
        Returns:
            Dictionary containing predictions and metadata
        """
        if not self.ready:
            raise RuntimeError("Model is not loaded")
        
        try:
            import pandas as pd
            df = pd.DataFrame(input_data)
            
            X_processed, _ = self.processor.transform(df)
            
            X_tensor = torch.FloatTensor(X_processed).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(X_tensor)
                
                horizon = str(prediction_horizon)
                if horizon not in outputs:
                    raise ValueError(
                        f"Prediction horizon {prediction_horizon} not supported. "
                        f"Available horizons: {list(outputs.keys())}"
                    )
                
                pred = outputs[horizon]
                mean = pred['mean'].cpu().numpy()[0]
                std = pred['std'].cpu().numpy()[0]
                
                mean = self.processor.inverse_transform_targets(mean.reshape(1, -1))[0]
                
                target_columns = self.processor.target_columns
                prediction = dict(zip(target_columns, mean))
                uncertainty = dict(zip(
                    [f"{col}_std" for col in target_columns],
                    std
                ))
                
                return {
                    "prediction": prediction,
                    "uncertainty": uncertainty,
                    "metadata": {
                        "model": "MultiHorizonTransformer",
                        "horizon_minutes": prediction_horizon,
                        "timestamp": pd.Timestamp.now().isoformat()
                    }
                }
                
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}", exc_info=True)
            raise
    
    def is_ready(self) -> bool:
        """Check if the model is ready for predictions."""
        return self.ready

model_service = ModelService()

def get_model_service() -> ModelService:
    """Get the global model service instance."""
    return model_service
