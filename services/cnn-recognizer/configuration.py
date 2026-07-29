import os
import torch

class Config:   
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Paths
    CURRENT_DIR = os.getcwd()
    print(f"Current directory: {CURRENT_DIR}")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    print(f"BASE_DIR: {BASE_DIR}")

    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
    TRAINED_MODELS_DIR = os.path.join(BASE_DIR, 'trained_models')
    TRAINED_MODEL_PATH = os.path.join(TRAINED_MODELS_DIR, 'digit_recognizer.pth')
        
    MANUAL_RECONGIZED_DATA_DIR = os.path.abspath(os.path.join(OUTPUT_DIR, 'manual_recongized_data'))
    WRONG_PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, 'wrong_predictions')
    VALIDATION_DIR = os.path.join(OUTPUT_DIR, 'validation')
    
    # Monitoring settings
    POLL_INTERVAL_SECONDS = 20 # 30
    MAX_HISTORY_SIZE = 1000
    MAX_ANOMALY_HISTORY_SIZE = 3

    # Create directories on startup
    @classmethod
    def create_directories(cls):
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.MANUAL_RECONGIZED_DATA_DIR, exist_ok=True)
        os.makedirs(cls.WRONG_PREDICTIONS_DIR, exist_ok=True)    
        os.makedirs(cls.VALIDATION_DIR, exist_ok=True)    