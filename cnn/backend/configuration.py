import os
import torch

class Config:
    CURRENT_DIR = os.getcwd()
    print(f"Current directory: {CURRENT_DIR}")
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    IMAGE_SIZE = 28
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    print(f"BASE_DIR: {BASE_DIR}")

    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    MODEL_PATH = os.path.join(MODELS_DIR, 'digit_recognizer.pth')
        
    MANUAL_RECONGIZED_DATA_DIR = os.path.abspath(os.path.join(OUTPUT_DIR, 'manual_recongized_data'))
    print(f"MANUAL_RECONGIZED_DATA_DIR: {MANUAL_RECONGIZED_DATA_DIR}")

    WRONG_PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, 'wrong_predictions')
    VALIDATION_DIR = os.path.join(OUTPUT_DIR, 'validation')
    
    # Monitoring settings
    POLL_INTERVAL_SECONDS = 20 # 30
    MAX_HISTORY_SIZE = 1000

    # Create directories on startup
    @classmethod
    def create_directories(cls):
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.MANUAL_RECONGIZED_DATA_DIR, exist_ok=True)
        os.makedirs(cls.WRONG_PREDICTIONS_DIR, exist_ok=True)    
        os.makedirs(cls.VALIDATION_DIR, exist_ok=True)    