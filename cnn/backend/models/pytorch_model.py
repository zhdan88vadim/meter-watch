import os
import torch
from configuration import Config
from models.digit_recognizer import DigitRecognizer
from utils.heatmap import SimpleHeatmap
from utils.preprocessing import prepare_for_model

# Global model variable
pytorch_model = None
heatmap_gen = None
device = Config.DEVICE

def load_pytorch_model():
    """Load the trained PyTorch model"""
    global pytorch_model
    global heatmap_gen

    if os.path.exists(Config.MODEL_PATH):
        try:
            pytorch_model = DigitRecognizer().to(Config.DEVICE)
            pytorch_model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
            pytorch_model.eval()
            heatmap_gen = SimpleHeatmap(pytorch_model)
            print(f"Model loaded: {Config.MODEL_PATH}")
            return True
        except Exception as e:
            print(f"⚠️ Could not load {Config.MODEL_PATH}: {e}")
    
    print("❌ No PyTorch model found. Using fallback methods.")
    return False


def predict_digit(roi):
    global pytorch_model
    global heatmap_gen

    prepared, resized_square = prepare_for_model(roi)
    tensor = torch.FloatTensor(prepared).unsqueeze(0).unsqueeze(0).to(Config.DEVICE)
    
    with torch.no_grad():
        output = pytorch_model(tensor)
        probs = torch.softmax(output, dim=1)
        confidence = probs[0].max().item()
        prediction = output.argmax(dim=1).item()
    
    # Generating heat maps
    cam, _, _ = heatmap_gen.generate(tensor)
    saliency, _, _ = heatmap_gen.generate_saliency(tensor)
    
    return prediction, confidence, cam, saliency, resized_square, prepared