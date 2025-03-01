# inference.py (Modified)
from detection import load_detection_model, detect_tumor
from PIL import Image
import numpy as np

detection_model = load_detection_model()

def perform_inference(image_path):
    """Performs detection on an image."""
    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        return {"error": "Image not found"}
    except Exception as e:
        return {"error": f"Error opening image: {e}"}

    if detection_model:
        detection_result = detect_tumor(image, detection_model)
        return {"detection": detection_result.tolist()}
    else:
        return {"error": "Detection model not loaded"}

if __name__ == "__main__":
    # Example usage for testing
    result = perform_inference("C:/Users/Yadav Ji/Desktop/Y1.jpg")  # Replace with your test image
    print(result)