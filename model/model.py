import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_weights.pth")

CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Consolidation",
    "Edema", "Fracture", "Lung Opacity", "Pleural Effusion",
    "Pneumonia", "Pneumothorax"
]

# Initialize the model
def initialize_model(num_classes=9, pretrained=False):
    model = models.efficientnet_v2_s(weights='IMAGENET1K_V1' if pretrained else None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def load_model():
    model = initialize_model()
    
    # Load weights and remove "module." prefixes if needed
    state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    
    if "module." in list(state_dict.keys())[0]:  # Check if keys contain "module."
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace("module.", "")] = v  # Remove "module." prefix
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Image preprocessing
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  

# Inference function
model = load_model()  

def predict(image: Image.Image):
    input_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(input_tensor)  # Get raw model outputs (logits)
        probs = torch.sigmoid(output).cpu().numpy().flatten()  # Convert to probabilities

    # Create dictionary mapping classes to probabilities
    prob_dict = {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(len(CLASS_NAMES))}

    return {"predictions": prob_dict}