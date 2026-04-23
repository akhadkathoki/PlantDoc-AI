import torch
import numpy as np
from torchvision import transforms
import torchvision.models as models

# Device
device = torch.device("cpu")

# Class names (38 classes)
class_names = [
'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy',
'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
'Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot',
'Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
'Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy',
'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight',
'Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy',
'Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy',
'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold',
'Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite',
'Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus',
'Tomato___Tomato_mosaic_virus','Tomato___healthy'
]

# Transform (ONLY ONCE)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# Load Model
def load_model():

    model = models.efficientnet_b0(weights=None)

    model.classifier[1] = torch.nn.Linear(1280, 38)

    model.load_state_dict(
        torch.load("models/plant_disease_model_final.pth", map_location=device)
    )

    model.to(device)
    model.eval()

    return model


# Predict Function (CLEAN VERSION)
def predict(image, model):

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)

        top3_prob, top3_idx = torch.topk(probs, 3)

    results = []

    for i in range(3):
        disease = class_names[top3_idx[0][i]]
        confidence = float(top3_prob[0][i])
        results.append((disease, confidence))

    return results