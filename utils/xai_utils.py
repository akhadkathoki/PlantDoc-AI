import torch
import numpy as np
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

def generate_gradcam(model, image, class_idx):

    image_tensor = transform(image).unsqueeze(0).to(device)

    target_layer = model.features[-1]

    cam = GradCAM(
        model=model,
        target_layers=[target_layer]
    )

    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    targets = [ClassifierOutputTarget(class_idx)]

    grayscale_cam = cam(
        input_tensor=image_tensor,
        targets=targets
    )

    grayscale_cam = grayscale_cam[0]

    image_np = np.array(image.resize((224,224))) / 255.0

    visualization = show_cam_on_image(
        image_np,
        grayscale_cam,
        use_rgb=True
    )

    return visualization, grayscale_cam

def calculate_severity(grayscale_cam):

    # Convert to numpy
    heatmap = np.array(grayscale_cam)

    # Severity = percentage of high activation
    threshold = 0.5
    infected_area = np.sum(heatmap > threshold)
    total_area = heatmap.size

    severity = (infected_area / total_area) * 100

    return round(severity, 2)