import cv2
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import models, transforms

def generate_grad_cam(model, image, class_index):
    model.eval()

    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = preprocess(image).unsqueeze(0)

    # Forward pass
    image.requires_grad_()
    output = model(image)
    prediction_score = output[0, class_index]

    # Backpropagate the gradients
    model.zero_grad()
    prediction_score.backward()

    # Get the gradients from the image
    gradients = image.grad.data.squeeze().cpu().numpy()

    # Compute the feature maps weights as the global average pooling of the gradients
    weights = np.mean(gradients, axis=(1, 2))

    # Get the feature maps from the last convolutional layer
    feature_maps = model.get_activation_maps(image).squeeze()

    # Generate the class activation map (CAM)
    cam = np.sum(np.multiply(weights, feature_maps), axis=0)
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (image.size(-1), image.size(-2)))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    # Apply colormap to the CAM
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    # Overlay the heatmap on the original image
    original_image = np.array(transforms.functional.to_pil_image(image.squeeze().cpu()))
    output_image = cv2.addWeighted(cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR), 0.8, heatmap, 0.4, 0)

    return output_image

# Load your trained PyTorch model
model = torch.load('logs/logo/efficientdet-d0_90_3000.pth')
model.eval()

# Load and preprocess an example image
image_path = 'datasets/logo/test/0203_10_jpg.rf.50d1b2e10f87fb2e83649c725132f33a.jpg'  # Provide the path to your own image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Generate the Grad-CAM visualization for a specific class index
class_index = 10  # Example class index
grad_cam_image = generate_grad_cam(model, image, class_index)

# Display the original image and the Grad-CAM visualization
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(grad_cam_image)
plt.title('Grad-CAM')

plt.tight_layout()
plt.show()
