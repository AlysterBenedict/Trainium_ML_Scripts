# biometric_estimator.py

import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import rembg
import numpy as np
from PIL import Image
import io

# --- Model Architecture Definition ---
class BodyM_MetricEstimator(nn.Module):
    def __init__(self, num_measurements):
        super(BodyM_MetricEstimator, self).__init__()
        # Initialize two separate instances of the model
        self.frontal_branch = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        self.side_branch = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)

        num_features = self.frontal_branch.classifier[1].in_features
        self.frontal_branch.classifier = nn.Identity()
        self.side_branch.classifier = nn.Identity()

        self.regression_head = nn.Sequential(
            nn.Linear(num_features * 2, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_measurements)
        )

    def forward(self, frontal_img, side_img):
        frontal_features = self.frontal_branch(frontal_img)
        side_features = self.side_branch(side_img)
        combined_features = torch.cat((frontal_features, side_features), dim=1)
        predictions = self.regression_head(combined_features)
        return predictions

class BiometricEstimator:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model = BodyM_MetricEstimator(num_measurements=15)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        self.image_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.target_columns = [
            'height_cm', 'weight_kg', 'ankle', 'arm-length', 'bicep', 'calf', 'chest',
            'forearm', 'hip', 'leg-length', 'shoulder-breadth', 'shoulder-to-crotch',
            'thigh', 'waist', 'wrist'
        ]

    def _process_image_to_silhouette(self, image_data, output_size=(224, 224)):
        """Internal helper to process a single image."""
        try:
            output_rgba_data = rembg.remove(image_data)
            output_rgba = Image.open(io.BytesIO(output_rgba_data))
            output_np = np.array(output_rgba)

            alpha_channel = output_np[:, :, 3]
            _, silhouette = cv2.threshold(alpha_channel, 50, 255, cv2.THRESH_BINARY)
            silhouette_resized = cv2.resize(silhouette, output_size, interpolation=cv2.INTER_AREA)
            return silhouette_resized
        except Exception as e:
            print(f"Error processing image: {e}")
            return None

    def predict(self, frontal_image_data, side_image_data):
        frontal_silhouette = self._process_image_to_silhouette(frontal_image_data)
        side_silhouette = self._process_image_to_silhouette(side_image_data)

        if frontal_silhouette is None or side_silhouette is None:
            raise ValueError("Could not process one or both images.")

        frontal_tensor = self.image_transforms(frontal_silhouette).unsqueeze(0).to(self.device)
        side_tensor = self.image_transforms(side_silhouette).unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.model(frontal_tensor, side_tensor)

        predictions_np = predictions.squeeze(0).cpu().numpy()
        results = {name: float(value) for name, value in zip(self.target_columns, predictions_np)}
        return results