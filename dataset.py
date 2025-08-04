import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# Color dictionary used for conditioning
COLOR_DICT = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "magenta": (255, 0, 255),
    "cyan": (0, 255, 255),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128)
}

class PolygonDataset(Dataset):
    def __init__(self, input_dir, output_dir, json_path, transform=None):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.transform = transform if transform else T.Compose([
            T.Resize((128, 128)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_path = os.path.join(self.input_dir, item["input_polygon"])
        output_path = os.path.join(self.output_dir, item["output_image"])
        color_name = item["colour"]

        # Load input polygon image (grayscale)
        input_img = Image.open(input_path).convert("L")
        target_img = Image.open(output_path).convert("RGB")

        # Transform images
        input_tensor = self.transform(input_img)      # Shape: [1, 128, 128]
        target_tensor = self.transform(target_img)    # Shape: [3, 128, 128]

        # Create color tensor from color name
        rgb = torch.tensor([v / 255.0 for v in COLOR_DICT[color_name]])
        color_tensor = rgb.view(3, 1, 1).expand(-1, 128, 128)  # Shape: [3, 128, 128]

        # Concatenate grayscale + RGB → 4 channel input
        conditioned_input = torch.cat([input_tensor, color_tensor], dim=0)  # Shape: [4, 128, 128]

        mask = (input_tensor[0] > 0).float().unsqueeze(0)  # Shape: [1, 128, 128]

        return conditioned_input, target_tensor, mask
