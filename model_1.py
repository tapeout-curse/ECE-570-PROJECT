# Install necessary libraries
!pip install torch torchvision matplotlib tqdm kagglehub

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import kagglehub
import os
import io

# Download the Kodak dataset using kagglehub
path = kagglehub.dataset_download("sherylmehta/kodak-dataset")
print("Path to dataset files:", path)

# Custom dataset to load images without class subfolders
class KodakDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0  # Placeholder for label

# Function to preprocess and load the Kodak dataset
def load_kodak_dataset(batch_size=8):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    dataset = KodakDataset(root_dir=path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# Transformer-based image compression model
class TransformerCompressionModel(nn.Module):
    def __init__(self, input_dim=256, num_heads=4, num_layers=2):
        super(TransformerCompressionModel, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=input_dim, nhead=num_heads),
            num_layers=num_layers
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded, encoded)
        return decoded

# Standard Compression Techniques
def compress_with_standard_techniques(image, quality=50):
    compressed_images = {}

    # Compress with JPEG
    jpeg_image = io.BytesIO()
    image.save(jpeg_image, format='JPEG', quality=quality)
    jpeg_image_size = len(jpeg_image.getvalue())
    jpeg_image = Image.open(io.BytesIO(jpeg_image.getvalue()))
    compressed_images['JPEG'] = (jpeg_image, jpeg_image_size)

    # Compress with PNG
    png_image = io.BytesIO()
    image.save(png_image, format='PNG')
    png_image_size = len(png_image.getvalue())
    png_image = Image.open(io.BytesIO(png_image.getvalue()))
    compressed_images['PNG'] = (png_image, png_image_size)

    # Compress with WebP
    webp_image = io.BytesIO()
    image.save(webp_image, format='WEBP', quality=quality)
    webp_image_size = len(webp_image.getvalue())
    webp_image = Image.open(io.BytesIO(webp_image.getvalue()))
    compressed_images['WebP'] = (webp_image, webp_image_size)

    return compressed_images

# Visualization function to compare ML-based and standard techniques
def visualize_compression(model, dataloader, quality=50):
    model.eval()
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.view(images.size(0), -1, 256).to(device)
            outputs = model(images)
            images_np = images.cpu().view(-1, 128, 128).numpy()
            outputs_np = outputs.cpu().view(-1, 128, 128).numpy()

            fig, axes = plt.subplots(3, len(images), figsize=(12, 8))

            for i in range(len(images)):
                # Original image
                original_image = transforms.ToPILImage()(images[i].cpu().view(3, 128, 128))
                compressed_images = compress_with_standard_techniques(original_image, quality)

                # Original
                axes[0, i].imshow(images_np[i], cmap='gray')
                axes[0, i].set_title('Original')
                axes[0, i].axis('off')

                # Compressed with ML model
                axes[1, i].imshow(outputs_np[i], cmap='gray')
                axes[1, i].set_title('ML Compressed')
                axes[1, i].axis('off')

                # Standard compression comparisons
                for idx, (comp_name, (comp_image, comp_size)) in enumerate(compressed_images.items()):
                    # Convert PIL Image to numpy
                    comp_image_np = np.array(comp_image.convert("L"))

                    # Plot compressed image
                    axes[2 + idx, i].imshow(comp_image_np, cmap='gray')
                    axes[2 + idx, i].set_title(f"{comp_name} (size: {comp_size/1024:.1f} KB)")
                    axes[2 + idx, i].axis('off')

            plt.tight_layout()
            plt.show()
            break  # Only visualize one batch

# Initialize model, device, and data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerCompressionModel().to(device)
dataloader = load_kodak_dataset()

# Train the model
train_model(model, dataloader, num_epochs=5)

# Visualize the results with comparison to standard compression techniques
visualize_compression(model, dataloader)
