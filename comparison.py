import cv2
import numpy as np
from io import BytesIO
from PIL import Image

# Function to compress an image using JPEG
def jpeg_compression(image, quality=90):
    img_np = np.array(image)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, img_encoded = cv2.imencode('.jpg', img_np, encode_param)
    img_decompressed = cv2.imdecode(img_encoded, cv2.IMREAD_COLOR)
    return img_decompressed

# Function for PNG compression
def png_compression(image):
    img_np = np.array(image)
    encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]  # Max compression level
    _, img_encoded = cv2.imencode('.png', img_np, encode_param)
    img_decompressed = cv2.imdecode(img_encoded, cv2.IMREAD_COLOR)
    return img_decompressed

# Function for WebP compression
def webp_compression(image, quality=80):
    img_np = np.array(image)
    encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
    _, img_encoded = cv2.imencode('.webp', img_np, encode_param)
    img_decompressed = cv2.imdecode(img_encoded, cv2.IMREAD_COLOR)
    return img_decompressed

# Function for basic Pillow compression
def pillow_compression(image, quality=70):
    output = BytesIO()
    image.save(output, format='JPEG', quality=quality)
    output.seek(0)
    return Image.open(output)

# Update the visualization function to include all compression methods
def visualize_compression_all_methods(model, dataloader):
    model.eval()
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.view(images.size(0), -1, 256).to(device)
            outputs = model(images)
            images_np = images.cpu().view(-1, 128, 128, 3).numpy()
            outputs_np = outputs.cpu().view(-1, 128, 128, 3).numpy()

            # Apply compression algorithms
            jpeg_images = [jpeg_compression(image) for image in images_np]
            png_images = [png_compression(image) for image in images_np]
            webp_images = [webp_compression(image) for image in images_np]
            pillow_images = [np.array(pillow_compression(Image.fromarray(image.astype('uint8')))) for image in images_np]

            # Plot original and compressed images
            fig, axes = plt.subplots(6, len(images), figsize=(15, 20))
            titles = ['Original', 'ML Compressed', 'JPEG', 'PNG', 'WebP', 'Pillow']

            for i in range(len(images)):
                for j, img in enumerate([images_np[i], outputs_np[i], jpeg_images[i], png_images[i], webp_images[i], pillow_images[i]]):
                    axes[j, i].imshow(img)
                    axes[j, i].set_title(titles[j])
                    axes[j, i].axis('off')

            plt.tight_layout()
            plt.show()
            break  # Only visualize one batch

# Call the updated visualization function
visualize_compression_all_methods(model, dataloader)
