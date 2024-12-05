import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.metrics import structural_similarity as compare_ssim, peak_signal_noise_ratio as compare_psnr
from skimage.transform import resize
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import kagglehub

# Step 1: Download the Kodak dataset
def download_kodak_dataset():
    print("Downloading Kodak dataset...")
    path = kagglehub.dataset_download("sherylmehta/kodak-dataset")
    print("Path to dataset files:", path)
    return path

# Step 2: Load images and resize to 256x256
def load_kodak_images(path):
    images = []
    for filename in os.listdir(path):
        if filename.endswith(".png"):
            img = imread(os.path.join(path, filename))
            img = resize(img, (256, 256), anti_aliasing=True)
            images.append(img)
    return np.array(images)

# Step 3: Build a CNN-based Compression Model
def build_compression_cnn_model():
    input_layer = Input(shape=(256, 256, 3))

    # Encoder
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    x2 = Conv2D(32, (3, 3), activation='relu', padding='same')(x1)
    bottleneck = Conv2D(8, (3, 3), activation='relu', padding='same')(x2)

    # Decoder
    x3 = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(bottleneck)
    x4 = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x3)
    output_layer = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(Add()([x4, x1]))

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Step 4: Train the CNN Model
def train_cnn_model(model, images, epochs=50, batch_size=8):
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    model.fit(images, images, epochs=epochs, batch_size=batch_size, callbacks=[lr_scheduler])
    return model

# Step 5: Evaluate Compression with Varying Bit Depths
def evaluate_compression_bit_depth(images, model, bit_depths):
    psnr_values = []
    ssim_values = []

    for bit_depth in bit_depths:
        current_psnr = []
        current_ssim = []

        for img in images:
            # Compress image with CNN
            compressed_img = model.predict(img[np.newaxis, ...])[0]

            # Quantize image to given bit depth
            quantized_img = np.round(compressed_img * (2 ** bit_depth - 1)) / (2 ** bit_depth - 1)

            # PSNR
            psnr_value = compare_psnr(img, quantized_img, data_range=1.0)
            current_psnr.append(psnr_value)

            # SSIM
            ssim_value = compare_ssim(
                img, quantized_img, multichannel=True, data_range=1.0, win_size=7, channel_axis=-1
            )
            current_ssim.append(ssim_value)

        # Append average values for this bit depth
        psnr_values.append(np.mean(current_psnr))
        ssim_values.append(np.mean(current_ssim))

    return psnr_values, ssim_values

# Step 6: Plot PSNR and SSIM vs Bit Depth
def plot_psnr_ssim_vs_bit_depth(bit_depths, psnr_values, ssim_values):
    plt.figure(figsize=(12, 6))

    # Plot PSNR vs Bit Depth
    plt.subplot(1, 2, 1)
    plt.plot(bit_depths, psnr_values, marker='o', color='b', label='PSNR')
    plt.xlabel("Bit Depth (Bits Per Pixel)")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR vs Bit Depth")
    plt.grid(True)
    plt.legend()

    # Plot SSIM vs Bit Depth
    plt.subplot(1, 2, 2)
    plt.plot(bit_depths, ssim_values, marker='o', color='r', label='SSIM')
    plt.xlabel("Bit Depth (Bits Per Pixel)")
    plt.ylabel("SSIM")
    plt.title("SSIM vs Bit Depth")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


# Execute the Pipeline
dataset_path = download_kodak_dataset()
images = load_kodak_images(dataset_path)

cnn_model = build_compression_cnn_model()
cnn_model = train_cnn_model(cnn_model, images)

# Define bit depths for compression (2, 4, 6, 8 bits)
bit_depths = [2, 4, 6, 8]

# Evaluate PSNR and SSIM for varying bit depths
psnr_values, ssim_values = evaluate_compression_bit_depth(images, cnn_model, bit_depths)

# Plot the results
plot_psnr_ssim_vs_bit_depth(bit_depths, psnr_values, ssim_values)
