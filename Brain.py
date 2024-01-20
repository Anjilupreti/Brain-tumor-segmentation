import streamlit as st
import torch
from PIL import Image
import numpy as np
from torchvision import transforms

# Define the U-Net model
class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Define your U-Net layers here

    def forward(self, x):
        # Implement the forward pass for your U-Net
        return x

# Function to perform brain tumor segmentation using the U-Net model
def segment_tumor(image_path, model):
    # Load and preprocess the input image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((256, 256))  # Resize for the U-Net model (adjust the size accordingly)
    img_tensor = transforms.ToTensor()(img).unsqueeze(0)  # Convert to PyTorch tensor

    # Perform segmentation using the loaded U-Net model
    with torch.no_grad():
        segmentation_map = model(img_tensor)

    # Convert the segmentation map to an RGB image with white color in the segmented region
    segmented_mask = (segmentation_map > 0.5).squeeze().numpy().astype(np.uint8) * 255
    segmented_rgb = np.zeros_like(img, dtype=np.uint8)
    segmented_rgb[segmented_mask[0] > 0, :] = [255, 255, 255]  # Set RGB values to white in the segmented region
    segmented_image = Image.fromarray(segmented_rgb)

    # Resize the segmented image to match the original image size
    segmented_image = segmented_image.resize(img.size)

    return segmented_image

# Load the pre-trained U-Net model
model_path = 'unet_model.pth'
model = UNet()
model(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Streamlit UI
st.title('Brain Tumor Segmentation with U-Net (PyTorch)')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=False)

    # Perform segmentation on the uploaded image
    if st.button('Segment Tumor'):
        segmented_image = segment_tumor(uploaded_file, model)
        st.image(segmented_image, caption='Segmented Tumor.', use_column_width=True)
