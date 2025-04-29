from __future__ import division
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import torchvision
import io

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_image(image_file, transform=None, max_size=None, shape=None):
    image = Image.open(image_file).convert("RGB")

    if max_size:
        scale = max_size / max(image.size)
        size = np.array(image.size) * scale
        image = image.resize(size.astype(int), Image.Resampling.LANCZOS)

    if shape:
        image = image.resize(shape, Image.LANCZOS)

    if transform:
        image = transform(image).unsqueeze(0)

    return image.to(device)


class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.select = ['0', '5', '10', '19', '28']
        self.vgg = models.vgg19(pretrained=True).features

    def forward(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features


def style_transfer(content_img, style_img, style_weight=100, total_step=500, lr=0.003):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))])

    content = load_image(content_img, transform, max_size=400)
    style = load_image(style_img, transform, shape=[content.size(2), content.size(3)])
    target = content.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([target], lr=lr)
    vgg = VGGNet().to(device).eval()

    for step in range(total_step):
        target_features = vgg(target)
        content_features = vgg(content)
        style_features = vgg(style)

        style_loss = 0
        content_loss = 0
        for f1, f2, f3 in zip(target_features, content_features, style_features):
            content_loss += torch.mean((f1 - f2) ** 2)

            _, c, h, w = f1.size()
            f1 = f1.view(c, h * w)
            f3 = f3.view(c, h * w)

            gram_f1 = torch.mm(f1, f1.t())
            gram_f3 = torch.mm(f3, f3.t())

            style_loss += torch.mean((gram_f1 - gram_f3) ** 2) / (c * h * w)

        loss = content_loss + style_weight * style_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
    result = target.clone().squeeze()
    result = denorm(result).clamp_(0, 1)

    return result


# Streamlit interface
st.title("🎨 Neural Style Transfer with PyTorch")

content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
style_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

style_weight = st.slider("Style Weight", 10, 1000, 100)
steps = st.slider("Training Steps", 100, 2000, 500)
lr = st.slider("Learning Rate", 0.001, 0.01, 0.003, step=0.001)

if st.button("Apply Style Transfer"):
    if content_file and style_file:
        with st.spinner("Processing..."):
            output = style_transfer(content_file, style_file, style_weight, steps, lr)
            
            # Save only once after processing
            output_path = "styled_output.png"
            torchvision.utils.save_image(output, output_path)
            
            # Display the saved image
            st.image(output_path, caption="Stylized Image", use_column_width=True)
            st.success("Image stylized and saved successfully!")
    else:
        st.warning("Please upload both content and style images.")
