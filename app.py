import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# ----- Class names -----
class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# ----- Define your model -----
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, 5)  # Output: (12, 28, 28)
        self.pool = nn.MaxPool2d(2, 2)    # Output: (12, 14, 14)
        self.conv2 = nn.Conv2d(12, 24, 5) # Output: (24, 10, 10) → pool → (24, 5, 5)
        self.fc1 = nn.Linear(24 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ----- Load the trained model -----
net = NeuralNet()
net.load_state_dict(torch.load('trained_net.pth', map_location=torch.device('cpu')))
net.eval()

# ----- Image transforms -----
new_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ----- Streamlit UI -----
st.title("CIFAR-10 Image Classifier")
st.write("Upload a single image (32x32 or larger), and the model will predict its class.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "avif"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    image_tensor = new_transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = net(image_tensor)
        _, predicted = torch.max(output, 1)
        prediction = class_names[predicted.item()]

    # Display result
    st.success(f"Prediction: **{prediction}**")
