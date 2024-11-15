import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st

# Define the CNN model class
class FashionMNIST_CNN(nn.Module):
    def __init__(self):
        super(FashionMNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64 * 6 * 6)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained model
model = FashionMNIST_CNN()
model.load_state_dict(torch.load("fashion_mnist_cnn.pth", map_location=torch.device('cpu')))
model.eval()

# Define transforms for input images
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Streamlit App
st.title("Fashion MNIST Classifier")
st.write("Upload an image to classify it into one of the Fashion MNIST categories.")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_tensor = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output.data, 1)

    # Map the prediction to the class labels
    classes = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot'
    ]
    predicted_class = classes[predicted.item()]

    # Display the prediction
    st.write(f"Prediction: **{predicted_class}**")
