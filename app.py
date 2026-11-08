import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.set_page_config(page_title="Skin Lesion AI", page_icon="🩺")
st.title("Skin Tag & Lesion Diagnostic AI")

@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 7)
    model.load_state_dict(torch.load('skin_tag_resnet50.pth', map_location=device))
    model.eval()
    return model

model = load_model()
classes = [
    'Benign Keratosis / Skin Tag', 
    'Melanocytic Nevus (Common Mole)', 
    'Melanoma', 
    'Basal Cell Carcinoma', 
    'Actinic Keratosis', 
    'Vascular Lesion', 
    'Dermatofibroma'
]

uploaded_file = st.file_uploader("Upload a skin image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, width=350)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        outputs = model(transform(image).unsqueeze(0))
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        
    st.subheader("Analysis Results:")
    results = {classes[i]: float(probs[i]) for i in range(7)}
    
    for cls, prob in sorted(results.items(), key=lambda x: x[1], reverse=True):
        st.write(f"**{cls}**: {prob*100:.1f}%")
        st.progress(float(prob))