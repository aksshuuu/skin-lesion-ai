import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.set_page_config(page_title="Skin Tag AI Screener", page_icon="+", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background-color: #f7fafc;
        color: #2d3748;
    }
    .main-title {
        color: #2c7a7b;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .sub-text {
        text-align: center;
        color: #4a5568;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .medical-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #319795;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

classes = [
    'Benign Keratosis / Skin Tag',
    'Melanocytic Nevus (Common Mole)',
    'Melanoma',
    'Basal Cell Carcinoma',
    'Actinic Keratosis',
    'Vascular Lesion',
    'Dermatofibroma'
]

DISEASE_INFO = {
    'Benign Keratosis / Skin Tag': {
        'risk_level': 'Low Risk (Benign)',
        'description': 'These are common, harmless skin growths like skin tags or seborrheic keratoses.',
        'malignancy': 'They do not turn into malignant cancer. They are purely benign.',
        'next_steps': 'No immediate action is needed. If it gets caught on clothing, bleeds, or bothers you cosmetically, a doctor can easily remove it in minutes.',
        'alert_type': 'success'
    },
    'Melanocytic Nevus (Common Mole)': {
        'risk_level': 'Usually Benign',
        'description': 'This is a standard mole caused by a cluster of pigment-forming cells (melanocytes).',
        'malignancy': 'While typically harmless, a very small percentage of moles can change into melanoma (skin cancer) over time.',
        'next_steps': 'Monitor it for the ABCDEs (Asymmetry, Border irregularity, Color changes, Diameter > 6mm, Evolving). If it starts changing shape or itching, visit a dermatologist.',
        'alert_type': 'info'
    },
    'Melanoma': {
        'risk_level': 'High Risk (Malignant)',
        'description': 'Melanoma is the most serious type of skin cancer, developing in the cells that produce melanin.',
        'malignancy': 'This is an active, aggressive cancer that can spread to other organs if not caught early.',
        'next_steps': 'Please do not panic, but do schedule an appointment with a dermatologist immediately for a professional biopsy. Early detection is highly treatable.',
        'alert_type': 'error'
    },
    'Basal Cell Carcinoma': {
        'risk_level': 'High Risk (Malignant)',
        'description': 'A type of skin cancer that most often develops on areas of skin exposed to the sun, like the face and neck.',
        'malignancy': 'It is malignant but tends to grow very slowly. It rarely spreads (metastasizes) to other parts of the body, but can invade surrounding local tissue.',
        'next_steps': 'Schedule a visit with a doctor. It is highly treatable with a simple in-office surgical removal, but it should not be ignored.',
        'alert_type': 'error'
    },
    'Actinic Keratosis': {
        'risk_level': 'Moderate Risk (Pre-cancerous)',
        'description': 'A rough, scaly patch on the skin that develops from years of sun exposure.',
        'malignancy': 'This is considered pre-cancerous. If left untreated, a small percentage can turn into Squamous Cell Carcinoma (a type of skin cancer).',
        'next_steps': 'Book a routine appointment with a dermatologist. They can easily freeze it off or prescribe a topical cream to prevent it from becoming cancerous.',
        'alert_type': 'warning'
    },
    'Vascular Lesion': {
        'risk_level': 'Low Risk (Benign)',
        'description': 'These are common red or purple marks (like cherry angiomas) caused by broken or clustered blood vessels.',
        'malignancy': 'They are entirely benign and do not turn into cancer.',
        'next_steps': 'No medical treatment is required. If they bleed frequently or you dislike how they look, a doctor can safely remove them with a laser.',
        'alert_type': 'success'
    },
    'Dermatofibroma': {
        'risk_level': 'Low Risk (Benign)',
        'description': 'A common, harmless nodule that feels like a hard lump under the skin, often occurring after a minor injury like a bug bite.',
        'malignancy': 'It is completely benign and has no risk of turning into malignant cancer.',
        'next_steps': 'No action needed. It may stay there permanently, but it is harmless. Have a doctor check it only if it becomes painful.',
        'alert_type': 'success'
    }
}

@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 7)
    model.load_state_dict(torch.load('skin_tag_resnet50.pth', map_location=device))
    model.eval()
    return model

model = load_model()

st.markdown("<h1 class='main-title'>+ Skin Tag AI Screener</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Upload a photo of your skin lesion for an AI-assisted analysis, complete with next steps and peace of mind.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Take a photo or upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2, col3 = st.columns([1, 2, 1])
    image = Image.open(uploaded_file).convert('RGB')
    with col2:
        st.image(image, caption="Your Uploaded Image", use_column_width=True, clamp=True)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    with st.spinner("Analyzing image patterns..."):
        with torch.no_grad():
            outputs = model(transform(image).unsqueeze(0))
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            
    results = {classes[i]: float(probs[i]) for i in range(7)}
    top_prediction = max(results, key=results.get)
    top_prob = results[top_prediction]
    
    st.markdown("---")
    
    st.subheader("📋 Analysis Results")
    
    st.markdown(f"### **Most Likely Match:** {top_prediction} ({top_prob*100:.1f}%)")
    
    info = DISEASE_INFO[top_prediction]
    
    if info['alert_type'] == 'success':
        st.success(f"**Risk Level:** {info['risk_level']}")
    elif info['alert_type'] == 'warning':
        st.warning(f"**Risk Level:** {info['risk_level']}")
    elif info['alert_type'] == 'info':
        st.info(f"**Risk Level:** {info['risk_level']}")
    else:
        st.error(f"**Risk Level:** {info['risk_level']}")
        
    st.markdown(f"""
        <div class="medical-card">
            <h4>What is it?</h4>
            <p>{info['description']}</p>
            <h4>Cancer Potential</h4>
            <p>{info['malignancy']}</p>
            <h4>Recommended Next Steps</h4>
            <p><strong>{info['next_steps']}</strong></p>
        </div>
    """, unsafe_allow_html=True)
    
    with st.expander("View other possibilities & model confidence"):
        for cls, prob in sorted(results.items(), key=lambda x: x[1], reverse=True):
            st.write(f"**{cls}**: {prob*100:.1f}%")
            st.progress(float(prob))

st.markdown("---")
st.caption("🩺 **Medical Disclaimer:** We know that checking skin conditions can be stressful. Please remember that this AI is a supportive tool, not a doctor. It analyzes visual patterns but cannot replace professional medical advice, diagnosis, or a physical biopsy. If you are ever worried about a spot on your skin, the best next step is always to schedule a quick checkup with a dermatologist.")