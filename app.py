import streamlit as st
import numpy as np
import joblib
from transformers import BertTokenizer, BertModel
import torch
import time
from PIL import Image
import os

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = joblib.load('/Users/nahin/Documents/deploy/best_log_reg.pkl')

# Load BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.to(device)
bert_model.eval()

# Embedding function
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.detach().cpu().numpy().flatten()

# Predict function
def predict_text(text):
    embedding = get_bert_embedding(text).reshape(1, -1)
    return model.predict(embedding)[0]

# Page config
st.set_page_config(page_title="AI Text Classifier", layout="wide", page_icon="🤖")

# Custom CSS
st.markdown("""
    <style>
    body {
        background-color: #1e1b2e;
    }
    h1 {
        text-align: center;
        font-family: 'Helvetica', sans-serif;
        background: linear-gradient(to right, #e0aaff, #c77dff, #9d4edd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
    }
    .stTextArea textarea {
        border: 3px solid #c77dff;
        border-radius: 10px;
    }
    .stButton button {
        background-color: #9d4edd;
        color: white;
        border-radius: 10px;
        padding: 0.5em 2em;
        font-size: 1em;
        font-weight: bold;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>AI vs Human Text Detector</h1>", unsafe_allow_html=True)
st.markdown("🎯 Enter a paragraph and find out who wrote it!")

# Layout: 2 columns
col1, col2 = st.columns([1, 1])

with col1:
    user_input = st.text_area("📝 Your Text:", height=300)
    if st.button("🔍 Predict"):
        if user_input.strip():
            with st.spinner("Analyzing... 🤔"):
                time.sleep(2)
                prediction = predict_text(user_input)
                st.session_state.prediction = prediction
                st.session_state.text_entered = True
        else:
            st.warning("⚠️ Please enter some text.")

# Show prediction and image
with col2:
    if 'text_entered' in st.session_state and st.session_state.text_entered:
        pred = st.session_state.prediction
        if pred == 0:
            st.success("🧑‍🎓 **Prediction: Human-Written**")
            st.image("/Users/nahin/Documents/deploy/happy.jpg", caption="Looks human to me! 😄", width=300)
        else:
            st.error("🤖 **Prediction: ChatGPT-Generated**")
            st.image("/Users/nahin/Documents/deploy/robot.jpg", caption="That's some AI magic! 🤖✨", width=300)










