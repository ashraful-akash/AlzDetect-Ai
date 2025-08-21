import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import requests

# CONFIG
st.set_page_config(page_title="Alzheimer's MRI Classifier", layout="centered")

CLASS_NAMES = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
CONFIDENCE_THRESHOLD = 0.5
MRI_VARIANCE_THRESHOLD = 0.01

# Google Drive direct download URLs for models
CUSTOM_CNN_URL = "https://drive.usercontent.google.com/download?id=1JjP803cKMeyWH-jE25VuwLJOjBozfkgL&export=download&confirm=t&uuid=c6a29eeb-1b71-4a39-b85d-9013dd84da90"
RESNET50_URL = "https://drive.usercontent.google.com/download?id=1AAWzPNF64apz6FNkMAnYODxyHjXMtMkL&export=download&confirm=t&uuid=4e6ae688-eb4f-43a5-a983-e136427e6fae"
XCEPTION_URL = "https://drive.usercontent.google.com/download?id=1coe86G1bGyeQWwAxn-sorDE6jiizmPHF&export=download&confirm=t&uuid=86b85896-2405-4b6e-b9e9-6602e03d73e1"

# Local filenames
CUSTOM_CNN_FILE = "custom_cnn.keras"
RESNET50_FILE = "resnet50.keras"
XCEPTION_FILE = "xception.keras"


# DOWNLOAD FUNCTION
def download_file_from_gdrive(url, destination):
    if os.path.exists(destination):
        return  # Skip download silently if file exists

    st.info(f"Downloading model file '{destination}' from Google Drive...")
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        st.error(f"Failed to download {destination} from Google Drive. Status code: {response.status_code}")
        return

    total_size = int(response.headers.get('content-length', 0))
    bytes_downloaded = 0
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        progress_bar = st.progress(0)
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                bytes_downloaded += len(chunk)
                if total_size:
                    progress_bar.progress(min(bytes_downloaded / total_size, 1.0))
        progress_bar.empty()
    st.success(f"Downloaded '{destination}' successfully from Google Drive.")

# Download models before loading
download_file_from_gdrive(CUSTOM_CNN_URL, CUSTOM_CNN_FILE)
download_file_from_gdrive(RESNET50_URL, RESNET50_FILE)
download_file_from_gdrive(XCEPTION_URL, XCEPTION_FILE)


# LOAD MODELS
@st.cache_resource
def load_customcnn():
    return load_model(CUSTOM_CNN_FILE)

@st.cache_resource
def load_resnet50():
    return load_model(RESNET50_FILE)

@st.cache_resource
def load_xception():
    return load_model(XCEPTION_FILE)

# PREPROCESSING FUNCTIONS
def preprocess_customcnn(img: Image.Image):
    img = img.convert('RGB').resize((224, 224))
    arr = np.array(img)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def preprocess_resnet50(img: Image.Image):
    img = img.convert('RGB').resize((224, 224))
    arr = np.array(img)
    arr = tf.keras.applications.resnet.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def preprocess_xception(img: Image.Image):
    img = img.convert('RGB').resize((244, 244))
    arr = np.array(img)
    arr = tf.keras.applications.xception.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def majority_vote(preds_list):
    counts = np.bincount(preds_list)
    return np.argmax(counts)

def is_probable_mri(image: Image.Image, threshold=MRI_VARIANCE_THRESHOLD) -> bool:
    gray = image.convert("L")
    arr = np.array(gray) / 255.0
    variance = np.var(arr)
    return variance > threshold

def show_prediction(preds):
    pred_idx = np.argmax(preds)
    confidence = preds[pred_idx]
    if confidence < CONFIDENCE_THRESHOLD:
        st.warning("Low confidence prediction. Please upload a valid brain MRI image.")
    else:
        st.subheader(f"Prediction: {CLASS_NAMES[pred_idx]}")
        st.write("Confidence Scores:")
        for i, score in enumerate(preds):
            st.write(f"{CLASS_NAMES[i]}: {score:.4f}")


# STREAMLIT UI
st.title("🧠 Alzheimer's MRI Classification")

model_choice = st.selectbox(
    "Choose Model",
    ["Custom CNN (fastest prediction)", "ResNet50", "Xception", "Ensemble (most accurate prediction)"]
)

uploaded_file = st.file_uploader("Upload any brain MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)  # fixed here

    if not is_probable_mri(img):
        st.error("Uploaded image does not appear to be a valid brain MRI. Please try again with a proper MRI scan.")
    else:
        try:
            if model_choice == "Custom CNN (fastest prediction)":
                model = load_customcnn()
                pre = preprocess_customcnn(img)
                with st.spinner("Predicting..."):
                    preds = model.predict(pre)[0]
                show_prediction(preds)

            elif model_choice == "ResNet50":
                model = load_resnet50()
                pre = preprocess_resnet50(img)
                with st.spinner("Predicting..."):
                    preds = model.predict(pre)[0]
                show_prediction(preds)

            elif model_choice == "Xception":
                model = load_xception()
                pre = preprocess_xception(img)
                with st.spinner("Predicting..."):
                    preds = model.predict(pre)[0]
                show_prediction(preds)

            elif model_choice == "Ensemble (most accurate prediction)":
                model_cnn = load_customcnn()
                model_resnet = load_resnet50()
                model_xcep = load_xception()

                with st.spinner("Predicting with ensemble..."):
                    preds_cnn = model_cnn.predict(preprocess_customcnn(img))[0]
                    preds_resnet = model_resnet.predict(preprocess_resnet50(img))[0]
                    preds_xcep = model_xcep.predict(preprocess_xception(img))[0]

                preds_cnn_idx = np.argmax(preds_cnn)
                preds_resnet_idx = np.argmax(preds_resnet)
                preds_xcep_idx = np.argmax(preds_xcep)

                final_pred = majority_vote([preds_cnn_idx, preds_resnet_idx, preds_xcep_idx])

                confidences = []
                if preds_cnn_idx == final_pred:
                    confidences.append(preds_cnn[final_pred])
                if preds_resnet_idx == final_pred:
                    confidences.append(preds_resnet[final_pred])
                if preds_xcep_idx == final_pred:
                    confidences.append(preds_xcep[final_pred])
                max_confidence = max(confidences) if confidences else 0.0

                if max_confidence < CONFIDENCE_THRESHOLD:
                    st.warning("Low confidence prediction in ensemble. Please upload a valid brain MRI image.")
                else:
                    st.subheader(f"Ensemble Prediction: {CLASS_NAMES[final_pred]}")
                    st.write(f"Votes → CustomCNN: {CLASS_NAMES[preds_cnn_idx]}, ResNet50: {CLASS_NAMES[preds_resnet_idx]}, Xception: {CLASS_NAMES[preds_xcep_idx]}")
                    st.write(f"Confidence: {max_confidence:.4f}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
