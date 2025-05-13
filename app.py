import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import io
import base64  # Import base64 for image encoding

# Load full model (Sequential)
try:
    full_model = tf.keras.models.load_model("model_xception (2).h5")
except Exception as e:
    st.error(f"Failed to load the model: {e}")
    # Stop execution if the model fails to load
    st.stop()

# Extract Xception base model (Functional part)
xception_model = full_model.get_layer("xception")
xception_model.trainable = False

# Name of the last conv layer in Xception
last_conv_layer_name = "block14_sepconv2_act"

# Preprocess image
def preprocess_image(image):
    try:
        image = image.convert("RGB")
        image = image.resize((224, 224))
        img_array = np.array(image).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Error in preprocess_image: {e}")
        return None  # Important: Return None on error

# Grad-CAM Function (DO NOT MODIFY)
def make_gradcam_heatmap(img_array):
    try:
        grad_model = tf.keras.models.Model(
            [xception_model.input],
            [xception_model.get_layer(last_conv_layer_name).output]
        )

        with tf.GradientTape() as tape:
            conv_outputs = grad_model(img_array)
            tape.watch(conv_outputs)
            x = conv_outputs
            x = tf.keras.layers.Dropout(0.5)(x)
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dense(32)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            x = tf.keras.layers.Dense(32)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            x = tf.keras.layers.Dense(32)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)
            preds = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        grads = tape.gradient(preds, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-10)
        return heatmap
    except Exception as e:
        st.error(f"Error in make_gradcam_heatmap: {e}")
        return None  # Important: Return None on error

# Overlay heatmap on image (DO NOT MODIFY)
def overlay_heatmap(original_img, heatmap, alpha=0.4):
    try:
        heatmap = cv2.resize(heatmap, (original_img.width, original_img.height))
        img_array = np.array(original_img)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img_array, 1 - alpha, heatmap_color, alpha, 0)
        superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(superimposed_img)
    except Exception as e:
        st.error(f"Error in overlay_heatmap: {e}")
        return None  # Important: Return None on error

# Predict using full model (DO NOT MODIFY)
def predict_label(img_array):
    try:
        return full_model.predict(img_array)
    except Exception as e:
        st.error(f"Error in predict_label: {e}")
        return None  # Important: Return None on error

# Dummy performance metric calculator (DO NOT MODIFY)
def calculate_metrics(true_labels, predicted_labels):
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)
        return accuracy, precision, recall, f1
    except Exception as e:
        st.error(f"Error in calculate_metrics: {e}")
        return None  # Important: Return None on error

# --- Streamlit Page Settings ---
st.set_page_config(page_title="🩺 Lung Cancer Prediction and Classification", layout="wide")

# --- Custom CSS for Improved Styling ---
st.markdown(
    """
    <style>
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes slideIn {
        from { transform: translateX(-100%); }
        to { transform: translateX(0); }
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    .stApp {
        background-color: #f0f8ff;
    }
    .main {
        background-color: white;
        padding: 3rem;
        border-radius: 15px;
        box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
        margin-top: 2rem;
        margin-bottom: 2rem;
        display: flex;
        flex-direction: column;
        align-items: center;
        animation: fadeIn 1s ease-in-out;
    }
    .left-column {
        width: 100%;
        padding-right: 0;
        margin-bottom: 2rem;
        animation: slideIn 1s ease-in-out;
    }
    .right-column {
        width: 100%;
    }
    h1 {
        color: #2e8b57;
        text-align: center;
        margin-bottom: 2rem;
        animation: pulse 2s infinite;
    }
    h2 {
        color: #4682b4;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    p {
        font-size: 1.1rem;
        color: #555;
        line-height: 1.7;
    }
    .stFileUploader label {
        background-color: #708090 !important;
        color: white !important;
        border-radius: 5px;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stFileUploader label:hover {
        background-color: #556b2f !important;
    }
    .stImage > div > img {
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        width: 100%;
        height: auto;
        animation: fadeIn 1s ease-in-out;
    }
    .stDownloadButton button {
        background-color: #4682b4 !important;
        color: white !important;
        border-radius: 5px;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        transition: background-color 0.3s ease;
        width: auto;
    }
    .stDownloadButton button:hover {
        background-color: #5f9ea0 !important;
    }
    .streamlit-expander header {
        background-color: #e0f2f7;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        margin-bottom: 1rem;
    }
    .streamlit-expander header p {
        color: #2e8b57;
        font-size: 1.1rem;
        font-weight: 500;
    }
    .streamlit-expander .streamlit-expander-content {
        padding: 1rem;
        font-size: 1rem;
        color: #555;
        line-height: 1.7;
    }
    .reportview-container .main .st-block-container {
        max-width: 100%;
        padding-left: 5rem;
        padding-right: 5rem;
        display: block;
    }
    @media (max-width: 768px) {
        .reportview-container .main .st-block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            flex-direction: column;
        }
        .left-column, .right-column {
            width: 100%;
            padding-right: 0;
        }
    }
    #about-box {
        background-color: #e0f7fa;
        border: 1px solid #b0e0e6;
        border-radius: 5px;
        padding: 0.75rem;
        margin-bottom: 1rem;
        width: 80%;
        margin-left: auto;
        margin-right: auto;
        text-align: center;
        animation: fadeIn 1s ease-in-out;
    }
    #about-box h2 {
        color: #2e8b57;
        margin-top: 0;
        margin-bottom: 0.25rem;
        font-size: 1.5rem;
    }
    #about-box p{
        font-size: 1rem;
    }
    .menu-button {
        background-color: #4CAF50;
        color: white;
        padding: 1rem 2rem;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 1.2rem;
        margin: 0.5rem 1rem;
        cursor: pointer;
        border-radius: 5px;
        transition: background-color 0.3s ease;
        width: 200px;
        animation: fadeIn 1s ease-in-out;
    }
    .menu-button:hover {
        background-color: #367c39;
    }
    .hidden {
        display: none;
    }
    #upload-container, #about-container {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 2rem;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
        width: 80%;
        margin-left: auto;
        margin-right: auto;
        animation: fadeIn 1s ease-in-out;
    }
    #home-content {
        text-align: center;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 2rem;
        background-color: #ffffff;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
        width: 80%;
        margin-left: auto;
        margin-right: auto;
        animation: fadeIn 1s ease-in-out;
    }
    #home-content h2{
        color: #2e8b57;
        font-size: 2.5rem;
        margin-bottom: 1rem;
        animation: pulse 2s infinite;
    }
    #home-content p{
        font-size: 1.2rem;
        color: #555;
        line-height: 1.7;
        margin-bottom: 2rem;
        animation: slideIn 1s ease-in-out;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Title and Subheader with Emojis ---
st.title("🩺 Lung Cancer Prediction and Classification")
st.subheader("Upload a CT image of lung to see the prediction and attention heatmap 🔍")

# --- Main App Logic ---
menu_selection = st.sidebar.selectbox("Menu", ["Home", "Upload Image", "About"])


# --- Columns for layout ---
left_column, right_column = st.columns(2)
# --- Left Column for Image and Information ---

if menu_selection == "Home":
    st.markdown(
        """
        <div id="home-content">
            <h2>Welcome</h2>
            <p>
                This application is a tool for research and educational purposes, designed to assist in the analysis of lung CT scans.
                It utilizes a sophisticated deep learning model to predict the likelihood of cancerous indicators and provides visual
                feedback, highlighting areas of interest in the image.
            </p>
            
            Please use the menu on the left to upload a CT scan for analysis or learn more about the application.
        </div>
        """,
        unsafe_allow_html=True
    )
elif menu_selection == "Upload Image":
    uploaded_file = st.file_uploader("Upload a lung CT image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        img_array = preprocess_image(image)
        if img_array is not None:
            prediction = predict_label(img_array)
            if prediction is not None:
                label = "Non-Cancerous" if prediction[0][0] >= 0.5 else "Cancerous"
                confidence = prediction[0][0] if prediction[0][0] >= 0.5 else 1 - prediction[0][0]

                with left_column:
                    st.markdown(f"<h2>🔍 Prediction</h2>"
                                f"<p style='font-size: 1.3rem; font-weight: bold; color: #2e8b57;'>{label}</p>",
                                unsafe_allow_html=True)
                    st.progress(int(confidence * 100))
                    st.markdown(f"<p style='font-size: 1.1rem;'>*Confidence Level:* {confidence:.2%}</p>", unsafe_allow_html=True)

                # --- Right Column for Grad-CAM Heatmap and Instructions ---
                with right_column:
                    try:
                        heatmap = make_gradcam_heatmap(img_array)
                        if heatmap is not None:
                            heatmap_image = overlay_heatmap(image, heatmap, alpha=0.4)
                            if heatmap_image is not None:
                                st.markdown("<h2>🔥 Grad-CAM Heatmap (Areas of Interest)</h2>")
                                st.image(heatmap_image, caption="Grad-CAM Heatmap", use_column_width=True)

                                # Download button
                                buffered = io.BytesIO()
                                heatmap_image.save(buffered, format="PNG")
                                img_str = base64.b64encode(buffered.getvalue()).decode()
                                href = f'<a href="data:image/png;base64,{img_str}" download="heatmap_result.png" download><button style="background-color:#4682b4;color:white;border-radius:5px;padding:0.75rem 1.5rem;font-size:1.1rem;">📥 Download Heatmap</button></a>'
                                st.markdown(href, unsafe_allow_html=True)
                            else:
                                st.error("Failed to generate overlay heatmap.")
                        else:
                            st.error("Failed to generate Grad-CAM heatmap.")
                    except Exception as e:
                        st.error(f"Error during Grad-CAM processing: {e}")
            else:
                st.error("Failed to get prediction from the model.")
        else:
            st.error("Failed to preprocess the image.")

elif menu_selection == "About":
    st.markdown(
        """
        <div id="about-box">
            <h2>About</h2>
            <p>Developers: Jami Jaswanth, Palameti Reddy Lakshmi Manoj</p>
            <p>This application uses a Xception model to classify medical images and highlight areas of interest using Grad-CAM.</p>
            
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Disclaimer and Footer ---
st.markdown("---")
st.markdown(
    """
    <p style="text-align: center; font-size: 0.9rem; color: #888;">
    <strong>Disclaimer:</strong> This application is for research and educational purposes only.
    It is not intended for diagnostic use. Always consult with a qualified medical professional
    for any health concerns.
    </p>
    """,
    unsafe_allow_html=True
)
