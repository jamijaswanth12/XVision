This project presents an intelligent web application built with Streamlit and TensorFlow that predicts the presence of lung cancer from CT scan images. It leverages a fine-tuned Xception CNN model and includes Grad-CAM visualizations to highlight the regions of interest contributing to the prediction.

🚀 Features
Upload CT scan images and receive real-time predictions.

Visualize attention heatmaps (Grad-CAM) for model interpretability.

Fully interactive, responsive UI using Streamlit with custom CSS animations.

Robust error handling for smoother user experience.

Modular codebase for easy customization or model replacement.

📂 Tech Stack
Python 🐍

TensorFlow / Keras 🤖

Streamlit 📊

OpenCV / PIL for image handling 🖼️

Scikit-learn for metrics 📈

🧠 Model Details
Base Model: Pretrained Xception

Architecture: Custom dense layers on top of Xception

Output: Binary classification (Cancerous / Non-Cancerous)

💡 Use Cases
Educational use for understanding deep learning in medical imaging.

Prototype for diagnostic support in healthcare applications.

Research projects on model interpretability with Grad-CAM
