import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Configure the Streamlit page
st.set_page_config(page_title="MNIST Digit Recognizer", page_icon="🔢")

st.title("MNIST Digit Recognizer")
st.write("Upload an image of a handwritten digit (0-9) to see the prediction!")

# Initialize session state for the file uploader key to allow resetting
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

# Cache the model so it doesn't reload on every interaction
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mnist_cnn.h5')

try:
    model = load_model()
except Exception as e:
    st.error("Model not found. Please ensure 'mnist_cnn.h5' is in the repository.")
    st.stop()

def process_image(image):
    # Convert to grayscale
    image = image.convert('L')
    # Invert colors (MNIST digits are white on a black background)
    image = ImageOps.invert(image)
    # Resize to 28x28 pixels
    image = image.resize((28, 28))
    # Convert to numpy array and normalize
    img_array = np.array(image) / 255.0
    # Add batch and channel dimensions for the model expected input (1, 28, 28, 1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array

# --- UI Layout ---
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["jpg", "jpeg", "png"], 
    key=str(st.session_state["uploader_key"])
)

col1, col2 = st.columns([1, 4])

with col1:
    predict_btn = st.button("Predict", type="primary")
with col2:
    reset_btn = st.button("Reset")

# --- App Logic ---
if reset_btn:
    # Incrementing the key forces the file uploader widget to remount and clear
    st.session_state["uploader_key"] += 1
    st.rerun()

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=200)

    if predict_btn:
        with st.spinner('Analyzing digit...'):
            processed_img = process_image(image)
            prediction = model.predict(processed_img)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction) * 100

        st.success(f"Predicted Digit: **{predicted_digit}**")
        st.info(f"Confidence: **{confidence:.2f}%**")
elif predict_btn:
    st.warning("Please upload an image first to make a prediction.")