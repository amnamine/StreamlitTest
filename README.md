# 🔢 MNIST Digit Recognizer — Streamlit Web App

A clean and interactive **Deep Learning web application** that recognizes handwritten digits (0–9) using a Convolutional Neural Network trained on the famous **MNIST dataset**.

Upload an image of a handwritten number and the model will instantly predict the digit with confidence score — all inside a beautiful Streamlit interface.

---

# 🌟 Project Overview

This project demonstrates an **end-to-end Machine Learning deployment pipeline**:

1. Train a CNN model using TensorFlow/Keras on MNIST
2. Save the trained model as `mnist_cnn.h5`
3. Build an interactive web interface using Streamlit
4. Deploy and run locally or on cloud platforms

The goal is to provide a beginner-friendly yet professional example of how to turn a trained model into a real usable application.

---

# 🧠 About the Model

The model is a **Convolutional Neural Network (CNN)** trained on the MNIST dataset.

### Why MNIST?

MNIST is the “Hello World” of Computer Vision:

* 70,000 grayscale images
* 28×28 pixel handwritten digits
* 10 classes (0–9)

CNNs are perfect for image recognition because they automatically learn:

* Edges
* Shapes
* Patterns
* Spatial relationships

The trained model achieves **very high accuracy (~99%)** on test data.

---

# 🖥️ App Features

✨ Upload your own handwritten digit image
✨ Automatic image preprocessing
✨ Real-time prediction
✨ Confidence score display
✨ Reset button to clear uploads
✨ Fast cached model loading
✨ Clean UI powered by Streamlit

---

# 📂 Repository Structure

```
MNIST-Digit-Recognizer/
│
├── streamlit_app.py      # Main Streamlit application
├── mnist_cnn.h5          # Trained CNN model
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

# ⚙️ How the App Works

## 1️⃣ Image Upload

The user uploads a JPG/PNG image containing a handwritten digit.

## 2️⃣ Image Preprocessing Pipeline

The image is transformed to match MNIST format:

| Step                   | Description                      |
| ---------------------- | -------------------------------- |
| Convert to grayscale   | MNIST uses single channel images |
| Invert colors          | MNIST digits are white on black  |
| Resize to 28×28        | Required input size              |
| Normalize              | Scale pixels to range 0–1        |
| Add batch/channel dims | Shape → `(1, 28, 28, 1)`         |

This ensures the uploaded image matches what the CNN expects.

---

## 3️⃣ Model Prediction

The CNN outputs probabilities for digits 0–9.

We then:

* Select the digit with highest probability
* Display prediction + confidence

---

# 🚀 Installation & Setup

## Step 1 — Clone the Repository

```bash
git clone https://github.com/your-username/mnist-digit-recognizer.git
cd mnist-digit-recognizer
```

---

## Step 2 — Create Virtual Environment (Recommended)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Step 3 — Install Requirements

Create a `requirements.txt` file:

```txt
streamlit
tensorflow
numpy
pillow
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Step 4 — Add the Trained Model

Place the file:

```
mnist_cnn.h5
```

in the project root directory (same folder as `streamlit_app.py`).

---

## Step 5 — Run the App 🎉

```bash
streamlit run streamlit_app.py
```

Your browser will open automatically:

```
http://localhost:8501
```

---

# 🧪 How to Use the App

1. Click **“Choose an image”**
2. Upload a handwritten digit image (0–9)
3. Click **Predict**
4. See the predicted digit + confidence score

Use the **Reset** button to upload a new image.

---

# 📸 Tips for Best Predictions

For highest accuracy:

* Use black background + white digit (or the app will invert)
* Center the digit
* Avoid extra noise
* Keep image simple and clear

---

# 🛠️ Technical Highlights

### Streamlit Techniques Used

* `st.cache_resource()` → prevents model reloading
* Session state → reset file uploader
* Columns layout → clean UI
* Spinners & alerts → better UX

### Machine Learning Stack

* TensorFlow / Keras
* NumPy
* Pillow (image processing)

---

# 🌍 Possible Improvements

Future upgrades you can add:

* Draw digit canvas ✏️
* Probability bar chart 📊
* Dark mode UI 🌙
* Deploy on Streamlit Cloud ☁️
* Convert model to TensorFlow Lite 📱
* Add multi-digit recognition 🔢

---

# 🎓 Learning Outcomes

This project teaches:

* CNN fundamentals
* Image preprocessing
* Model saving/loading
* Web app deployment
* ML → Product pipeline

Perfect for:

* Beginners in AI
* Portfolio projects
* University assignments
* Hackathons

---

# 🤝 Contributing

Contributions are welcome!

If you’d like to improve the project:

1. Fork the repo
2. Create a feature branch
3. Submit a Pull Request

---

# ⭐ Support the Project

If you found this helpful:

⭐ Star the repo
🍴 Fork it
📢 Share it

---

# 📜 License

This project is open-source and available under the **MIT License**.

---

# ❤️ Acknowledgments

* MNIST Dataset creators
* TensorFlow & Streamlit teams
* Open-source community

---

**Enjoy predicting handwritten digits! 🔢✨**
