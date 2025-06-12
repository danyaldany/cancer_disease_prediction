# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import pickle
#
# # Load the trained model
# MODEL_PATH = './ss_model/my_model.h5'  # Ensure this is correct
#
# try:
#     model = tf.keras.models.load_model(MODEL_PATH, compile=False)
#
#     # Recompile to avoid reduction='auto' error
#     model.compile(
#         optimizer='adam',
#         loss=tf.keras.losses.BinaryCrossentropy(),
#         metrics=['accuracy']
#     )
# except Exception as e:
#     st.error(f"❌ Error loading the model: {e}")
#     st.stop()
#
# # Load class names
# try:
#     with open('class_names.pkl', 'rb') as f:
#         class_names = pickle.load(f)
# except FileNotFoundError:
#     st.warning("⚠️ 'class_names.pkl' not found. Using default class names.")
#     class_names = ['no', 'yes']  # 'no' = 0, 'yes' = 1
#
# # UI
# st.title("🩺 Cancer Disease Prediction")
# st.write("Upload a medical image to check for signs of cancer.")
#
# # File uploader
# uploaded_file = st.file_uploader("📷 Upload an image...", type=["jpg", "jpeg", "png"])
#
# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert('RGB')
#     st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)
#     st.write("🔍 Classifying...")
#
#     # Convert to array (no resizing/scaling needed here if model handles it)
#     img_array = np.array(image)
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#
#     # Predict
#     predictions = model.predict(img_array)
#     probability = predictions[0][0]  # sigmoid gives one value between 0-1
#
#     # Classification
#     if probability >= 0.5:
#         predicted_class = class_names[1]  # 'yes'
#         confidence = round(probability * 100, 2)
#     else:
#         predicted_class = class_names[0]  # 'no'
#         confidence = round((1 - probability) * 100, 2)
#
#     # Show prediction result
#     st.subheader(f"📌 Prediction: **{predicted_class.upper()}**")
#     st.write(f"🧠 Confidence: `{confidence}%`")
#     st.write(f"🔬 Raw model output: `{round(probability, 4)}`")
#
#     if predicted_class.lower() == 'yes':
#         st.error("⚠️ The model predicts the **presence of cancer**.")
#     else:
#         st.success("✅ The model predicts **no cancer detected**.")



import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle

# DNA-like animated background using HTML canvas
st.markdown("""
<style>
html, body, [class*="css"] {
    height: 100%;
    margin: 0;
    padding: 0;
    background: #000;
    overflow: hidden;
    font-family: 'Segoe UI', sans-serif;
    color: white;
}

canvas#dnaCanvas {
    position: fixed;
    top: 0;
    left: 0;
    z-index: -1;
}
</style>

<canvas id="dnaCanvas"></canvas>

<script>
const canvas = document.getElementById("dnaCanvas");
const ctx = canvas.getContext("2d");

function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}
resizeCanvas();
window.addEventListener("resize", resizeCanvas);

let angle = 0;
function drawDNA() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const length = 100;
    const radius = 80;
    const spacing = 12;
    const turns = 20;

    for (let i = -turns; i < turns; i++) {
        const y = i * spacing + (angle * 4);
        const phase = i * 0.3 + angle;
        const x1 = centerX + Math.sin(phase) * radius;
        const x2 = centerX - Math.sin(phase) * radius;

        ctx.beginPath();
        ctx.strokeStyle = `rgba(0, 255, 255, ${1 - Math.abs(i) / turns})`;
        ctx.moveTo(x1, centerY + y);
        ctx.lineTo(x2, centerY + y);
        ctx.stroke();

        ctx.beginPath();
        ctx.fillStyle = "cyan";
        ctx.arc(x1, centerY + y, 3, 0, Math.PI * 2);
        ctx.fill();

        ctx.beginPath();
        ctx.fillStyle = "magenta";
        ctx.arc(x2, centerY + y, 3, 0, Math.PI * 2);
        ctx.fill();
    }

    angle += 0.01;
    requestAnimationFrame(drawDNA);
}
drawDNA();
</script>
""", unsafe_allow_html=True)

# Load model
MODEL_PATH = './ss_model/my_model.h5'
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
except Exception as e:
    st.error(f"❌ Error loading the model: {e}")
    st.stop()

# Load class names
try:
    with open('class_names.pkl', 'rb') as f:
        class_names = pickle.load(f)
except FileNotFoundError:
    st.warning("⚠️ 'class_names.pkl' not found. Using default class names.")
    class_names = ['no', 'yes']

# UI
st.title("🧬 Cancer Detection")
st.markdown("Upload a medical image to check for **cancer presence**.")

uploaded_file = st.file_uploader("📷 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)
    st.write("🔍 Classifying...")

    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    probability = predictions[0][0]

    if probability >= 0.5:
        predicted_class = class_names[1]
        confidence = round(probability * 100, 2)
    else:
        predicted_class = class_names[0]
        confidence = round((1 - probability) * 100, 2)

    st.markdown(f"""
    <div style='background-color: rgba(0,0,0,0.7); padding: 20px; border-radius: 10px; text-align:center'>
        <h2>📌 Prediction: <span style='color:#00ffae'>{predicted_class.upper()}</span></h2>
        <h3>🧠 Confidence: <span style='color:#FFD700'>{confidence}%</span></h3>
        <p>🔬 Raw model output: <code>{round(probability, 4)}</code></p>
    </div>
    """, unsafe_allow_html=True)

    if predicted_class.lower() == 'yes':
        st.error("⚠️ The model predicts the **presence of cancer**.")
    else:
        st.success("✅ The model predicts **no cancer detected**.")
