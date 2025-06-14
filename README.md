**Brain MRI Tumor Detection (CNN Model)**

## Brain MRI Tumor Detection (CNN Model)

A deep learning project that uses Convolutional Neural Networks (CNN) to detect brain tumors in MRI images. The goal is to assist in early and accurate detection of tumors through automated image classification.


### Overview

Brain tumors are life-threatening conditions that require timely diagnosis. This project leverages CNN-based deep learning techniques to analyze MRI scans and classify them as **Yes** or **No**. The model
is trained on a labeled MRI dataset and achieves high accuracy in detecting the presence of tumors.

### Features

* Upload MRI images for instant tumor prediction
* Deep learning classification using a custom CNN model
* High accuracy on validation/test sets
* Clean and simple code structure
* Optionally deployable using Streamlit

---

### Tech Stack

| Area       | Tools/Frameworks             |
| ---------- | ---------------------------- |
| Language   | Python                       |
| Model      | TensorFlow / Keras (CNN)     |
| Libraries  | NumPy, Matplotlib, PIL       |
| Interface  | Streamlit                    |
| Deployment | Localhost                    |

---

### 📊 Dataset

* **Name:** Brain MRI Images for Brain Tumor Detection
* **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
* **Classes:** `Yes`, `No`
* Images are grayscale, preprocessed, and resized for optimal training performance.

---

### Model Details

* **Architecture:** Custom CNN with multiple Conv2D, MaxPooling, and Dense layers
* **Loss Function:** Binary Crossentropy
* **Optimizer:** Adam
* **Accuracy:** e.g., 80% 
* **Saved Model:** `model.h5`

---


### Sample Output *(optional)*

![Streamlit - Google Chrome 6_14_2025 11_43_10 AM](https://github.com/user-attachments/assets/c771b039-fc8a-4679-9e28-026e71bffc80)
![Streamlit - Google Chrome 6_14_2025 11_43_31 AM](https://github.com/user-attachments/assets/6961a771-f9f8-4452-94b2-dc86b04dff43)



### 📜 License

This project is open-source under the [MIT License](LICENSE).

