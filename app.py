import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input


# Назви класів Fashion-MNIST
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


# Налаштування сторінки
st.set_page_config(
    page_title="Fashion MNIST Classifier",
    page_icon="👕",
    layout="wide"
)

st.title("👕 Fashion MNIST Classifier")
st.write("Класифікація зображень за допомогою CNN та VGG16")


# Вибір моделі
model_choice = st.selectbox(
    "Оберіть модель:",
    ["CNN", "VGG16"]
)


# Завантаження відповідної моделі та історії
if model_choice == "CNN":

    model = tf.keras.models.load_model("cnn_model.keras")

    with open("cnn_history.json", "r") as f:
        history = json.load(f)

else:

    model = tf.keras.models.load_model("vgg16_model.keras")

    with open("vgg16_history.json", "r") as f:
        history = json.load(f)


# Завантаження зображення
uploaded_file = st.file_uploader(
    "Завантажте зображення одягу:",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Відкриваємо зображення
    image = Image.open(uploaded_file).convert("L")

    st.subheader("Завантажене зображення")

    st.image(
        image,
        caption="Ваше зображення",
        width=250
    )


    # Перетворюємо зображення у numpy
    img = np.array(image)


    # Підготовка для CNN
    if model_choice == "CNN":

        img_pil = Image.fromarray(img)
        img_pil = img_pil.resize((28, 28))

        img = np.array(img_pil)

        img = img.astype("float32") / 255.0

        img = img.reshape(1, 28, 28, 1)


    # Підготовка для VGG16
    else:

        img_pil = Image.fromarray(img)
        img_pil = img_pil.resize((224, 224))

        img = np.array(img_pil)

        # Перетворюємо grayscale у RGB
        img = np.stack([img] * 3, axis=-1)

        img = img.astype("float32")

        # Preprocessing VGG16
        img = preprocess_input(img)

        img = np.expand_dims(img, axis=0)


    # Передбачення
    prediction = model.predict(img, verbose=0)

    predicted_class = np.argmax(prediction[0])

    predicted_probability = prediction[0][predicted_class]


    # Результат
    st.subheader("Результат класифікації")

    st.success(
        f"Передбачений клас: {class_names[predicted_class]}"
    )

    st.write(
        f"Ймовірність: {predicted_probability * 100:.2f}%"
    )


    # Ймовірності всіх класів
    st.subheader("Ймовірності для всіх класів")

    probabilities = pd.DataFrame({
        "Клас": class_names,
        "Ймовірність (%)": prediction[0] * 100
    })

    probabilities = probabilities.sort_values(
        "Ймовірність (%)",
        ascending=False
    )

    st.dataframe(
        probabilities,
        use_container_width=True
    )


    # Графік ймовірностей
    st.subheader("Розподіл ймовірностей")

    chart_data = probabilities.set_index("Клас")

    st.bar_chart(chart_data)


# Графіки навчання
st.header("Графіки навчання моделі")


# Accuracy
st.subheader("Accuracy")

fig1, ax1 = plt.subplots()

ax1.plot(
    history["accuracy"],
    label="Train Accuracy"
)

ax1.plot(
    history["val_accuracy"],
    label="Validation Accuracy"
)

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.set_title(f"{model_choice} - Accuracy")
ax1.legend()

st.pyplot(fig1)


# Loss
st.subheader("Loss")

fig2, ax2 = plt.subplots()

ax2.plot(
    history["loss"],
    label="Train Loss"
)

ax2.plot(
    history["val_loss"],
    label="Validation Loss"
)

ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.set_title(f"{model_choice} - Loss")
ax2.legend()

st.pyplot(fig2)