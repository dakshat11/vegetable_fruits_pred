import streamlit as st
import tensorflow as tf
import numpy as np

# TensorFlow model prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.h5')
    test_image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(test_image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Us", "Prediction"])

# Home Page
if app_mode == "Home":
    st.header("FRUITS AND VEGETABLES PREDICTION SYSTEM", anchor="home")
    st.image("home_img.jpg", use_container_width=True)

# About Us Page
elif app_mode == "About Us":
    st.header("About Us", anchor="about")
    st.write("This is a project to predict the class of fruits and vegetables using deep learning.")
    st.write("We use machine learning models trained on image datasets.")
    st.write("Our goal is to help farmers and consumers make informed decisions.")
    
    st.subheader("Fruits")
    st.text("Banana, Apple, Pear, Grapes, Orange, Kiwi, Watermelon, Pomegranate, Pineapple, Mango")

    st.subheader("Vegetables")
    st.text("Cucumber, Carrot, Capsicum, Onion, Potato, Lemon, Tomato, Radish, Beetroot, Cabbage, Lettuce, Spinach, Soybean, Cauliflower, Bell Pepper, Chilli Pepper, Turnip, Corn, Sweetcorn, Sweet Potato, Paprika, Jalapeño, Ginger, Garlic, Peas, Eggplant")

    st.subheader("Dataset Info")
    st.text("Train: 100 images/category\nTest: 10 images/category\nValidation: 10 images/category")

    st.header("By- Dakshat Pawale")
# Prediction Page
elif app_mode == "Prediction":
    st.header("Model Prediction", anchor="prediction")
    test_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    if test_image:
        if st.button("Show Image"):
            st.image(test_image, use_container_width=True)
        
        if st.button("Predict"):
            st.balloons()
            st.write("Running prediction...")
            result_index = model_prediction(test_image)
            
            with open('labels.txt', 'r') as f:
                labels = [line.strip() for line in f.readlines()]
            
            st.success(f"Model prediction: **{labels[result_index]}**")
