# #CALCULATOR 
# import streamlit as st

# def main():
#     st.title("Square Calculator")

#     # User input
#     number = st.number_input("Enter a number")

#     # Calculate square
#     square = number ** 2

#     # Display result
#     st.write(f"The square of {number} is {square}")

# if __name__ == "__main__":
#     main()



import streamlit as st
import cv2
import numpy as np
from keras.models import load_model

from keras import backend as K

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    margin = 1
    #print("y_pred",y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


# Load the pre-trained model
mod = load_model('/Users/adityadubey/Desktop/SigVerify/model_2.h5', custom_objects={'contrastive_loss': contrastive_loss})

# Define the function to preprocess the images
def preprocess_image(image):
    # Resize the image
    image = cv2.resize(image, (220, 155))
    # Invert the image (assuming it's a signature)
    image = cv2.bitwise_not(image)
    # Normalize the pixel values
    image = image / 255.0
    # Add a batch dimension
    image = np.expand_dims(image, axis=0)
    return image

# Define the Streamlit app
def main():
    st.title("Signature Verification App")
    st.write("Upload two signature images and get the prediction result.")

    # File uploaders for the two images
    uploaded_file_1 = st.file_uploader("Upload first image", type=["jpg", "jpeg", "png"])
    uploaded_file_2 = st.file_uploader("Upload second image", type=["jpg", "jpeg", "png"])

    # Button to trigger prediction
    if st.button("Predict"):
        if uploaded_file_1 is None or uploaded_file_2 is None:
            st.warning("Please upload both images.")
        else:
            # Read and preprocess the images
            image_1 = cv2.imdecode(np.fromstring(uploaded_file_1.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            image_2 = cv2.imdecode(np.fromstring(uploaded_file_2.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            preprocessed_image_1 = preprocess_image(image_1)
            preprocessed_image_2 = preprocess_image(image_2)

            # Make prediction
            prediction = mod.predict([preprocessed_image_1, preprocessed_image_2])

            # Display prediction result
            st.write("Prediction:", prediction)

if __name__ == "__main__":
    main()
