import streamlit as st
import cv2
import numpy as np
from keras.models import load_model

import tensorflow as tf
import numpy as np
np.random.seed(0)
import cv2
import os
import pandas as pd
import cv2
from natsort import natsorted, ns
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from keras import models
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, BatchNormalization, Input, Dropout, Flatten
from keras.models import Model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout, Input, Lambda, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing import image
from keras import backend as K
from keras.models import Sequential, Model
from keras.optimizers import SGD, RMSprop, Adadelta
from keras.layers import BatchNormalization
from keras.regularizers import l2
from keras.metrics import AUC,Precision,Recall,Accuracy
from keras import optimizers

from keras import backend as K

def get_base_net(input_shape):
  seq = Sequential()
  seq.add(Conv2D(128, (11, 11),strides=(1, 1),activation='relu', name='conv1_1', input_shape= input_shape ))
  seq.add(BatchNormalization(epsilon=1e-06, axis=1, momentum=0.9))
  seq.add(MaxPooling2D((2,2), strides=(2, 2)))
  seq.add(ZeroPadding2D((2, 2)))

  seq.add(Conv2D(96, (5, 5), activation='relu', name='conv2_1', strides=(1, 1)))
  seq.add(BatchNormalization(epsilon=1e-06,  axis=1, momentum=0.9))
  seq.add(MaxPooling2D((2,2), strides=(2, 2)))
  seq.add(ZeroPadding2D((1, 1)))


  seq.add(Conv2D(96, (3, 3), activation='relu', name='conv3_1', strides=(1, 1)))
  seq.add(ZeroPadding2D((1, 1)))

  seq.add(Conv2D(64, (3, 3), activation='relu', name='conv3_2', strides=(1, 1)))
  seq.add(MaxPooling2D((2,2), strides=(2, 2)))
  seq.add(Dropout(0.3))# added extra

  seq.add(Flatten(name='flatten'))
  seq.add(Dense(512, kernel_regularizer=l2(0.0005), activation='relu'))
  seq.add(Dropout(0.5))

  seq.add(Dense(64, kernel_regularizer=l2(0.0005), activation='relu')) # softmax changed to relu

  return seq


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


input_a = Input(shape=(155,220,3))
input_b = Input(shape=(155,220,3))

base_net = get_base_net((155,220,3))
processed_a = base_net(input_a)
processed_b = base_net(input_b)

distance = Lambda(euclidean_distance,output_shape=eucl_dist_output_shape)([processed_a, processed_b])
model = Model([input_a, input_b], distance)

# print(model.summary(expand_nested=True))


# Load the pre-trained model
# mod = load_model('/Users/adityadubey/Desktop/SigVerify/tryout1.keras', custom_objects={'contrastive_loss': contrastive_loss})
# weights = load_model('/Users/adityadubey/Desktop/SigVerify/tryout1.keras', custom_objects={'contrastive_loss': contrastive_loss}, safe_mode=False)

# mod=get_base_net((155,220,3))
model.load_weights('tryout1.keras')


def Predict(model, path1, path2):
  im_1 = path1
  im_2 = path2
  im_1 = cv2.resize(im_1,(220,155))
  im_2 = cv2.resize(im_2,(220,155))
  im_1 = cv2.bitwise_not(im_1)
  im_2 = cv2.bitwise_not(im_2)
  im_1 = im_1/255
  im_2 = im_2/255
  im_1 = np.expand_dims(im_1,axis=0)
  im_2 = np.expand_dims(im_2,axis=0)
  y_pred = model.predict([im_1,im_2])

  return y_pred[0][0]

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
            image_1 = cv2.imdecode(np.fromstring(uploaded_file_1.read(), np.uint8), cv2.IMREAD_COLOR)
            image_2 = cv2.imdecode(np.fromstring(uploaded_file_2.read(), np.uint8), cv2.IMREAD_COLOR)

            # Make prediction
            prediction = Predict(model,image_1,image_2)

            # Display prediction result
            # st.write("Prediction:", prediction)
            if prediction < 0.5:
                st.success("Prediction: {}".format(prediction))
            else:
                st.error("Prediction: {}".format(prediction))
            

if __name__ == "__main__":
    main()
