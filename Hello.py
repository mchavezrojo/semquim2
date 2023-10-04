import streamlit as st
import tensorflow as tf
import numpy as np
from scipy.ndimage.interpolation import zoom
from streamlit_drawable_canvas import st_canvas
from utils import process_image
st.markdown("# ¿Cuánto apuestas a que te :blue[adivino] :green[el número que escribes]? :P")

# Load trained model
model = tf.keras.models.load_model('mi_modelo.h5')

st.write('Draw a digit:')
# Display canvas for drawing
canvas_result = st_canvas(stroke_width=10, height=28*5, width=28*5)
  
# Process drawn image and make prediction using model
if np.any(canvas_result.image_data):
    # Convert drawn image to grayscale and resize to 28x28
    processed_image = process_image(canvas_result.image_data)
    # Make prediction using model
    prediction = model.predict(processed_image).argmax()
    # Display prediction
    st.header('Predicción:')
    st.markdown('Creo que escribiste un  \n # :red[' + str(prediction) + ']')
else:
    # Display message if canvas is empty
    st.header('Prediction:')
    st.write('No number drawn, please draw a digit to get a prediction.')
