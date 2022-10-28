import streamlit as st
import tensorflow as tf
from PIL import Image

CLASS_LABELS = [chr(i) for i in range(65, 92)]

st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_page_config(page_title="Braille-Lipi",layout="wide")

@st.cache(allow_output_mutation=True)
def load_model():
	model = tf.keras.models.load_model(r'models/BrailleNet.h5')
	return model

def load_and_prep_image(filename: Image,img_shape=28):
    img = tf.cast(filename,tf.float32)
    img = img.resize((img_shape, img_shape))
    return img

def predict(image: Image,model):
    image = load_and_prep_image(image)
    st.image(image)
    pred_prob = model.predict(tf.expand_dims(image, axis=0),verbose=0) # make prediction on image with shape [None, 28, 28, 3]
    pred_class = CLASS_LABELS[pred_prob.argmax()]
    st.markdown(pred_class)
    prob_pred_class = tf.reduce_max(pred_prob).numpy()*100
    prob_class_str = "{:.2f}".format(prob_pred_class)
    st.success(f"It is a {pred_class} with {prob_class_str}% confidence")


image = st.file_uploader(label="Upload an image",type=['png','jpg','jpeg'])
picture = st.camera_input("Take a picture")

if picture:
    st.image(picture)
    test_image = Image.open(picture)
    model = load_model()
    if st.button("Predict"):
        predict(test_image, model)
if image is not None:
    st.image(image=image)
    test_image = Image.open(image)
    model = load_model()
    if st.button("Predict"):
        predict(test_image,model)