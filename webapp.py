from os import listdir
import numpy as np
from numpy import array
from numpy import argmax
from pickle import load
from PIL import Image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
import streamlit as st

@st.cache
def load_file():
    model = load_model('model-concat_incpt3-kaggle-ep004-val_loss3.587.h5') # to be modified
    tokenizer = load(open('tokenizer.pkl', 'rb'))
    return model,tokenizer

def word_for_id(integer, tokenizer):
    '''Map an integer to a word from tokenizer'''
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    '''Generate description for an image'''
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        # yhat give an integer, which is word_index in tokenizer
        yhat = model.predict([photo,sequence], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text

def extract_single_features(filename):
    '''Extract features from single photo in the directory'''
    # load the model
    model = InceptionV3()
    # re-structure the model
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # load the photo
#     image = load_img(filename, target_size=(299, 299))
    # convert the image pixels to a numpy array
    image = img_to_array(filename)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # get features
    feature = model.predict(image, verbose=0)
    return feature

def gen_image_capion(image):
    '''Generate image caption from trained model'''
    photo = extract_single_features(image)
    # generate description
    description = generate_desc(icg, desc_tokenizer, photo, max_caption_length)
    return description

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

def main():
    # setup main canvas
    st.title("Image Caption Generator")
    st.write('## Feed me a photo!')
    
    # Create image upload widget
    img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if img_file_buffer is not None:
        img = Image.open(img_file_buffer)
        st.image(img)
        # resize image for model
        img = img.convert('RGB')
        img = img.resize(target_size, Image.ANTIALIAS)
        st.write('## Suggested caption:')
        st.subheader(gen_image_capion(img)[9:-7])
    else:
        st.write('## No image uploaded')
        

# Global variables
max_caption_length=34
target_size=299,299
icg, desc_tokenizer=load_file()

if __name__ == "__main__":
    main()

    
    