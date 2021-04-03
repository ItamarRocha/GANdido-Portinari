from io import BytesIO
import base64
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
import tensorflow_addons as tfa
st.set_option('deprecation.showfileUploaderEncoding', False)

IMAGE_SIZE = [256,256]

def decode_image(image):
    image = (tf.cast(image, tf.float32)/127.5) - 1
    
    image = tf.image.resize(image, IMAGE_SIZE)

    image = tf.reshape(image, [1, *IMAGE_SIZE,3])

    return image

def get_image_download_link(img, capt):
    """Generates a link allowing the PIL image to be downloaded
    in:  PIL image
    out: href string
    """
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'       <a href="data:file/jpg;base64,{img_str}">{capt}</a>  '
    return href

st.title("GANdido Portinari")

st.sidebar.title("Options")
select = st.sidebar.selectbox('Choose the option you want to test!', ["Introduction","Generate Candido's style", "Generate real photos"])

if select == "Introduction":

    st.header("Introduction")

    st.markdown("""
    Hey there! Welcome to our project GANdido Portinari (Pun intended :P)!

    In this project we trained a CycleGan to reproduce Candido's Portinari style into photos of places and people.

    """)

    st.subheader("Ok, but who was Candido Portinari?")
    # Texto com Metodologia utilizada no trabalho
    st.markdown("""
    Candido Portinari was a Brazilian painter. He is considered one of the most important Brazilian painters as well as a prominent and influential practitioner of the neo-realism style in painting.
    
    """)
    
    candido = Image.open("imgs/candido.jpeg")
    st.image(candido, use_column_width=True)
    
    st.markdown("""
    
    One of his most famous paintings is **Retirantes (1944)**

    """)

    retirantes = Image.open("imgs/retirantes.jpg")
    st.image(retirantes, use_column_width=True)
    

    st.subheader("Data")
    st.markdown("""
    All the data from Candido Portinari work used in this project was collected from [Projeto Portinari](http://www.portinari.org.br/) using a simple download script after discovering the patterns of storage used by the website.
    """)

    st.subheader("Results")
    st.markdown("""
    You can see some of our results in our [Github repository](https://github.com/ItamarRocha/GANdido-Portinari#results)! 
    """)
    st.subheader("Autores")
    # Autores do Trabalho
    author_1, author_2, author_3, author_4 = st.beta_columns(4)

    jp = Image.open('imgs/jp.png')
    jw = Image.open('imgs/wallace.png')
    ita = Image.open('imgs/itamar.png')
    felipe = Image.open('imgs/felipe.png')


    with author_1:
        st.markdown('**[Itamar Filho](https://linkedin.com/in/itamarrocha)**')
        st.image(ita, use_column_width=True)
        st.markdown('Github: **[ItamarRocha](https://github.com/ItamarRocha)**')
    
    with author_2:
        st.markdown('**[João Pedro Teixeira](https://www.linkedin.com/in/jpvt/)**')
        st.image(jp, use_column_width=True)
        st.markdown('Github: **[jpvt](https://github.com/jpvt)**')

    with author_3:
        st.markdown('**[João Wallace Lucena](https://www.linkedin.com/in/jo%C3%A3o-wallace-b821bb1b0/)**')
        st.image(jw, use_column_width=True)
        st.markdown('Github: **[joallace](https://github.com/joallace)**')
    
    with author_4:
        st.markdown('**[Felipe Honorato](https://www.linkedin.com/in/felipehonoratodesousa/)**')
        st.image(felipe, use_column_width=True)
        st.markdown('Github: **[felipe](https://github.com/Felipehonorato1)**')

elif select == "Generate Candido's style":
    st.header("Generate Candido's style")
    
    uploaded_file = st.file_uploader("Choose one image to generate GANdido's style to it", type= ['png', 'jpg'], )
    
    gandido = tf.keras.models.load_model("weights/gandido_generator_1.h5", compile=False) # , custom_objects={'InstanceNormalization':tfa.layers.InstanceNormalization}

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        tensor = tf.convert_to_tensor(np.array(image))
        photo = decode_image(tensor)

        prediction = gandido(photo, training=False)[0].numpy()

        st.image(prediction * 0.5 + 0.5)
 
elif select == "Generate real photos":
    st.header("Generate real photos")

    uploaded_file = st.file_uploader("Choose one Candido painting to give it a real world style to it", type= ['png', 'jpg'] )
    
    real = tf.keras.models.load_model("weights/gandido_generator_2.h5", compile=False)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        tensor = tf.convert_to_tensor(np.array(image))
        photo = decode_image(tensor)

        prediction = real(photo, training=False)[0].numpy()

        st.image(prediction * 0.5 + 0.5)
 