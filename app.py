from operator import index
import numpy as np
import pandas as pd
import pickle
import streamlit as st
st.markdown("<h1 id='soft_name' style='text-align: center; color: #c06300; font-size: 40px ; text-decoration: underline;'>SpamXSpotter</h1><br>", unsafe_allow_html=True)
loaded_model = open("model.pkl", "rb")
classifier = pickle.load(loaded_model)

st.markdown("<h3>Enter Mail<h3>", unsafe_allow_html=True)

form = st.form("my_form", clear_on_submit=True)
mail_txt = form.text_area("Enter Mail", label_visibility='hidden')

# Now add a submit button to the form:
submit1=form.form_submit_button("Submit")

# Load the vectorizer with the same configuration used during training
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def prediction(mail_txt):
    mail_txt_vectorized = vectorizer.transform([mail_txt])
    # Reshape input data to 2D array
    mail_txt_2d = mail_txt_vectorized.reshape(1, -1)
    if classifier.predict(mail_txt_2d) == 1:
        st.markdown("<h5 style='color: green; text-align:center;'>Mail is not Spam</h5>", unsafe_allow_html=True)
    elif classifier.predict(mail_txt_2d) == 0:
        st.markdown("<h5 style='color: red; text-align:center;'>Mail is Spam</h5>", unsafe_allow_html=True)

if submit1:    
    prediction(mail_txt)
