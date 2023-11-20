import streamlit as st
import pickle 
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
from transformers import pipeline
from transformers import BertTokenizer, BertForSequenceClassification


st.title("Truth Seeker App")
st.image("1.jpg")

@st.cache(hash_funcs={'self': lambda _: 0})
def get_model():
   tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
   model = BertForSequenceClassification.from_pretrained("sai/NewsTunedBert")
   return tokenizer, model
    
user_input = st.text_area('Enter Text to Analyze')

if user_input is not None:
    if st.button("Analyse"):
        classifier = pipeline("sentiment-analysis")
        prediction  = classifier(user_input)
        result = "Propagandistic" if (prediction[0]["label"]) == 'NEGATIVE' else "Non-Propagandistic"
        st.subheader("Result:")
        st.info("The article is "+ result)
