from src.transformers import TopPublishersEncoder
import streamlit as st
import pandas as pd
import joblib
import time
import os

@st.cache_data
def fetch_books_data():
    time.sleep(3)
    return pd.read_csv("data/books_data_2.csv")

@st.cache_resource
def fetch_model():
    model_path = "models/book_saleability_rf_v1.pkl"
    return(joblib.load(model_path))

df = fetch_books_data()
model = fetch_model()
top_pub = model.named_steps["preparation"].named_transformers_["pub"].named_steps["mask"].top_pub
categories = categories = df["category"].dropna().unique()
    
left, middle, right = st.columns([1,2,1])

with middle:
    form_values = {
        "year" : None,
        "author": None,
        "page": None,
        "publisher": None,
        "category": None,
        "maturity": None
    }
    with st.form(key="predictor"):
        form_values["year"] = st.number_input("Enter the year the book was published: ",min_value=0)
        form_values["author"] = st.number_input("Enter the number of authors the book has", min_value=1)
        form_values["page"] = st.number_input("Enter the number of pages in the book", min_value=1)
        form_values["publisher"] = st.selectbox(label="Select publisher: ", options=top_pub)
        form_values["category"] = st.selectbox(label="Select category: ", options=categories)
        form_values["maturity"] = st.radio(options=["Mature", "Not mature"])

        submit_button = st.form_submit_button(label="Predict")

        if submit_button:
            if not all(form_values.values()):
                st.warning("Please full in all fields")
            else:
                



