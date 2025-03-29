import streamlit as st
from transformers import pipeline
def main():
  st.title("Hugging Face Model Demo")
  input_text=st.text_input("Enter your text","")
  model=pipeline("sentiment-analysis",model="distilbert-base-uncased-finetuned-sst-2-english")
  if st.button("Analyze"):
    result=model(input_text)
    st.write("Prediction:",result[0]['label'],"|Score:",result[0]['score'])
if__name__=="__main__":
  main()
