from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


model=genai.GenerativeModel(model_name="models/gemini-1.5-flash")

def get_gemini_response(question):
    try:
        response = model.generate_content(question)
        return response.text
    except Exception as e:
        return f"Error: {e}"

## initialize out streamlit app

st.set_page_config(page_title="Q&A Demo")
st.header("Abhirup LLM Application")
input=st.text_input("Input: ",key="input")
submit=st.button("Ask the Question")

if submit:
    response=get_gemini_response(input)
    st.subheader("The Response is")
    st.write(response)
st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        padding: 10px;
        font-size: 14px;
        color: grey;
    }
    </style>

    <div class="footer">
        Made by Abhirup
    </div>
""", unsafe_allow_html=True)
