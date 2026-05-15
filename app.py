import streamlit as st

st.title('AI Text to image generator')

prompt = st.text_input("Enter Prompt")

if st.button("Generate"):
    st.write(F'Generating Image for prompt:{prompt}')
