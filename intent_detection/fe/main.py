import streamlit as st

from intent_detection.fe.get_intent import main

st.set_page_config(layout="wide")

# selected_tool = st.sidebar.selectbox(label="Select a model", options=SUPPORTED_MODELS, index=0)

main()
