import streamlit as st

from intent_detection.intent.intent import clf


def main():

    st.title("Intent Detection")

    with st.expander("Info", expanded=False):
        st.write(
            """
    - Intent detection model UI
    - Inputs a query, outputs an label - intent
            """
        )

    help_str = "Some very helpful text"
    st.write("Embedding model: `{}`".format(clf.model_name))

    query = st.text_input(
        label="Query", value="Translate shark from english to spanish ", help=help_str
    )

    if st.button("Get intent"):
        intent = clf.get_intent(query)
        st.write(intent)

    # st.write(f"Predicted intent: {intent['intent']}, confidence: {intent['score']}")
