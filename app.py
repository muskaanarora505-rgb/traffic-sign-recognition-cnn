import streamlit as st


st.title("Traffic Sign Recognition")

st.write(
    "Upload a traffic sign image to classify it."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded Image"
    )

    st.info(
        "Model prediction will be added after training."
    )
