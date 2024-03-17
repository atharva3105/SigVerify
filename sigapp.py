import streamlit as st
from PIL import Image
from io import BytesIO

# Function to verify signatures (replace with your actual verification logic)
def verify_signature(signature1, signature2):
    # Placeholder logic, replace with actual signature verification code
    return signature1 == signature2

def main():
    st.title("Signature Verification App")

    st.write("Upload two signature images and click on 'Verify' to check if they are forged or not.")

    uploaded_files = st.file_uploader("Upload Signature Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if uploaded_files is not None and len(uploaded_files) == 2:
        signature1 = Image.open(uploaded_files[0])
        signature2 = Image.open(uploaded_files[1])

        st.image(signature1, caption='Signature 1', use_column_width=True)
        st.image(signature2, caption='Signature 2', use_column_width=True)

        if st.button("Verify"):
            # Convert images to bytes
            signature1_bytes = BytesIO()
            signature2_bytes = BytesIO()
            signature1.save(signature1_bytes, format='PNG')
            signature2.save(signature2_bytes, format='PNG')

            # Verify signatures
            result = verify_signature(signature1_bytes.getvalue(), signature2_bytes.getvalue())
            
            if result:
                st.success("Signatures are genuine.")
            else:
                st.error("Signatures are forged.")
    elif uploaded_files is not None and len(uploaded_files) < 2:
        st.warning("Please upload two signature images.")

if __name__ == "__main__":
    main()
