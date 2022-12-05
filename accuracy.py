import streamlit as st
import cv2
import numpy as np
st.set_page_config(page_title="EduAR",layout="wide")

st.title("EduAR")

st.subheader("Accuracy Check")
c1,c2 = st.columns(2)

with c1:
    img1 = st.file_uploader("Upload Image1", type=["png","jpg","jpeg"])

with c2:
    img2 = st.file_uploader("Upload Image2", type=["png","jpg","jpeg"])

if img1 and img2:
    img1 = cv2.imread(img1.name)
    img2 = cv2.imread(img2.name)
    
    img1 = cv2.resize(img1, (300,300))
    img2 = cv2.resize(img2, (300,300))
    
    c1.image(img1,width=250)
    c2.image(img2,width=250)
    
    st.write('---')
    st.subheader('Result')

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    def mse(img1, img2):
        h, w = img1.shape
        diff = cv2.subtract(img1, img2)
        err = np.sum(diff**2)
        mse = err/(float(h*w))
        return mse, diff

    error, diff = mse(img1, img2)

    cl1, cl2 = st.columns(2)
    
    # st.subheader('Result')
    with cl2:
        st.write("Accuracy:",(100-error))

    with cl1:
        st.image(diff)


    cv2.waitKey(0)
    cv2.destroyAllWindows()