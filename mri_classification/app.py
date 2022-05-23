import streamlit as st
from PIL import Image

st.set_option('deprecation.showfileUploaderEncoding', False) # deprecation 표시 안함 
st.title("머신러닝 이용 뇌종양 MRI 사진 판독 서비스")
st.markdown("""
뇌종양 MRI 사진을 분류합니다. 
이 서비스는 의사들의 뇌 MRI 사진 판독의 도우미일 뿐입니다. 
정확한 최종 진단 결과는 반드시 전문 담당 의사의 확인과 승인을 거치십시요.""")

image2 = Image.open('brain.jpg')
st.image(image2, caption='shop2world.com 뇌종양 MRI 판독 서비스',use_column_width=True)


st.text("***이미지 분류를 위해 뇌 MRI 이미지를 업로드 해 주세요***")
#Streamlit 파일 처리 및 결과
from PIL import Image, ImageOps

from img_classification import mri_machine_classification
uploaded_file = st.file_uploader("뇌 MRI 사진을 업로드 해 주세요.", type=['jpeg', 'png', 'jpg', 'webp'])
if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI.', use_column_width=True)
        st.write("")
        st.write("처리중입니다...")
        label = mri_machine_classification(image, 'keras_model.h5')
        if label == 0:
            st.write("***결과 : MRI 스캔에는 뇌종양이 있습니다***")
        else:
            st.write("***결과 : MRI 스캔은 건강합니다***")
