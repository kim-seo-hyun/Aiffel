# 필요한 라이브러리 가져 오기
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os

st.set_option('deprecation.showfileUploaderEncoding', False)#FileUploaderEncodingWarning
# cascade classifier에 대해 사전 훈련 된 매개 변수로드
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
except Exception:
    st.write("Error loading cascade classifiers")

def detect(image):
    '''
    이 function에 전달 된 이미지에서 얼굴 / 눈 및 미소를 감지
    '''

    
    image = np.array(image.convert('RGB'))
    
    faces = face_cascade.detectMultiScale(image=image, scaleFactor=1.3, minNeighbors=5)
    # face_cascade 분류기는 이미지에서 얼굴이 위치 할 수있는 영역의 좌표를 반환합니다.
    # 이 좌표는 (x, y, w, h)
    # 우리는 전체 이미지에서 눈을 찾는 대신 이 영역에서 눈을 찾고 미소를 지을 것입니다.


    # 얼굴  주위에 직사각형 그리기
    for (x, y, w, h) in faces:
        
        # 다음은 cv2.rectangle ()의 매개 변수 안내 입니다.
        # cv2.rectangle(image_to_draw_on, start_point, end_point, color, line_width)
        cv2.rectangle(img=image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
        
        roi = image[y:y+h, x:x+w]
        
        # 감지된 얼굴에서 눈 감지
        eyes = eye_cascade.detectMultiScale(roi)
        
        # 감지된 얼굴에서 얼굴의 미소 감지
        smile = smile_cascade.detectMultiScale(roi, minNeighbors = 25)
        
        # 눈 주위에 직사각형 그리기
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)
            
        # 미소 주위에 직사각형 그리기
        for (sx,sy,sw,sh) in smile:
            cv2.rectangle(roi, (sx, sy), (sx+sw, sy+sh), (0,0,255), 2)

    # 경계 상자가 그려진 이미지를 반환(감지 된 개체의 경우) , 그리고 faces array
    return image, faces


def about():
	st.write(
		'''
		**Haar Cascade** 은 객체 탐지 알고리즘입니다.
		이미지 또는 비디오에서 물체를 감지하는 데 사용할 수 있습니다. 
		알고리즘에는 4 단계가 있습니다.:
			1. Haar Feature 선택 
			2. 통합 이미지 만들기
			3. Adaboost 트레이닝 
			4. Cascading Classifiers
참고 자료 :point_right: https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html
https://sites.google.com/site/5kk73gpu2012/assignment/viola-jones-face-detection#TOC-Image-Pyramid
		''')


def main():
    st.title("얼굴 인식 앱 :😉 ")
    st.write("**정면이 보이는 칼라 사진을 업로드 해 주세요**")


    activities = ["Home", "About"]
    choice = st.sidebar.selectbox("메뉴", activities)

    if choice == "Home":

    	st.write("인식된 얼굴의 눈과 미소 주위에 직사각형이 그려 집니다.")
        # 원하는 경우 아래에서 더 많은 파일 형식을 지정할 수 있습니다.
    	image_file = st.file_uploader("이미지 업로드", type=['jpeg', 'png', 'jpg', 'webp'])

    	if image_file is not None:

    		image = Image.open(image_file)

    		if st.button("진행"):
                
                #result_img는 직사각형이 그려진 이미지입니다 (얼굴이 감지 된 경우).
                # result_faces는 경계 상자의 좌표가있는 배열입니다.
    			result_img, result_faces = detect(image=image)
    			st.image(result_img, use_column_width = True)
    			st.success("Found {} faces\n".format(len(result_faces)))

    elif choice == "About":
    	about()




if __name__ == "__main__":
    main()



    
