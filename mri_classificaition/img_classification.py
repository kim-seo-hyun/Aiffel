import keras
from PIL import Image, ImageOps
import numpy as np


def mri_machine_classification(img, weights_file):
    # 모델로드
    model = keras.models.load_model(weights_file)

    # keras 모델에 공급할 올바른 모양의 배열을 만듭니다.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    #이미지 크기 조정
    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #이미지를 numpy 배열로 바꿈.
    image_array = np.asarray(image)
    # 이미지 정규화
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # 이미지를 배열에로드
    data[0] = normalized_image_array

    # 예측 시작 
    prediction = model.predict(data)
    return np.argmax(prediction) # 가장 높은 확률의 반환 위치