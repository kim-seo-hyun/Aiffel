import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.set_option('deprecation.showfileUploaderEncoding', False)#8.15일 이후로 엔코딩 지원 문제로 워닝 메세지 표현 하지 않기 위함 

st.write("""
# 팔머(Palmer) 펭귄 3종 예측 앱
이앱은  **팔머 펭귄(Palmer Penguin)** 의 종을 예측합니다.!
1 젠투 펭귄(Gentoo Penguin): 머리에 모자처럼 둘러져 있는 하얀 털 때문에 알아보기가 쉽다. 암컷이 회색이 뒤에, 흰색이 앞에 있다. 
2 아델리 펭귄(Adelie Penguin): 각진 머리와 작은 부리 때문에 알아보기 쉽고, 다른 펭귄들과 마찬가지로 암수가 비슷하게 생겼지만 암컷이 조금 더 작다.
3 턱끈 펭귄(Chinstrap Penguin): 목에서 머리 쪽으로 이어지는 턱끈 같은 검은 털이 눈에 띈다. 
데이터 출처  [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) 
사용할 데이터 
http://ai.shop2world.net/data/penguins.csv
전체 파일 
http://ai.shop2world.net/data/penguins.zip
""")


st.sidebar.header('사용자 입력 값(Feature)')

st.sidebar.markdown("""
[Example CSV input file](https://ai.shop2world.net/data/penguins_example.csv)
""")

# 사용자 입력 feature를 데이터 프레임에 수집



uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
        sex = st.sidebar.selectbox('Sex',('male','female'))
        bill_length_mm = st.sidebar.slider('부리 길이 Bill length (mm)', 32.1,59.6,43.9)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1,21.5,17.2)
        flipper_length_mm = st.sidebar.slider('날개 길이 Flipper length (mm)', 172.0,231.0,201.0)
        body_mass_g = st.sidebar.slider('몸무게 Body mass (g)', 2700.0,6300.0,4207.0)
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# 사용자 입력 피쳐값을 전체 펭귄 데이터 세트와 결합
# 인코딩 단계에 유용합니다.
penguins_raw = pd.read_csv('http://ai.shop2world.net/data/penguins.csv')
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df,penguins],axis=0)

# 서수(ordinal features)의 인코딩
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
encode = ['sex','island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # 첫 번째 행 (사용자 입력 데이터) 만 선택합니다.

# 사용자 입력 피쳐값을 표시합니다.
st.subheader('사용자 입력 features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('CSV 파일 업로드를 기다리고 있습니다. 현재 예제 입력 매개 변수를 사용하고 있습니다 (아래 참조).')
    st.write(df)

# 저장된 분류 모델에서 읽습니다.
load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

# 모델을 적용하여 예측하기
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('예측')
penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(penguins_species[prediction])

st.subheader('예측 확률')
st.write(prediction_proba)