import streamlit as st
import pandas as pd
df = pd.read_csv('http://ai.shop2world.net/data/penguins.csv')

st.write("""
# 모델 생성 앱 - 하드디스크에 penguins_clf.pkl 이 생성 되었습니다.
""")

# 서수형 피처 인코딩
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
#df = penguins.copy()
target = 'species' #예측하고 싶은것  
encode = ['sex','island'] #인풋값 

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}#종 인코딩 
def target_encode(val):
    return target_mapper[val]

df['species'] = df['species'].apply(target_encode)

# X 와 y 분리해서 sklearn 모델 구축에 사용 
X = df.drop('species', axis=1)
Y = df['species']

# 모델 구축 (random forest model)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# 모델 저장
import pickle
pickle.dump(clf, open('penguins_clf.pkl', 'wb'))


