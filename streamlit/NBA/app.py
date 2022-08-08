import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title('NBA 선수 통계 제공 서비스')

st.markdown("""
이 앱은 NBA 선수 통계 데이터의 간단한 웹 스크랩을 수행합니다!
* **필요 파이썬 모듈:** base64, pandas, streamlit
* **Data source:** [Basketball-reference.com](https://www.basketball-reference.com/)
""")


st.sidebar.header('통계를 원하는 값 입력')
selected_year = st.sidebar.selectbox('활동 년도', list(reversed(range(1950,2021))))

# 데이터 스크래핑 
@st.cache
def load_data(year):
    url = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_per_game.html" #https://www.basketball-reference.com/leagues/NBA_1999_per_game.html
    html = pd.read_html(url, header = 0)
    df = html[0]
    raw = df.drop(df[df.Age == 'Age'].index) # 반복되는 라인 지워주기
    raw = raw.fillna(0)
    playerstats = raw.drop(['Rk'], axis=1)
    return playerstats
playerstats = load_data(selected_year)

# 팀선택 
sorted_unique_team = sorted(playerstats.Tm.unique())
selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team)

# 포지션 선택
unique_pos = ['C','PF','SF','PG','SG']
selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos)

# 데이터 필터링 
df_selected_team = playerstats[(playerstats.Tm.isin(selected_team)) & (playerstats.Pos.isin(selected_pos))]

st.header('선택한 팀의 선수 통계 표시')
st.write('데이터 구조: ' + str(df_selected_team.shape[0]) + ' 열과  ' + str(df_selected_team.shape[1]) + '행')
test = df_selected_team.astype(str)
st.dataframe(test)

# NBA 선수 통계 데이터 다운로드
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">통계 정보 CSV 파일로 다운받기</a>'
    return href

st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)

# 히트맵 차트 생성
if st.button('상호 상관 관계 히트 맵'):
    st.header('상호 상관 행렬 히트 맵')
    df_selected_team.to_csv('output.csv',index=False)
    df = pd.read_csv('output.csv')

    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
    st.pyplot()