import yfinance as yf
import streamlit as st
st.title("""간단한 주식 차트 종가(clsoing price)와 거래량(volume)보기 - 테슬라""")
Stock_Symbol = 'TSLA'
StockData = yf.Ticker(Stock_Symbol)
StockChart = StockData.history(period = 'id', start='2019-7-1',end='2020-5-1')
st.line_chart(StockChart.Close)
st.line_chart(StockChart.Volume)