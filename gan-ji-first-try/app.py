import yfinance as yf
import streamlit as st
st.title("""간단한 주식 차트 	종가(closing price)와 거래량 (volume) 보기 - 	테슬라""")
Stock_Symbol = 'TSLA'
StockData = yf.Ticker(Stock_Symbol)
StockChart = StockData.history(period = '1d', start='2019-7-1',end='2020-7-11')
st.line_chart(StockChart.Close)
st.line_chart(StockChart.Volume)# Open	High	Low	Close	Volume	Dividends	Stock Splits
