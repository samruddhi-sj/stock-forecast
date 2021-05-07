import streamlit as st
from datetime import date
import time
import pandas as pd
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

def stock_selection():
    url = 'https://in.finance.yahoo.com/quote/%5ENSEI/components/'
    html = pd.read_html(url, header = 0)
    df = html[0]
    return df

df = stock_selection()
stock = df.groupby('Symbol')

sorted_stock_unique = sorted( df['Symbol'].unique() )
selected_stock = st.selectbox('Stock', sorted_stock_unique)


# stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
# selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365



@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


with st.spinner(text='Loading data...'):
    time.sleep(5)
data = load_data(selected_stock)
st.success('Here!')
	
# data_load_state = st.text('Loading data...')
# data = load_data(selected_stock)
# data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write('Data Dimension: ' + str(data.shape[0]) + ' rows and ' + str(data.shape[1]) + ' columns.')
st.write(data.head())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)