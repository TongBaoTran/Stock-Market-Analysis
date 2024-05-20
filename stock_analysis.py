# -*- coding: utf-8 -*-
"""
Created on Sat May 18 23:35:22 2024

@author: baotr
"""

import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
import plotly.express as px
import yfinance as yf
import bs4 as bs
import requests
import datetime
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Stock Market Analysis")

#get list of all tickers
resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
soup = bs.BeautifulSoup(resp.text, 'lxml')
table = soup.find('table', {'class': 'wikitable sortable'})

tickers = []

for row in table.findAll('tr')[1:]:
    ticker = row.findAll('td')[0].text
    tickers.append(ticker)

tickers = [s.replace('\n', '') for s in tickers]



###### Part 1 ##############
st.header("1. Descriptive information of a specific stock")

selected_ticker = st.selectbox('Please choose a sticker?', tickers, key = "1")
st.write('You selected:', selected_ticker)

stock = yf.Ticker(selected_ticker)
stock_share_price_data = stock.history(period="5y")
stock_share_price_data.reset_index(inplace=True)
st.dataframe(stock_share_price_data)


st.subheader("Chart for Open Price of the Stock in recent 5 years: {fname}".format(fname =selected_ticker ))
st.line_chart(stock_share_price_data, x="Date", y="Open")

st.subheader("Chart for Volume of the Stock: {fname}".format(fname =selected_ticker ))
st.bar_chart(stock_share_price_data, x="Date", y="Volume")


#Moving Average
st.subheader("Moving Average : {fname}".format(fname =selected_ticker ))
ma_day = [10,20,50]
for ma in ma_day:
    column_name = "MA for %s days" %(str(ma)) 
    stock_share_price_data[column_name] = stock_share_price_data['Close'].rolling(window=ma,center=False).mean()

st.line_chart(stock_share_price_data, x="Date", y=['Close','MA for 10 days','MA for 20 days','MA for 50 days'])

#Daily Return
st.subheader("Daily Return : {fname}".format(fname =selected_ticker ))
stock_share_price_data['Daily Return'] = stock_share_price_data['Close'].pct_change()

st.line_chart(stock_share_price_data, x="Date", y="Daily Return")

# Create histplot with custom bin_size
st.subheader("Daily Return Histogram : {fname}".format(fname =selected_ticker ))
daily_return = stock_share_price_data['Daily Return']

daily_return= daily_return.dropna()

fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)

# We can set the number of bins with the *bins* keyword argument.
axs.hist(daily_return, bins=20)
st.pyplot(fig)
#fig = ff.create_histplot([daily_return] ,group_labels =["Daily Returns"],  bin_size=[0.5])
#st.plotly_chart(fig, use_container_width=True)
sns.histplot(x=daily_return.dropna(),bins=100,color='red')



############################
###### Part 2 ##############
st.header("2. Correlation between daily returns of different stocks")
#Stock price correletion
options = st.multiselect(
    "Please choose some stocks:",tickers)

st.write("You selected:", options)

if len(options)> 0:
    close_data = pd.DataFrame()
    for ti in options:
        close_price = yf.Ticker(ti).history(period="5y")
        #close_price.reset_index(inplace=True)
        close_data[ti] = close_price["Close"]

    st.write("Daily close prices")
    st.dataframe(close_data)

    rets_df = close_data.pct_change()
    rets_df.dropna(inplace=True)
    st.write("Daily returns")
    st.dataframe(rets_df)

    st.subheader("Daily return correlation between  {stock1} and {stock2}".format(stock1 =options[0], stock2 = options[1] ))
    #fig = plt.figure(figsize=(50, 30))
    plot=sns.jointplot(data=rets_df, x=options[0], y=options[1], kind='scatter')
    st.pyplot(plot)

    st.subheader("Pair plots")
    plot1 = sns.pairplot(rets_df)
    st.pyplot(plot1)  
    plot1.fig.clf()

    st.subheader("Heat Map")
    plot2 = sns.heatmap(rets_df,annot=True)
    st.pyplot(plot2.get_figure())


############################
###### Part 3 ##############
    st.header("3. Dayss vs Risk")
    plt.figure(figsize=(8,5))
    plt.scatter(rets_df.mean(),rets_df.std(),s=25)

    plt.xlabel('Expected Return')
    plt.ylabel('Risk')
  

    #For adding annotatios in the scatterplot
    for label,x,y in zip(rets_df.columns,rets_df.mean(),rets_df.std()):
        plt.annotate(
        label,
        xy=(x,y),xytext=(-120,20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        arrowprops = dict(arrowstyle='->',connectionstyle = 'arc3,rad=-0.5'))
    st.pyplot(plt)


############################
###### Part 4 ##############

st.header("4. Value at Risk (VAR)")

selected = st.selectbox('Please choose a sticker:', tickers, key = "2")
st.write('You selected:', selected)

stock = yf.Ticker(selected)
data = stock.history(period="10y")
data.reset_index(inplace=True)
st.dataframe(data)
data['Daily Return'] = data['Close'].pct_change()

# Create histplot with custom bin_size
st.subheader("Daily Return Histogram : {fname}".format(fname =selected ))
rets = data
rets= rets.dropna()
st.dataframe(rets)

fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
axs.hist(rets['Daily Return'], bins=100,color='red')
st.pyplot(fig)
fig.clf()
#fig = ff.create_histplot([daily_return] ,group_labels =["Daily Returns"],  bin_size=[0.5])
#st.plotly_chart(fig, use_container_width=True)
#sns.histplot(x=rets.dropna(),bins=100,color='red')

quan = rets['Daily Return'].quantile(0.05)
st.write("The 0.05 empirical quantile of daily returns is at {ret}. This means that with 95% confidence, the worst daily loss will not exceed {ret2}% (of the investment).".format(ret = quan, ret2 = abs(quan*100)))

st.header("4. Predict stock price")
print(data.head())
time_elapsed = (data['Date'].iloc[-1] - data['Date'].iloc[0]).days

#Current price / first record (e.g. price at beginning of 2009)
#provides us with the total growth %
total_growth = (data['Close'].iloc[-1] / data['Close'].iloc[1])

#Next, we want to annualize this percentage
#First, we convert our time elapsed to the # of years elapsed
number_of_years = time_elapsed / 365.0
#Second, we can raise the total growth to the inverse of the # of years
#(e.g. ~1/10 at time of writing) to annualize our growth rate
cagr = total_growth ** (1/number_of_years) - 1

#Now that we have the mean annual growth rate above,
#we'll also need to calculate the standard deviation of the
#daily price changes
std_dev = data['Close'].pct_change().std()

#Next, because there are roughy ~252 trading days in a year,
#we'll need to scale this by an annualization factor
#reference: https://www.fool.com/knowledge-center/how-to-calculate-annualized-volatility.aspx

number_of_trading_days = 252
std_dev = std_dev * np.sqrt(number_of_trading_days)


#Now that we've created a single random walk above,
#we can simulate this process over a large sample size to
#get a better sense of the true expected distribution
number_of_trials = 10000

#set up an additional array to collect all possible
#closing prices in last day of window.
#We can toss this into a histogram
#to get a clearer sense of possible outcomes
closing_prices = []

for i in range(number_of_trials):
    #calculate randomized return percentages following our normal distribution
    #and using the mean / std dev we calculated above
    daily_return_percentages = np.random.normal(cagr/number_of_trading_days, std_dev/np.sqrt(number_of_trading_days),number_of_trading_days)+1
    price_series = [data['Close'].iloc[-1]]

    for j in daily_return_percentages:
        #extrapolate price out for next year
        price_series.append(price_series[-1] * j)

    #append closing prices in last day of window for histogram
    closing_prices.append(price_series[-1])

    #plot all random walks
    plt.plot(price_series)
plt.title("Monte Carlo Analysis for {x} (252 trading days)".format(x = selected))
plt.xlabel('Days')
plt.ylabel('Price')
st.pyplot(plt)

mean_end_price = round(np.mean(closing_prices),2)
st.write("Expected price: ", str(mean_end_price))