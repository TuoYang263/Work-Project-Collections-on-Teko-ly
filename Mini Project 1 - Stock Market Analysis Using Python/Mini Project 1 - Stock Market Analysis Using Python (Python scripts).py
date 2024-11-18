#!/usr/bin/env python
# coding: utf-8

# ## 1.Read the Data from Yahoo finance website directly, collect stock data of four various industries in the last one year

# In[16]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
# use this library to call yahoo publicly available APIs to download finance data from yahoo finance
import yfinance as yf  
plt.rcParams.update({'font.size': 15})


# In[17]:


start_date = "2023-01-22"
end_date = "2024-01-22"

"""
Define the ticket list
AAPL: Apple
GOOG: Google
MSFT: Microsoft
AMZN: Amazon
"""
tickers_list = ["AAPL", "GOOG", "MSFT", "AMZN"]

four_industries_data = yf.download(tickers_list, start=start_date, end=end_date)


# In[18]:


# display first 5 rows of the data
four_industries_data.head()


# In[19]:


# check the architecture of dataframes
"""
Notes for dataframe columns 
o Open = the price when the market opened in the morning.
o Close = the price when the market closed in the afternoon.
o High = the highest price during that trading day.
o Low = the lowest price during that trading day.
o Volume = number of shares of the stock traded that day.
o Adj Close (Adjusted Close) = a price adjusted to make prices comparable over time.
"""
four_industries_data.info()


# ## 2.Perform cleaning

# ### Detect if data contains null values, and then remove rows if they contained any null values

# In[20]:


four_industries_data.isnull().sum()


# ### Remove if the data contains any duplicates, it it does, remove them

# In[21]:


# make counts on duplicated rows
four_industries_data.duplicated().sum()


# ## 3.What was the change in stock price over time?

# In[22]:


# get all four stocks' stock price (closing price) in the past one year
four_ind_close_price = four_industries_data["Close"]

# create a figure
four_ind_close_price.plot(figsize=(20, 10))

plt.title("Stock Price over Time ("+ str(start_date) + " - " + str(end_date) + ")")
plt.xlabel("Date")
plt.ylabel("Close Price (USD)")
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
plt.show()


# **From the the figure above we can tell except for Microsoft's stock, stock prices of Amazon, Google, and Apple generally remains slowly increase in the past year, and their closing price increasing amplitude overall stays around 50.**
# 
# **Among all these stocks, the stock price of Microsoft has the most highest increasing amplitude, up to 150, and overall stays the increasing trend**

# ## 4.Visualize the change in a stock’s volume being traded, over time?

# In[23]:


# get all four stocks' traded volume in the past one year to be modified
four_ind_volume_traded = four_industries_data["Volume"]
appl_volume_traded = four_ind_volume_traded["AAPL"]
goog_volume_traded = four_ind_volume_traded["GOOG"]
msft_volume_traded = four_ind_volume_traded["MSFT"]
amzn_volume_traded = four_ind_volume_traded["AMZN"]

# create subplots for four different industries
fig, ax = plt.subplots(4, 1, figsize=(18, 17))

# create a bar plot to visualize the Apple's stock's volume change over time
ax[0].bar(appl_volume_traded.index, appl_volume_traded.values, color='green', alpha=0.7)
ax[0].set_title("Apple Stock Volume Traded over Time ("+ str(start_date) + " - " + str(end_date) + ")")
ax[0].set_xlabel("Date")
ax[0].set_ylabel("Stock Volume Traded")
ax[0].grid(which="major", color='k', linestyle='-.', linewidth=0.5)

# create a bar plot to visualize the Google's stock's volume change over time
ax[1].bar(goog_volume_traded.index, goog_volume_traded.values, color='blue', alpha=0.7)
ax[1].set_title("Google Stock Volume Traded over Time ("+ str(start_date) + " - " + str(end_date) + ")")
ax[1].set_xlabel("Date")
ax[1].set_ylabel("Stock Volume Traded")
ax[1].grid(which="major", color='k', linestyle='-.', linewidth=0.5)

# create a bar plot to visualize the Microsoft's stock's volume change over time
ax[2].bar(msft_volume_traded.index, msft_volume_traded.values, color='red', alpha=0.7)
ax[2].set_title("Microsoft Stock Volume Traded over Time ("+ str(start_date) + " - " + str(end_date) + ")")
ax[2].set_xlabel("Date")
ax[2].set_ylabel("Stock Volume Traded")
ax[2].grid(which="major", color='k', linestyle='-.', linewidth=0.5)

# create a bar plot to visualize the Amazon's stock's volume change over time
ax[3].bar(amzn_volume_traded.index, amzn_volume_traded.values, color='brown', alpha=0.7)
ax[3].set_title("Amazon Stock Volume Traded over Time ("+ str(start_date) + " - " + str(end_date) + ")")
ax[3].set_xlabel("Date")
ax[3].set_ylabel("Stock Volume Traded")
ax[3].grid(which="major", color='k', linestyle='-.', linewidth=0.5)

# add a title to the Figure
fig.suptitle("Stock Volume Being Traded over Time")

# tight the layout of plots to make it tidy
plt.tight_layout()

# show the figure
plt.show()


# ## 5.What was the moving average of various stocks?

# In[24]:


# use the 30 days as the window to compute the moving average of various stocks
# get all four stocks' closing price in the past one year
four_ind_closed_price = four_industries_data["Close"]
appl_closed_price = four_ind_closed_price["AAPL"]
goog_closed_price = four_ind_closed_price["GOOG"]
msft_closed_price = four_ind_closed_price["MSFT"]
amzn_closed_price = four_ind_closed_price["AMZN"]

# create subplots for four different industries
fig, ax = plt.subplots(4, 1, figsize=(18, 17))

# create plots to visualize the Apple's closing price and 30-day moving average
appl_closed_price.plot(label="Apple Closing Price", linewidth=2, color='blue', ax=ax[0])
appl_closed_price.rolling(window=30).mean().plot(label='Apple 30-Day Avg', linestyle='--', color='orange', ax=ax[0])
ax[0].set_title("Apple Stock Closing Prices with 30-Day Moving Average ("+ str(start_date) + " - " + str(end_date) + ")")
ax[0].set_xlabel("Date")
ax[0].set_ylabel("Close Price (USD)")
ax[0].legend()
ax[0].grid(which="major", color='k', linestyle='-.', linewidth=0.5)

# create plots to visualize the Google's closing price and 30-day moving average
goog_closed_price.plot(label="Google Closing Price", linewidth=2, color='green', ax=ax[1])
goog_closed_price.rolling(window=30).mean().plot(label='Google 30-Day Avg', linestyle='--', color='orange', ax=ax[1])
ax[1].set_title("Google Closing Prices with 30-Day Moving Average ("+ str(start_date) + " - " + str(end_date) + ")")
ax[1].set_xlabel("Date")
ax[1].set_ylabel("Close Price (USD)")
ax[1].legend()
ax[1].grid(which="major", color='k', linestyle='-.', linewidth=0.5)

# create plots to visualize the Microsoft's closing price and 30-day moving average
msft_closed_price.plot(label="Microsoft Closing Price", linewidth=2, color='purple', ax=ax[2])
msft_closed_price.rolling(window=30).mean().plot(label='Microsoft 30-Day Avg', linestyle='--', color='orange', ax=ax[2])
ax[2].set_title("Microsoft Closing Prices with 30-Day Moving Average ("+ str(start_date) + " - " + str(end_date) + ")")
ax[2].set_xlabel("Date")
ax[2].set_ylabel("Close Price (USD)")
ax[2].legend()
ax[2].grid(which="major", color='k', linestyle='-.', linewidth=0.5)

# create plots to visualize the Amazon's closing price and 30-day moving average
amzn_closed_price.plot(label="Amazon Closing Price", linewidth=2, color='brown', ax=ax[3])
amzn_closed_price.rolling(window=30).mean().plot(label='Amazon 30-Day Avg', linestyle='--', color='orange', ax=ax[3])
ax[3].set_title("Amazon Closing Prices with 30-Day Moving Average ("+ str(start_date) + " - " + str(end_date) + ")")
ax[3].set_xlabel("Date")
ax[3].set_ylabel("Close Price (USD)")
ax[3].legend()
ax[3].grid(which="major", color='k', linestyle='-.', linewidth=0.5)

# add a title to the Figure
fig.suptitle("Closing Prices with 30-Day Moving Average")

# tight the layout of plots to make it tidy
plt.tight_layout()

# show the figure
plt.show()


# ## 6.What was the daily return average of a stock?

# In[25]:


# get all four stocks' adjusted closing price in the past one year
four_ind_closed_price = four_industries_data["Adj Close"]
appl_adj_closed_price = four_ind_closed_price["AAPL"]
goog_adj_closed_price = four_ind_closed_price["GOOG"]
msft_adj_closed_price = four_ind_closed_price["MSFT"]
amzn_adj_closed_price = four_ind_closed_price["AMZN"]


# In[26]:


# compute daily return average
appl_daily_return_avg = appl_adj_closed_price.pct_change(1).mean()
goog_daily_return_avg = goog_adj_closed_price.pct_change(1).mean()
msft_daily_return_avg = msft_adj_closed_price.pct_change(1).mean()
amzn_daily_return_avg = amzn_adj_closed_price.pct_change(1).mean()

print("Daily return average between {} and {}".format(start_date, end_date))
print("Apple: {:.8f}".format(appl_daily_return_avg))
print("Google: {:.8f}".format(goog_daily_return_avg))
print("Microsoft: {:.8f}".format(msft_daily_return_avg))
print("Amazon: {:.8f}".format(amzn_daily_return_avg))


# ## 7.Add a new column ‘Trend’ whose values are based on the 'Daily Return'.
# 

# In[27]:


# compute daily returns of each stock
appl_daily_return = appl_adj_closed_price.pct_change()
goog_daily_return = goog_adj_closed_price.pct_change()
msft_daily_return = msft_adj_closed_price.pct_change()
amzn_daily_return = amzn_adj_closed_price.pct_change()

# add them to data frames
four_industries_data['Trend', 'AAPL'] = appl_daily_return
four_industries_data['Trend', 'AMZN'] = amzn_daily_return
four_industries_data['Trend', 'GOOG'] = goog_daily_return
four_industries_data['Trend', 'MSFT'] = msft_daily_return

# check the data added
four_industries_data['Trend'].head()


# ## 8.Visualize trend frequency through a Pie Chart.

# In[28]:


# check the minimum and maximum in four different stocks
apple_trend = four_industries_data['Trend', 'AAPL']
google_trend = four_industries_data['Trend', 'GOOG']
microsoft_trend = four_industries_data['Trend', 'MSFT']
amazon_trend = four_industries_data['Trend', 'AMZN']

print("Range of Trend of Apple Stock: [{:.8f}, {:.8f}]".format(apple_trend.min(), apple_trend.max()))
print("Range of Trend of Google Stock: [{:.8f}, {:.8f}]".format(google_trend.min(), google_trend.max()))
print("Range of Trend of Microsoft Stock: [{:.8f}, {:.8f}]".format(microsoft_trend.min(), microsoft_trend.max()))
print("Range of Trend of Amazon Stock: [{:.8f}, {:.8f}]".format(amazon_trend.min(), amazon_trend.max()))


# In[29]:


# define the bins to split the trend data into smaller segments
step = 0.03
uniform_bins =  np.arange(-0.09, 0.09 + step, step).tolist()
uni_labels = ['-9% to -6%', '-6% to -3%', '-3% to 0', '0 to 3%', '3% to 6%', '6% to 9%']

# group the data by returning categories
appl_return_categories = pd.cut(apple_trend, bins=uniform_bins, labels=uni_labels)
goog_return_categories = pd.cut(google_trend, bins=uniform_bins, labels=uni_labels)
msft_return_categories = pd.cut(microsoft_trend, bins=uniform_bins, labels=uni_labels)
amzn_return_categories = pd.cut(amazon_trend, bins=uniform_bins, labels=uni_labels)

# count the frequency of each category
appl_return_counts = appl_return_categories.value_counts()
appl_return_counts = appl_return_categories.value_counts()
# define the bins to split the trend data into smaller segments
step = 0.03
uniform_bins =  np.arange(-0.09, 0.09 + step, step).tolist()
uni_labels = ['-9% to -6%', '-6% to -3%', '-3% to 0', '0 to 3%', '3% to 6%', '6% to 9%']

# group the data by returning categories
appl_return_categories = pd.cut(apple_trend, bins=uniform_bins, labels=uni_labels)
goog_return_categories = pd.cut(google_trend, bins=uniform_bins, labels=uni_labels)
msft_return_categories = pd.cut(microsoft_trend, bins=uniform_bins, labels=uni_labels)
amzn_return_categories = pd.cut(amazon_trend, bins=uniform_bins, labels=uni_labels)

# count the frequency of each category
appl_return_counts = appl_return_categories.value_counts()
goog_return_counts = goog_return_categories.value_counts()
msft_return_counts = msft_return_categories.value_counts()
amzn_return_counts = amzn_return_categories.value_counts()


# In[69]:


# draw four pie charts to display the trend frequency
# create subplots for four different industries

colors_list = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen', 'pink', 'cornflowerblue']

# Ratio for each continent with which to offset each wedge.
explode_list = [0.1, 0.1, 0.1, 0, 0, 0] 

fig, ax = plt.subplots(2, 2, figsize=(19, 18))

# Apple
ax[0, 0].pie(appl_return_counts,
             startangle=90,
             shadow=True,
             colors=colors_list,
             explode=explode_list,  # 'explode' lowest 3 continents
            )
# make sure the pie is shown as a circle
ax[0, 0].axis('equal')
# disable y-axis label
ax[0, 0].yaxis.label.set_visible(False)
# compute percents
appl_percents = appl_return_counts.to_numpy() * 100 / appl_return_counts.to_numpy().sum()
# add the legend
ax[0, 0].legend(['%s, %1.1f %%' % (l,s) for l,s in zip(appl_return_counts.index, appl_percents)], 
                                  loc='best', bbox_to_anchor=(-0.0001, 1.), fontsize=15)

# add a title
ax[0, 0].set_title('Apple Trend Frequency')

# Google
ax[0, 1].pie(goog_return_counts,
             startangle=90,
             shadow=True,
             colors=colors_list,
             explode=explode_list,  # 'explode' lowest 3 continents
            )
# make sure the pie is shown as a circle
ax[0, 1].axis('equal')
# disable y-axis label
ax[0, 1].yaxis.label.set_visible(False)
# compute percents
goog_percents = goog_return_counts.to_numpy() * 100 / goog_return_counts.to_numpy().sum()
# add the legend
ax[0, 1].legend(['%s, %1.1f %%' % (l,s) for l,s in zip(goog_return_counts.index, goog_percents)], 
                                  loc='best', bbox_to_anchor=(-0.0001, 1.), fontsize=15)
# add a title
ax[0, 1].set_title('Google Trend Frequency')

# Microsoft
ax[1, 0].pie(msft_return_counts,
             startangle=90,
             shadow=True,
             colors=colors_list,
             explode=explode_list,  # 'explode' lowest 3 continents
            )
# make sure the pie is shown as a circle
ax[1, 0].axis('equal')
# disable y-axis label
ax[1, 0].yaxis.label.set_visible(False)
# compute percents
msft_percents = msft_return_counts.to_numpy() * 100 / msft_return_counts.to_numpy().sum()
# add the legend
ax[1, 0].legend(['%s, %1.1f %%' % (l,s) for l,s in zip(msft_return_counts.index, msft_percents)], 
                                  loc='best', bbox_to_anchor=(-0.0001, 1.), fontsize=15)
# add a title
ax[1, 0].set_title('Microsoft Trend Frequency')

# Amazon
ax[1, 1].pie(amzn_return_counts,
             startangle=90,
             shadow=True,
             colors=colors_list,
             explode=explode_list,  # 'explode' lowest 3 continents
            )
# make sure the pie is shown as a circle
ax[1, 1].axis('equal')
# disable y-axis label
ax[1, 1].yaxis.label.set_visible(False)
# compute percents
amzn_percents = amzn_return_counts.to_numpy() * 100 / amzn_return_counts.to_numpy().sum()
# add the legend
ax[1, 1].legend(['%s, %1.1f %%' % (l,s) for l,s in zip(amzn_return_counts.index, amzn_percents)], 
                                  loc='best', bbox_to_anchor=(-0.0001, 1.), fontsize=15)
# add a title
ax[1, 1].set_title('Amazon Trend Frequency')

plt.tight_layout()

plt.show()


# ## 9.What was the correlation between the daily returns of different stocks?

# In[77]:


# draw the correlation heatmap
correlation_matrix = four_industries_data['Trend'].dropna().reset_index().corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title(f'Correlation Heatmap for four industries daily returns')
plt.show()


# ####  Through the comparsion of the following table below, from the figure above we can tell the correlation between daily returns of Amazon and Apple is weak-positively correlated. ####
# 
# #### The correlations between daily returns of Google and Apple, Microsoft and Apple, Amazon and Google, Amazon and Microsoft are strongly correlated ####

# ![image.png](attachment:image.png)
# 
# Image source: https://www.scribbr.com/statistics/correlation-coefficient/
