# -*- coding: utf-8 -*-
"""
Version:1.0

@author: Tuo Yang
"""
"""
The codes of this file will build a dashboard to analyze 
stocks. Dash will be used here, which is a python framework that
provides an abstraction over flask and react.js to buid analytical
web applications
"""
# import libraries used in this project
import dash
import dash_core_components as doc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = dash.Dash()
server = app.server

"""
codes below extracted from stock_prediction.ipynb
to process the data from NSE-Tata-Global-Beverages-Limited.csv
"""
"""
2. Read the dataset
"""
nse_tata = pd.read_csv("./NSE-Tata-Global-Beverages-Limited.csv")

"""
3. Making analysis to closing prices from dataframe
"""
# Changing the date format of the data attribute to make it is sorted in the order of year-month-day
nse_tata["Date"] = pd.to_datetime(nse_tata["Date"], format="%Y-%m-%d")
nse_tata.index = nse_tata["Date"]

"""
4. Sort the dataset on date time and filter "Date" and "Close" columns
"""
# sorting the date attribute in the ascending order
sorted_data = nse_tata.sort_index(ascending=True, axis=0)
# creating a new dataframe only contains two attributes of the original frame ("Date" and "Close")
new_dataset = pd.DataFrame(index=range(0, len(nse_tata)), columns=['Date', 'Close'])

for i in range(0, len(nse_tata)):
    new_dataset['Date'][i] = sorted_data['Date'][i]
    new_dataset['Close'][i] = sorted_data['Close'][i]

"""
5. Normalize the new filtered dataset
"""
# Use data coming from the column 'Date' as the index column, making sure it plotted out in the figure
new_dataset.index = new_dataset.Date
# Drop the column "Date" in the new dataset, the missing values will be filled out
new_dataset.drop("Date", axis=1, inplace=True)

# Take values of new_dataset as the final dataset used for training
final_dataset = new_dataset.values

# Use the first 987 out of 1235 instances as the training dataset, 
# the rest is used as the validation dataset
train_data = final_dataset[0:987, :]
valid_data = final_dataset[987:, :]

# Use min-max scaling to do the normalization to the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(final_dataset)

# Build the training dataset and testing dataset
x_train_data, y_train_data = [], []
for i in range(60, len(train_data)):
    # for each instance, every 60 previous records as the inputs of LSTM network, 
    # the instance itself is used as prediction results
    x_train_data.append(scaled_data[i-60:i, 0])  
    y_train_data.append(scaled_data[i, 0])

x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

# loading the trained model from h5 file to make prediction
lstm_model = load_model('lstm_model.h5')

# Building testing dataset for making prediction, every 60 instances in the training dataset
# corresponds to 60 testing samples
test_inputs = new_dataset[len(new_dataset)-len(valid_data)-60:].values
test_inputs = test_inputs.reshape(-1, 1)
test_inputs = scaler.transform(test_inputs) # normalization to the test data

"""
7. Take a sample of a dataset to make stock price predictions using the LSTM model
"""
# Put all testing samples into a list
X_test = []
for i in range(60, test_inputs.shape[0]):
    X_test.append(test_inputs[i-60:i, 0])
X_test = np.array(X_test)

# Add another dimension to the testing data for inputs of CNN
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_closing_price = lstm_model.predict(X_test)
# Perform anti-normalization operation to prediction results
predicted_closing_price = scaler.inverse_transform(predicted_closing_price)

# Build three datasets used for the visualization
# training dataset, testing dataset, and prediction results from testing
# dataset
train_data = new_dataset[:987]
valid_data = new_dataset[987:]
valid_data['Predictions'] = predicted_closing_price

# Loading data from another csv file into DataFrames
stock_data = pd.read_csv('./stock_data.csv')
app.layout = html.Div([
    # Web page title
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
    
    # Creat sub pages of tabs
    doc.Tabs(id="tabs", children=[
        
        # The first tab in 
        doc.Tab(label='NSE-TATAGLOBAL Stock Data', children=[
            # The DIV layer of the first tab
            html.Div([
                # The title of the 1st figure in the 1st tab
                html.H2("Actual closing price", style={"textAlign": "center"}),
                # The Graph component used for putting figures
                doc.Graph(
                    id = "Actual Data",
                    figure={
                        "data":[
                            go.Scatter(
                                x = train_data.index,
                                y = valid_data["Close"],
                                mode = 'markers')
                            ],
                        "layout":go.Layout(
                            title='scatter plot',
                            xaxis={'title':'Dates'},
                            yaxis={'title':'Closing Price'}
                            )
                        }
                    ),
                html.H2("LSTM Predicted Closing Price", 
                        style={"textAlign": "center"}),
                doc.Graph(
                    id="Predicted Data",
                    figure={
                        "data":[
                            go.Scatter(
                                x = valid_data.index,
                                y = valid_data["Predictions"],
                                mode = 'markers'
                                )
                            ],
                        "layout":go.Layout(
                            title='scatter plot',
                            xaxis={'title': 'Dates'},
                            yaxis={'title': 'Closing Price'}
                                )
                            }
                    )
                ])
            ]),
        # The second tab of the web page
        doc.Tab(label='Facebook Stock Data', children=[
            html.Div([
                # The title of the second tab
                html.H1("Facebook Stocks High vs Lows", 
                        style={'textAlign': 'center'}),
                
                # The 1st Dropdown component 
                doc.Dropdown(id='my-dropdown',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple', 'value': 'AAPL'},
                                      {'label': 'Facebook', 'value': 'FB'},
                                      {'label': 'Microsoft', 'value': 'MSFT'}],
                             # The mulitple options of dropdown are enabled, 
                             # with the default value 'FB'
                             multi=True, value=['FB'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                doc.Graph(id='highlow'),
                html.H1("Facebook Market Volume", style={'textAlign':'center'}),
                
                # The 2nd Dropdown component 
                doc.Dropdown(id='my-dropdown2',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple', 'value': 'AAPL'},
                                      {'label': 'Facebook', 'value': 'FB'},
                                      {'label': 'Microsoft', 'value': 'MSFT'}],
                             # The mulitple options of dropdown are enabled, 
                             # with the default value 'FB'
                             multi=True, value=['FB'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                doc.Graph(id='volume')
                ], className="container"),
            ])
        ])
    ])

@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"TSLA": "Tesla", "AAPL": "Apple", "FB": "Facebook",
                "MSFT": "Microsoft"}
    trace1 = []
    trace2 = []
    # The link will jump to various pages according to various
    # selections in the Dropdown
    for stock in selected_dropdown:
        trace1.append(
            go.Scatter(x = stock_data[stock_data["Stock"] == stock]["Date"],
                       y = stock_data[stock_data["Stock"] == stock]["High"],
                       mode = "lines", opacity=0.7,
                       name = f'High {dropdown[stock]}',
                       textposition='bottom center'))
        trace2.append(
            go.Scatter(x = stock_data[stock_data["Stock"] == stock]["Date"],
                       y = stock_data[stock_data["Stock"] == stock]["Low"],
                       mode = "lines", opacity=0.6,
                       name = f'Low {dropdown[stock]}',
                       textposition='bottom center'))
    traces = [trace1, trace2]
    # Get every data in traces plotted in the figure
    data = [val for sublist in traces for val in sublist]
    figure = {'data':data,
              'layout': go.Layout(colorway=['#5E0DAC', '#FF4F00', '#375CB1',
                                            '#FF7400', '#FFF400', '#FF0056'],
                                  height=600,
    title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
    xaxis={"title": "Date",
           'rangeselector': {'buttons': list([{'count': 1,  'label': '1M',
                                              'step': 'month', 'stepmode': 'backward'},
                                              {'count': 6, 'label': '6M',
                                               'step': 'month', 'stepmode': 'backward'},
                                              {'step': 'all'}])},
           'rangeslider': {'visible': True}, 'type': 'date'},
    yaxis={"title":"Price (USD)"})}
    return figure

@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = {"TSLA": "Tesla", "AAPL": "Apple", "FB": "Facebook",
                "MSFT": "Microsoft"}
    trace1 = []
    # The link will jump to various pages according to various
    # selections in the Dropdown
    for stock in selected_dropdown_value:
        trace1.append(
            go.Scatter(x = stock_data[stock_data["Stock"] == stock]["Date"],
                       y = stock_data[stock_data["Stock"] == stock]["High"],
                       mode = "lines", opacity=0.7,
                       name = f'Volume {dropdown[stock]}',
                       textposition='bottom center'))
    traces = [trace1]
    # Get every data in traces plotted in the figure
    data = [val for sublist in traces for val in sublist]
    figure = {'data':data,
              'layout': go.Layout(colorway=['#5E0DAC', '#FF4F00', '#375CB1',
                                            '#FF7400', '#FFF400', '#FF0056'],
                                  height=600,
    title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
    xaxis={"title": "Date",
           'rangeselector': {'buttons': list([{'count': 1,  'label': '1M',
                                              'step': 'month', 'stepmode': 'backward'},
                                              {'count': 6, 'label': '6M',
                                               'step': 'month', 'stepmode': 'backward'},
                                              {'step': 'all'}])},
           'rangeslider': {'visible': True}, 'type': 'date'},
    yaxis={"title":"Transactions Volume"})}
    return figure

if __name__=='__main__':
    app.run_server(debug=True)

                
                