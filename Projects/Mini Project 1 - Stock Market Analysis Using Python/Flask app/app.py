from flask import Flask, render_template, request, redirect, url_for
from concurrent.futures import ThreadPoolExecutor
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import yfinance as yf

app = Flask(__name__)

executor = ThreadPoolExecutor(max_workers=4)
tasks = {}

# Utility functions
def download_stock_data(tickers, start_date, end_date):
    try:
        return yf.download(tickers, start=start_date, end=end_date)
    except Exception as e:
        raise RuntimeError(f"Error fetching data: {e}")

# Plotting functions with Plotly
def generate_close_price_plot(data):
    fig = go.Figure()
    for ticker in data['Close'].columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'][ticker], mode='lines', name=ticker))
    fig.update_layout(title='Stock Prices Over Time', xaxis_title='Date', yaxis_title='Price')
    return fig.to_html(full_html=True)

def generate_volume_plot(data, tickers):
    fig = make_subplots(rows=len(tickers), cols=1, shared_xaxes=True, subplot_titles=tickers)
    for i, ticker in enumerate(tickers):
        fig.add_trace(go.Bar(x=data.index, y=data['Volume'][ticker], name=f"{ticker} Volume"), row=i+1, col=1)
    fig.update_layout(title=f"Volume Traded", height=300*len(tickers), xaxis_title='Date', yaxis_title='Stock Volume Traded')
    return fig.to_html(full_html=True)

def generate_moving_avg_plot(data, tickers):
    fig = make_subplots(rows=len(tickers), cols=1, shared_xaxes=True, subplot_titles=tickers)
    for i, ticker in enumerate(tickers):
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'][ticker], mode='lines', name='Close Price'), row=i+1, col=1)
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'][ticker].rolling(window=30).mean(),
                mode='lines',
                name='30-Day Avg',
                line=dict(dash='dash')
            ),
            row=i+1,
            col=1
        )
    fig.update_layout(title='30-Day Moving Averages', height=300 * len(tickers))
    return fig.to_html(full_html=True)

def generate_pie_chart(tickers, start_date, end_date):
    # Create a list of specs, one for each ticker, specifying "domain" for pie charts
    specs = [[{'type': 'domain'}] for _ in tickers]

    # Create the subplot with the correct specs for pie charts
    fig = make_subplots(rows=len(tickers), cols=1, subplot_titles=tickers, specs=specs)
    bins = [-0.09, -0.06, -0.03, 0, 0.03, 0.06, 0.09]
    labels = ['-9% to -6%', '-6% to -3%', '-3% to 0%', '0% to 3%', '3% to 6%', '6% to 9%']
    
    for i, ticker in enumerate(tickers):
        stock_data = yf.Ticker(ticker).history(start=start_date, end=end_date)
        returns = stock_data['Close'].pct_change()
        categorized = pd.cut(returns, bins=bins, labels=labels)
        counts = categorized.value_counts().sort_index()
        # Add a pie chart to the correct row and column
        fig.add_trace(go.Pie(labels=counts.index.astype(str), values=counts, name=ticker), row=i+1, col=1)
    # Update the layout
    fig.update_layout(title='Trend Frequencies', height=400 * len(tickers))
    return fig.to_html(full_html=False)

# Task executor
def generate_plots(task_id, tickers, start_date, end_date):
    try:
        data = download_stock_data(tickers, start_date, end_date)

        plots = {
            "close_price_plot": generate_close_price_plot(data),
            "volume_plot": generate_volume_plot(data, tickers),
            "moving_avg_plot": generate_moving_avg_plot(data, tickers),
            "pie_chart_plot": generate_pie_chart(tickers, start_date, end_date)
        }

        tasks[task_id] = {"status": "completed", **plots}
    except Exception as e:
        tasks[task_id] = {"status": "error", "message": str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def results():
    tickers = request.form.get('tickers').split(',')
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')

    task_id = len(tasks) + 1
    tasks[task_id] = {"status": "running"}
    executor.submit(generate_plots, task_id, tickers, start_date, end_date)

    return render_template('waiting.html', task_id=task_id, status="running")

@app.route('/status/<int:task_id>')
def task_status(task_id):
    task = tasks.get(task_id, {"status": "unknown"})
    if task["status"] == "completed":
        return redirect(url_for('display_plots', task_id=task_id))
    return render_template('waiting.html', task_id=task_id, status=task["status"])

@app.route('/display/<int:task_id>')
def display_plots(task_id):
    task = tasks.get(task_id)
    if task and task["status"] == "completed":
        return render_template(
            "results.html",
            close_price_plot=task["close_price_plot"],
            volume_plot=task["volume_plot"],
            moving_avg_plot=task["moving_avg_plot"],
            pie_chart_plot=task["pie_chart_plot"]
        )
    return "Task not completed or does not exist", 404

if __name__ == '__main__':
    app.run(debug=True)