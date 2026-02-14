FROM python:3.10-slim

WORKDIR /app

# Avoid python buffering
ENV PYTHONUNBUFFERED=1

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the whole app
COPY . /app

# Start Streamlit
# CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Expose port
EXPOSE 8501

CMD ["/app/start.sh"]