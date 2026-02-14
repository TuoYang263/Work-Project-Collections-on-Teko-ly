FROM python:3.10-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Install dependencies
COPY streamlit_app/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy app code
COPY streamlit_app/ /app/

# Copy start script (from .streamlit)
COPY .streamlit/start.sh /app/start.sh
RUN sed -i 's/\r$//' /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8501
CMD ["/app/start.sh"]