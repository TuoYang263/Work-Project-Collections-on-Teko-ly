import os
import json
from utils import config
from google.cloud import bigquery
from google.oauth2 import service_account

def get_bigquery_client():
    return bigquery.Client()