# How to install bq_helper https://github.com/SohierDane/BigQuery_Helper/blob/master/bq_helper.py
import pandas as pd
import os
from bq_helper import BigQueryHelper

#Based on: https://www.kaggle.com/jessicali9530/how-to-query-google-patents-public-data
#API_KEY_PATH is the path to the API key json in email. Do not upload the json to github, it's a security risk 
API_KEY_PATH = "/Users/jideofor/Documents/cs397/Patents-Research-abd8b4aaf0a8.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = API_KEY_PATH
PROJECT_ID = "patents-research-275923"


bq_assistant = BigQueryHelper("patents-public-data", "patents")

pdDF = bq_assistant.head("publications", num_rows=3)
bq_assistant.table_schema("publications")

# citations
pdDF["citation"]
pdDF["publication_date"]
