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

# if in the schema its says its a `RECORD` then you have to unset it or do something special cuz records are arrays can
# return arrays in select, just atomic ele like strings, numbers, etc
# can check by looking @the output for a dataset w/ 3 rows
pdDF["cpc.code"]
pdDF["publication_date"]



QUERY="""
SELECT DISTINCT publication_number
FROM `patents-public-data.patents.publications`,
unnest(cpc) as cpc
WHERE substr(cpc.code, 1,4) IN ('A01G', 'A01H') 
OR substr(cpc.code, 1,3) IN ('C07', 'C08', 'C09') 
LIMIT 3;
"""

df = bq_assistant.query_to_pandas(QUERY)
print(df)

