from google.cloud import bigquery
import os
import pandas as pd 
API_KEY_PATH = "/Users/jideofor/Documents/cs397/Patents-Research-abd8b4aaf0a8.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = API_KEY_PATH
PROJECT_ID = "patents-research-275923"

#Set destination_table_id to the ID of the destination table.
table_id = "patents-research-275923.patents.publications".format(PROJECT_ID)
DANGEROUS_tableId = "patents-public-data.patents.publications"
# USE: DANGEROUS_tableId for descrptive stats  be careful!
# USE: table_id for everything else. 


def getStats(query):
	"""
	Hardcoded to copy the table for our CPC filters for grant_date>=19500101 AND grant_date < 19550101
	So it only has about 90k rows

	Based on: https://cloud.google.com/bigquery/docs/tables#copyingtable 
	"""


	# Start the query using pandas based on - https://github.com/google/patents-public-data/blob/master/examples/claim-text/claim_text_extraction.ipynb
	df = pd.read_gbq(query, project_id=PROJECT_ID, dialect='standard', progress_bar_type='tqdm')

	print(df.head())
	return df

# USE: DANGEROUS_tableId for descrptive stats  be careful!
# USE: table_id for everything else. 

query="""
SELECT COUNT(DISTINCT publication_number) AS count
FROM `{0}`,unnest(cpc) as cpc
WHERE 
	(substr(cpc.code, 1,4) IN ('A01G', 'A01H', 'A61K', 'A61P', 'A61Q', 'B01F', 'B01J', 'B81B', 'B82B', 'B82Y',
	'G01N', 'G16H')
			OR substr(cpc.code, 1,3) IN ('C05', 'C07', 'C08', 'C09', 'C11', 'C12', 'C13', 'C25', 'C40'))


""".format(table_id)

# USE: DANGEROUS_tableId for descrptive stats and be careful!!!
# USE: table_id for everything else. 

#populate the newtable with a copy of a slice of the old
df = getStats(query)


