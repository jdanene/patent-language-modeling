from google.cloud import bigquery
import os


def _setSchema(client, source_table_id: str,destination_table_id: str):
	"""
	Set the schema of `source_table_id` to be that of `destination_table_id`


    Keyword arguments:
    source_table_id (string) -- tablename (eg project-name.database-name.table-name) of table being copied
    destination_table_id -- tablename (eg project-name.database-name.table-name) of table getting new schema
	
	Note:
	Based on https://cloud.google.com/bigquery/docs/managing-table-schemas
	"""

	client.delete_table(destination_table_id, not_found_ok=True)
	client.create_table(destination_table_id)
	# Get the schema from original table
	orginal_table = client.get_table(source_table_id)  # Make an API request.
	original_schema = orginal_table.schema

	#Get the schema from the new table
	new_table = client.get_table(destination_table_id)  # Make an API request.
	new_table.schema = original_schema
	client.update_table(new_table, ["schema"])  # Make an API request.

def copyTable(client,source_table_id,destination_table_id):
	"""
	Hardcoded to copy the table for our CPC filters for grant_date>=19500101 AND grant_date < 19550101
	So it only has about 90k rows

	Based on: https://cloud.google.com/bigquery/docs/tables#copyingtable 
	"""

	_setSchema(client,source_table_id,destination_table_id)
	job_config = bigquery.QueryJobConfig(destination=destination_table_id)

	query="""
	SELECT P.*
	FROM `{0}` as P, unnest(cpc) as cpc
	WHERE 
		(substr(cpc.code, 1,4) IN ('A01G', 'A01H', 'A61K', 'A61P', 'A61Q', 'B01F', 'B01J', 'B81B', 'B82B', 'B82Y',
		'G01N', 'G16H')
				OR substr(cpc.code, 1,3) IN ('C05', 'C07', 'C08', 'C09', 'C11', 'C12', 'C13', 'C25', 'C40'))
		AND (P.grant_date>=19500101 AND P.grant_date < 19550101)
	""".format(source_table_id)

	# Start the query, passing in the extra configuration.
	query_job = client.query(query, job_config=job_config)  # Make an API request.
	query_job.result()  # Wait for the job to complete.

	print("Query results loaded to the table {}".format(source_table_id))


if __name__ == "__main__": 

	API_KEY_PATH = "/Users/jideofor/Documents/cs397/Patents-Research-abd8b4aaf0a8.json"
	os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = API_KEY_PATH
	PROJECT_ID = "patents-research-275923"

	#  Set source_table_id to the ID of the original table.
	source_table_id = "patents-public-data.patents.publications"

	#Set destination_table_id to the ID of the destination table.
	destination_table_id = "patents-research-275923.patents.publications".format(PROJECT_ID)

	client = bigquery.Client()

	#populate the newtable with a copy of a slice of the old
	copyTable(client,source_table_id,destination_table_id)


