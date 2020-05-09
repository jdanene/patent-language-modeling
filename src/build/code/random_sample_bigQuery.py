from google.cloud import bigquery
import os


#### Update Schema: Based on https://cloud.google.com/bigquery/docs/managing-table-schemas ###
def setSchema(source_table_id,destination_table_id):
	# Get the schema from original table
	orginal_table = client.get_table(source_table_id)  # Make an API request.
	original_schema = orginal_table.schema

	#Get the schema from the new table
	new_table = client.get_table(destination_table_id)  # Make an API request.
	new_table.schema = original_schema
	client.update_table(new_table, ["schema"])  # Make an API request.

#### Create Table from query result: https://cloud.google.com/bigquery/docs/tables#copyingtable ########
def copyTable(source_table_id,destination_table_id)
	job_config = bigquery.QueryJobConfig(destination=destination_table_id)

	QUERY= """
	SELECT *
	FROM `patents-public-data.patents.publications`
	WHERE RAND() < 0.1
	LIMIT 10000;
	"""


	# Start the query, passing in the extra configuration.
	query_job = client.query(QUERY, job_config=job_config)  # Make an API request.
	query_job.result()  # Wait for the job to complete.

	print("Query results loaded to the table {}".format(table_id))


if __name__ == "__main__": 
	API_KEY_PATH = "/Users/jideofor/Documents/cs397/Patents-Research-abd8b4aaf0a8.json"
	os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = API_KEY_PATH
	PROJECT_ID = "patents-research-275923"

	#  Set source_table_id to the ID of the original table.
	source_table_id = "patents-public-data.patents.publications"

	#Set destination_table_id to the ID of the destination table.
	destination_table_id = "{0}.patents.publications".format(PROJECT_ID)


	# set the schema of the new table
	setSchema(source_table_id,destination_table_id)

	#populate the newtable w/ random sample from old
	copyTable(source_table_id,destination_table_id)


