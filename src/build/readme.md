build: contains the code to build a useable data files from the raw inputs

The data set can be accessed at: https://console.cloud.google.com/bigquery?p=patents-public-data. 
<br>
A summary of the data set can be found at:
https://www.kaggle.com/bigquery/patents


- There are 123,040,349 rows in the database
- There are 86,867,729 rows in the database that match our CPC filter. 
- So CPC isn't that selective however patents come with an `embedding_v1` array 
    - The patent embeddings were built using a machine learning model that predicted a patent's CPC code from its text. Therefore, the learned embeddings are a vector of 64 continuous numbers intended to encode the information in a patent's text. Distances between the embeddings can then be calculated and used as a measure of similarity between two patents. 
    - https://cloud.google.com/blog/products/data-analytics/expanding-your-patent-set-with-ml-and-bigquery
    - So we can control for poor filtering by including the embedding for CPC code???? 
