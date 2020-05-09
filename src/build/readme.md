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
- There are 14,406,362 rows in the database that match our CPC filter and have grant_date >= 20100101 
- There are 2,549,438 rows in the database that match our CPC filter and have grant_date >= 20050101 AND grant_date < 20100101
- There are 1,983,912 rows in the database that match our CPC filter and have grant_date >= 20000101 AND grant_date < 20050101 
- There are 1,521,278 rows in the database that match our CPC filter and have grant_date>=19950101 AND grant_date < 20000101
- There are 965,082 rows in the database that match our CPC filter and have grant_date>=19900101 AND grant_date < 19950101
- There are 666,256 rows in the database that match our CPC filter and have grant_date>=19850101 AND grant_date < 19900101
- There are 565,570 rows in the database that match our CPC filter and have grant_date>=19800101 AND grant_date < 19850101
- There are 472,812 rows in the database that match our CPC filter and have grant_date>=19750101 AND grant_date < 19800101
- There are 90,246 rows grant_date>=19550101 AND grant_date < 19600101


