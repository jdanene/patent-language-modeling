# patent-language-modeling

## Setting up</br>
**Update Ananconda**:`conda update conda` </br>
*(Update any packages if necessary by typing y to proceed)*</br>

**Create venv**:`conda create -n patent-modeling python=3.7.3` 

**Activate venv**:`conda activate patent-modeling` </br>
*(To deactivate venv `conda deactivate`)* </br> </br> 
**Install the required packages**:`pip install -r requirements.txt`</br> </br>
**Add the virtual env kernell to Jupyter notebook**:`ipython kernel install --user --name=patent-modeling`</br></br>
`jupyter nbextension enable --py widgetsnbextension
`
## Using Jupyter Notebook

**Opening up notebook**:`jupyter notebook`</br>
*(opening notebook on cloud-> `jupyter notebook --port=8081 --no-browser`)* </br>
*When open select the venv  "patent-modeling" as the kernel in Jupyter. Example below creates a new notebook using "patent-modeling" kernel.*
![JupyterKenerlExample](https://i.imgur.com/pBVcUme.png)


## Running

`App.py` is like main.c or App.js. Entry point of applications runs all the code

## Using gCloud
Follow this tutorial: https://medium.com/@senthilnathangautham/colab-gcp-compute-how-to-link-them-together-98747e8d940e</br>
From the tutorial you can figure out that you actually don't need collab can strictly use juptyer on you VM
- https://jupyter.readthedocs.io/en/latest/running.html
- https://cloud.google.com/tpu/docs/creating-deleting-tpus#console

Still need to figure out how to use the provisioned TPUs
  
Batch size and epoch easy explanation ->https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/#:~:text=The%20batch%20size%20is%20a%20number%20of%20samples%20processed%20before,samples%20in%20the%20training%20dataset.

Going to need terminal multiplexing use `tmux` -->https://lukaszwrobel.pl/blog/tmux-tutorial-split-terminal-windows-easily/

tpu service name - service-474881951903@cloud-tpu.iam.gserviceaccount.com
## Raw Data
Raw Data located at:
- 'gs://patents-research/patent_research/data_frwdcorrect.tsv' [contains column of date, the text blob contains list of id backword cited, count of backword cited]

All textblob = p(date, backcited, title, abstract, claims), depending on the data set backcited is either
- number of patents a specific patent cites
- number of patents a specific patent cites + the individual id's of these patents
    - in this case textblob= p(id,date, backcited, title, abstract, claims), we add the id to hopefully get some network features of citation network
**Example of how to load date:**</br>
`df.read_csv('gs://patents-research/patent_research/data_frwdcorrect.tsv', sep='\t')` </br></br>
**If using colab must authenticate with:**
`from google.colab import auth`</br>
`auth.authenticate_user()`</br>
`print('Authenticated')`</br>

**If running from VM on google no auth needed if have permission**</br></br>
**If running from local machine must set env variable `GOOGLE_APPLICATION_CREDENTIALS` to the location of json with api/credentials for project `PROJECT_ID = "patents-research-275923"`**
