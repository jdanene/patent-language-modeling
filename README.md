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
  
