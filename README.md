# Identifying and Categorizing Offensive Language in Social Media

Task from website:

https://sites.google.com/site/offensevalsharedtask/offenseval2019

Solutions to task a and task b are in two respective jupyter notebook files:

Task a: [task_a_w2vec.ipynb](task_a_w2vec.ipynb)

Task b: [task_b_w2vec.ipynb](task_b_w2vec.ipynb)

The comments to the solutions are in Polish, but I'm planning to eventually translate them. :)

## Environment

Graphics card and drivers: nvidia + cuda.

I've made a special conda environment mainly using pycharm/conda installation instructions from the Pytorch website.

Libraries installed in the environment:
_Pycharm_ is a popular pythonic Deep Learning library for Python.
_Pandas_ is an open source data analysis and manipulation tool written for Python. Pandas offers data structures and operations for manipulating numerical tables and time series.
_Spacy_ is a Natural Language Processing library for Python.
_Matplotlib_ is a basic visualizaiton library for Python.

Additional environment packages are requirements for the above and jupyter notebook binaries/libraries/pligins.

To speed up matrix multiplication operation used in deep learning I'm using CUDA.

To create this environment one needs to run these commands in terminal after installing conda/anaconda/miniconda etc.


    conda create -n torch
    conda activate torch
    conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
    conda install -c conda-forge pandas matplotlib numpy scikit-learn spacy ipython
    conda install -c conda-forge ipython notebook jupyterlab jupytext

To increase download/install speed one can install and substitute mamba for conda in the above commands, as mamba is written in C and is faster.

## Reading data, creating a model and learning from the beginning to the end

The process applies to both tasks.

First I import libraries and check whether cuda-pytorch works.

I load training and testing data to different Pandas DataFrames.

In the case of training data I combine input data (tweets) with labels (which are different files) to one DataFrame object. 

Then I process every tweet in Spacy, which breaks up the tweet into Spacy objects, which I save in a new column (tokens) of both trainining and Testing DataFrame.

In a new DataFrame column (lemmas) I write lemmas, which I get from .lemma_ Spacy object attribute from the token column. I skip lemmas with @, #, stop words etc.

Then I create a new dictionary dict_lemmas (work: unique number) for every work in both data sources (training and testing).

Every tweet I assign a number corresponding to the lemma from dict_lemmas dictionary. These numbers I save as a list with a lenth of longest tweet, moved to the right, with zeros representing empty spaces. That I save in a new column "numbers" of the respective DataFrame.

Then I generate(or load) vectors of embeddings from Spacy (width, or depth 96) in a roundabout way, by iterating though dict_lemmas and asking Spacy for an embedding for every lemma and saving it embedding list. Then I convert that list to a Pytorch tensor. I do it in a way, so that every row of the embedding tensor was represented the number from dict_lemmas.

Then I define model in Pytorch. To the model I pass:

* amount (width) of input data - corresponding to the longest tweet from both training and testing data.
* amound of neurons in the first and second (hidden) layer, optionally third if it has an assigned number.
* amount of output data - binary classification, so one output neuron.