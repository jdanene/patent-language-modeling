from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
import tensorflow_datasets as tfds
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from bert import run_classifier_with_tfhub
#https://github.com/google-research/bert.git
import sys
from tensorflow import keras
import os
import re
from transformers import *
import numpy as np
from tensorflow.python.lib.io import file_io
import pickle
import gc
import threading
import logging
import argparse

def loadPdData(gsPath):
	return pd.read_csv(gsPath, sep = "\t")

def generateLable(dataset):
	# convert to datetime
	dataset['publication_date'] = pd.to_datetime(dataset['publication_date'], errors="coerce",format="%Y-%m-%d")
	dataset = dataset.sort_values('publication_date', ascending = False)

	#drop if date is NaN - only one 1082-03-15
	dataset = dataset[dataset.publication_date.isnull() == False]

	# calculate the top 1% by publication date - give it label 1
	top1_perc =  dataset.groupby(dataset.publication_date.dt.year)["fwrdCitations_5"].transform(lambda x: x.quantile(.99))
	dataset["label"] = dataset["fwrdCitations_5"] >= top1_perc

	# calculate top 5% by publication date - give it label 2
	top5_perc = dataset.groupby(dataset.publication_date.dt.year)["fwrdCitations_5"].transform(lambda x: x.quantile(.95))
	dataset["label"] = np.where(np.logical_and(dataset["fwrdCitations_5"] >= top5_perc, dataset["label"]==0), 2, dataset["label"])

	return dataset


def loadData(MAX_SEQ_LENGTH,TRAIN_TFRecord_PATH,TEST_DF_PATH,DATA_PATH):
	DATA_COLUMN = 'text'
	LABEL_COLUMN = 'label'
	# label_list is the list of labels, i.e. True, False or 0, 1 or 'dog', 'cat'
	label_list = [0, 1, 2] 

	print("Max seq length {}".format(MAX_SEQ_LENGTH))
	print("Train dataset will be saved at {}".format(TRAIN_TFRecord_PATH))
	print("Test dataset will be saved at {}".format(TEST_DF_PATH))
	print(f'Loading data!')
	dataset = loadPdData(DATA_PATH)
	print(f'Finised loading data!')
	dataset = generateLable(dataset)
	print(f'Test/Train Split!')
	train,test=train_test_split(dataset, test_size=0.2)
	del dataset
	gc.collect()
	print(f'Finished Test/Train Split!')

	print('Saving Test/Train Split to gCloud')
	#saveToGloud(TRAIN_DF_PATH,train,isPandas=True)
	#saveToGloud(TEST_DF_PATH,test,isPandas=True)
	print('Finished Saving Test/Train Split to gCloud!')

	print("Use the InputExample class from BERT's run_classifier code to create examples from the data")
	train_InputExamples = train.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
	                                                                   text_a = x[DATA_COLUMN], 
	                                                                   text_b = None, 
	                                                                   label = x[LABEL_COLUMN]), axis = 1)
	del train
	gc.collect()
	test_InputExamples = test.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
	                                                                   text_a = x[DATA_COLUMN], 
	                                                                   text_b = None, 
	                                                                   label = x[LABEL_COLUMN]), axis = 1)
	del test
	gc.collect()

	print("Finished using  InputExample class from BERT's run_classifier code to create examples from the data")

	print("Convert our train and test features to InputFeatures that BERT understands")
	tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased') # get scientific tokenizer + pointer to the model

	train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
	saveToGcloud(TRAIN_TFRecord_PATH,train_features)
	
	del train_InputExamples
	gc.collect()

	del train_features
	gc.collect()

	test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
	saveToGcloud(TEST_TFRecord_PATH,test_features) 
	
	del test_InputExamples
	gc.collect()

	del test_features
	gc.collect()

	print("Finished converting  train and test features to InputFeatures that BERT understands")
	print("Train dataset saved at {}".format(TRAIN_TFRecord_PATH))
	print("Test dataset saved at {}".format(TEST_DF_PATH))
	return



if __name__ == "__main__":
	my_parser = argparse.ArgumentParser(
		prog="convertToFeatures",
		description='Generate a bert tokenized dataset'
		)

	# Add the arguments
	my_parser.add_argument('SeqLength',
	                        metavar='seqLength',
	                       type=int,
	                       help='max seq length')

	# Execute the parse_args() method
	args = my_parser.parse_args()
	MAX_SEQ_LENGTH = args.SeqLength


	if MAX_SEQ_LENGTH<= 0:
		sys.exit("0< SeqLength < 512")
	if MAX_SEQ_LENGTH > 512:
		sys.exit("0< SeqLength < 512")


	DATA_PATH = "gs://patents-research/patent_research/data_frwdcorrect.tsv"
	TRAIN_DF_PATH= "gs://patents-research/patent_research/{}_{}.tsv".format("bert_train_df",MAX_SEQ_LENGTH) 
	TEST_DF_PATH="gs://patents-research/patent_research/{}_{}.tsv".format("bert_test",MAX_SEQ_LENGTH)

	loadData(MAX_SEQ_LENGTH,TRAIN_DF_PATH, TEST_DF_PATH,DATA_PATH)
	sys.exit()

