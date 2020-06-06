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
"""
Usage
>> python -u runBert.py @args.txt

python -u runBert.py @params_model1.txt
------Example args.txt file -----

--tpuAddress node-3
--tpuZone us-central1-f
--outputDir test
--seqLen 15
--modelHub https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1
--batchSize 64
--epochs 40
--dropout .9

"""

####################################################
############ Setting output directory ##############
####################################################
def getDir(bucket, output_dir):
    return 'gs://{}/{}'.format(bucket, output_dir)

def setUp_output_dir():
    DO_DELETE = True
    USE_BUCKET =True
    
    if USE_BUCKET:
        OUTPUT_DIR = getDir(BUCKET, OUTPUT_DIR)
    
    
    if DO_DELETE:
        try:
            tf.gfile.DeleteRecursively(OUTPUT_DIR)
        except:
            # doesn't matter if the directory didn't exist
            pass
    tf.gfile.MakeDirs(OUTPUT_DIR)
    print('***** Model output directory: {} *****'.format(OUTPUT_DIR))
    
#################################################
############# Load Data set #####################
#################################################
def loadPdData(gsPath):
    return pd.read_csv(gsPath, sep = "\t")

def saveToGcloud(path,data,isPandas = False ):
    '''Saves to gcloud so we dont have to do this long ass step every time'''
    if isPandas:
        data.to_csv(path, index=False, sep="\t")
    else:
        with file_io.FileIO(path, mode='w') as f:
            pickle.dump(data,f)


def readFromGcloud(path, isPandas = False):
    if isPandas:
        return pd.read_csv(path,sep="\t" )
    else:
        with file_io.FileIO(path, mode='rb') as f:
            return pickle.load(f)
    
def worker_downloadTestData(name):
    """
    Worker so we can download test data asynch
    """
    logging.info("Thread %s: starting for loading test data", name)
    global test_features
    train_features = readFromGcloud(TEST_TFRecord_PATH)
    logging.info("Thread %s: finishing for loading test data", name) 
        
#######################################################
############# Creating a model  #######################
#######################################################      
def create_model(is_training, input_ids, input_mask, segment_ids, labels,
                 num_labels, bert_hub_module_handle, dropout):
    """Creates a classification model."""
    tags = set()
    if is_training:
        tags.add("train")
    bert_module = hub.Module(bert_hub_module_handle, tags=tags, trainable=True)
    bert_inputs = dict(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids)
    bert_outputs = bert_module(
      inputs=bert_inputs,
      signature="tokens",
      as_dict=True)


    output_layer = bert_outputs["pooled_output"]

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=dropout)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)
    
def model_fn_builder(num_labels, learning_rate, num_train_steps,
                     num_warmup_steps, use_tpu, bert_hub_module_handle, dropout):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
                is_training, input_ids, input_mask, segment_ids, label_ids, num_labels,
                bert_hub_module_handle, dropout)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                    total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                  mode=mode,
                  loss=total_loss,
                  train_op=train_op)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(label_ids, predictions)
                loss = tf.metrics.mean(per_example_loss)
                true_pos = tf.metrics.true_positives(
                            label_ids,
                            predictions)
                true_neg = tf.metrics.true_negatives(
                            label_ids,
                            predictions)   
                false_pos = tf.metrics.false_positives(
                            label_ids,
                            predictions)  
                false_neg = tf.metrics.false_negatives(
                            label_ids,
                            predictions) 
                return {
                    "true_positives": true_pos,
                    "true_negatives": true_neg,
                    "false_positives": false_pos,
                    "false_negatives": false_neg,
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            eval_metrics = (metric_fn, [per_example_loss, label_ids, logits])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
              mode=mode,
              loss=total_loss,
              eval_metrics=eval_metrics)
        
        elif mode == tf.estimator.ModeKeys.PREDICT:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
              mode=mode, predictions={"probabilities": probabilities})
        else:
            raise ValueError(
              "Only TRAIN, EVAL and PREDICT modes are supported: %s" % (mode))

        return output_spec

    return model_fn

####################################################
###### FUnctions to train + evaluate model #########
####################################################
def get_run_config(output_dir):
    """
    Used for run configuration when TPU used
    """
    return tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=output_dir,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=ITERATIONS_PER_LOOP,
            num_shards=NUM_TPU_CORES,
            per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))

def getEstimator(mode_fn):
    """
    Returns the estimator used to train/eval model
    """
    return tf.estimator.tpu.TPUEstimator(
          use_tpu=True,
          model_fn=mode_fn,
          config=get_run_config(OUTPUT_DIR),
          train_batch_size=BATCH_SIZE,
          eval_batch_size=EVAL_BATCH_SIZE,
          predict_batch_size=PREDICT_BATCH_SIZE,
          eval_on_tpu = True
        ) 

def model_train(estimator):
    """
    Trains the model, rt only good for TPU
    """
    #Set drop_remainder =True to fix a TPU error
    #https://stackoverflow.com/questions/58029896/bert-fine-tuning-with-estimators-on-tpus-on-colab-typeerror-unsupported-operand

    print('***** Started training at {} *****'.format(datetime.now()))
    print('  Num examples = {}'.format(len(train_features)))
    print('  Batch size = {}'.format(BATCH_SIZE))
    tf.logging.info("  Num steps = %d", num_train_steps)
    
    current_time = datetime.now()
    train_input_fn = run_classifier.input_fn_builder(
        features=train_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    print("Finished: Training took time ", datetime.now() - current_time)

    #train_features
def model_evaluate(estimator, data):
    """
    Evaluates the model
    """
    print('***** Started evaluation at {} *****'.format(datetime.now()))
    print('  Num examples = {}'.format(len(data)))
    print('  Batch size = {}'.format(EVAL_BATCH_SIZE))

    # Eval will be slightly WRONG on the TPU because it will truncate
    # the last batch.
    eval_steps = int(len(data) / EVAL_BATCH_SIZE)
    
    eval_input_fn = run_classifier.input_fn_builder(
        features=data,
        seq_length=MAX_SEQ_LENGTH,
        is_training=False,
        drop_remainder=True)
    
    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
    print('***** Finished evaluation at {} *****'.format(datetime.now()))
    output_eval_file = os.path.join(OUTPUT_DIR, "eval","eval_results.txt")
    tf.gfile.MakeDirs(os.path.join(OUTPUT_DIR, "eval"))
    with tf.gfile.GFile(output_eval_file, "w") as writer:
        print("***** Eval results *****")
        for key in sorted(result.keys()):
            print('  {} = {}'.format(key, str(result[key])))
            writer.write("%s = %s\n" % (key, str(result[key])))
      
def saveModelParams(params, _dir):
    """
    Save model params to gCloud
    """
    model_params_file = os.path.join(_dir,"modelParams","model_parameters.txt")
    tf.gfile.MakeDirs(os.path.join(_dir, "modelParams"))
    
    with tf.gfile.GFile(model_params_file, "w") as writer:
        print("***** Model Parameters *****")
        for key in sorted(params.keys()):
            print('  {} = {}'.format(key, str(params[key])))
            writer.write("%s = %s\n" % (key, str(params[key])))
    print("Model paramters at: {}".format(os.path.join(_dir, "modelParams")))

class LoadFromFile (argparse.Action):
    def __call__ (self, parser, namespace, values, option_string = None):
        with values as f:
            parser.parse_args(f.read().split(), namespace)    
    
if __name__ == "__main__":
    my_parser = argparse.ArgumentParser(
            fromfile_prefix_chars='@',
            prog="runBert",
            description='Run bert on patent data!!')
    
    #my_parser.add_argument('--file', type=open, action=LoadFromFile)
    
    my_parser.add_argument(
            '--tpuAddress', 
            action='store', 
            type=str, 
            required=True, 
            help="The address of TPU node"
            )
    
    my_parser.add_argument(
            '--tpuZone', 
            action='store', 
            type=str, 
            required=False, 
            nargs='?',
            const="us-central1-f",
            help="The zone that the TPU is in: default us-central1-f"
            )
    
    my_parser.add_argument(
            '--outputDir', 
            action='store', 
            type=str, 
            required=True,
            help="The output dir of results: will be stored in gs bucket `patents-research` under folder bertResults{outputDir}"
            )

    my_parser.add_argument(
            '--seqLen', 
            action='store', 
            type=int, 
            required=True,
            help="The sequence length for the language model"
    )

    my_parser.add_argument(
            '--modelHub', 
            action='store', 
            type=str, 
            required=False,
            nargs='?',
            const="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
            help="The Bert model Hub"
    )   
        
    my_parser.add_argument(
            '--batchSize', 
            action='store', 
            type=int, 
            required=False,
            const=64,
            nargs='?',
            help="The training batch size"
    )  
    
    my_parser.add_argument(
            '--epochs', 
            action='store', 
            type=float, 
            required=False,
            const=40.0,
            nargs='?',
            help="The number of epochs"
    )  
    
    my_parser.add_argument(
            '--dropout', 
            action='store', 
            type=float, 
            required=False,
            const=0.7,
            nargs='?',
            help="Percent of data to keep"
    )  
        
        
          
    args = my_parser.parse_args()
    
    
    ##### SET TPU CONSTANTS AND CONNECT TO IT #######
    TPU_ADDRESS = args.tpuAddress
    TPU_ZONE = args.tpuZone
    USE_TPU =True
    ITERATIONS_PER_LOOP = 1000
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=TPU_ADDRESS, zone=TPU_ZONE)
    tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)
    tf.distribute.experimental.TPUStrategy(tpu_cluster_resolver)
    NUM_TPU_CORES = len(tf.config.experimental.list_logical_devices('TPU'))
    
    if NUM_TPU_CORES==0:
        sys.exit("Problem with tpu make sure region is correct or tpu is runnign")


    ###################################
    ####### CONSTANTS ##################
    ####################################
    
    DATA_PATH = "gs://patents-research/patent_research/data_frwdcorrect.tsv"
    OUTPUT_DIR = "bertResults_{}".format(args.outputDir)# where the model will be saved
    BUCKET = "patents-research"
    
    DATA_COLUMN = 'text'
    LABEL_COLUMN = 'label'
    label_list = [0, 1, 2] 

    MAX_SEQ_LENGTH = args.seqLen
    TRAIN_TFRecord_PATH= "gs://patents-research/patent_research/{}_{}.pickle".format("train_features",MAX_SEQ_LENGTH)
    TEST_TFRecord_PATH= "gs://patents-research/patent_research/{}_{}.pickle".format("test_features",MAX_SEQ_LENGTH)
    BERT_MODEL_HUB = args.modelHub

    #Set output directory
    setUp_output_dir()
    # Force TF Hub writes to the GS bucket we provide.
    os.environ['TFHUB_CACHE_DIR'] =  os.path.join(OUTPUT_DIR,"tfhub_cache")
    tf.gfile.MakeDirs(os.path.join(OUTPUT_DIR,"tfhub_cache"))


    # Model Parameters
    # These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
    BATCH_SIZE = args.batchSize
    EVAL_BATCH_SIZE = NUM_TPU_CORES
    PREDICT_BATCH_SIZE = NUM_TPU_CORES
    LEARNING_RATE = 2e-5
    NUM_TRAIN_EPOCHS = args.epochs
    DROPOUT_KEEP_PROB = args.dropout
    # Warmup is a period of time where hte learning rate 
    # is small and gradually increases--usually helps training.
    WARMUP_PROPORTION = 0.1
    # Model configs
    SAVE_CHECKPOINTS_STEPS = 1000
    SAVE_SUMMARY_STEPS = 100

    # Compute # train and warmup steps from batch size
    num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS) 
    num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
    
    
    params={
            "TPU_ADDRESS":TPU_ADDRESS,
            "TPU_ZONE":TPU_ZONE,
            "TPU_ITERATIONS_PER_LOOP":ITERATIONS_PER_LOOP,
            "NUM_TPU_CORES":NUM_TPU_CORES,
            "TFHUB_CACHE_DIR":os.path.join(OUTPUT_DIR,"tfhub_cache"),
            "DATA_PATH":DATA_PATH,
            "OUTPUT_DIR":OUTPUT_DIR,
            "MAX_SEQ_LENGTH":MAX_SEQ_LENGTH,
            "TRAIN_TFRecord_PATH":TRAIN_TFRecord_PATH,
            "TEST_TFRecord_PATH":TEST_TFRecord_PATH,
            "BERT_MODEL_HUB":BERT_MODEL_HUB,
            "BATCH_SIZE":BATCH_SIZE,
            "EVAL_BATCH_SIZE":EVAL_BATCH_SIZE,
            "PREDICT_BATCH_SIZE":PREDICT_BATCH_SIZE,
            "LEARNING_RATE":LEARNING_RATE,
            "NUM_TRAIN_EPOCHS":NUM_TRAIN_EPOCHS,
            "DROPOUT_KEEP_PROB":DROPOUT_KEEP_PROB,
            "WARMUP_PROPORTION":WARMUP_PROPORTION,
            "SAVE_CHECKPOINTS_STEPS":SAVE_CHECKPOINTS_STEPS,
            "SAVE_SUMMARY_STEPS":SAVE_SUMMARY_STEPS,
            "num_train_steps":num_train_steps,
            "num_warmup_steps":num_warmup_steps
            }
    saveModelParams(params,OUTPUT_DIR)




    #####################################################
    ########### RUNNING SET UP FUNCTIONS ################
    #####################################################
    
    # Download train data
    print("Reading {} from gCloud".format(TRAIN_TFRecord_PATH))
    #train_features = readFromGcloud(TRAIN_TFRecord_PATH)
    print("Finished {} from gCloud!".format(TRAIN_TFRecord_PATH))
    
    # Download test data async - test data will be saved at test_features
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,datefmt="%H:%M:%S")
    getTestData_thread = threading.Thread(target=worker_downloadTestData, args=(1,))
    #getTestData_thread.start() #async download of test data
    
    #####################################################
    ########## Train + Eval Model #######################
    #####################################################
    mode_fn = model_fn_builder(
      num_labels=len(label_list),
      learning_rate=LEARNING_RATE,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      dropout = DROPOUT_KEEP_PROB,
      use_tpu = USE_TPU,
      bert_hub_module_handle = BERT_MODEL_HUB
    )
    
    #estimator = getEstimator(mode_fn) 
    #model_train(estimator)
    #gc.collect()
    #del train_features # Remove train_features might cause mem limit
    #gc.collect() # Remove train_features might cause mem limit
    
    #getTestData_thread.join()
    #gc.collect()
    #model_evaluate(estimator, train_features)

