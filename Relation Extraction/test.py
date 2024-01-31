#! /usr/bin/env python3

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_load
from data_load import print_results
from tensorflow.contrib import learn
from nltk import word_tokenize
#word_tokenize = list # char-level
#tokenizer_fn = lambda iterator: [(yield word_tokenize(value)) for value in iterator]
def tokenizer_fn(iterator):
    return [(yield word_tokenize(value)) for value in iterator]
#tokenizer_pos_fn = lambda iterator: [(yield value.split(" ")) for value in iterator]
def tokenizer_pos_fn(iterator):
    return [(yield value.split(" ")) for value in iterator]
import csv

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("test_path", "", "Data source for the test data.")
tf.flags.DEFINE_string("test_entity_path", "", "Data source for the test entities.")
tf.flags.DEFINE_string("test_relation_path", "", "Data source for the test relations.")

# Testing parameters
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size (default: 50)")
tf.flags.DEFINE_string("checkpoint_file", "", "Checkpoint file from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
print("\nParameters:"+"\n".join([(attr.upper()+"="+str(FLAGS[attr].value)) for attr in sorted(FLAGS)])+"\n")

# Data Preparation
# ==================================================

# Load data
print("Loading data...")
x_text_test, y_test_ids = data_load.load_data(FLAGS.test_path.split(","), FLAGS.test_entity_path.split(","), FLAGS.test_relation_path.split(","))
distances_test, pos_tags_test, sdp_test = data_load.load_features(x_text_test)

# Save data
print("...Saving data")
with open("data_labels_test.txt", 'w') as f:
    f.write("\n".join([x_text_test[i] + '\t' + y_test_ids[i][0] for i in range(len(x_text_test))]))

# One-hot labels
y_test = y_test_ids[:,0]
labelsnames = np.load(os.path.join(os.path.dirname(FLAGS.checkpoint_file), "..", "labels.npy")).tolist()
y_test = np.eye(len(labelsnames), dtype=int)[np.searchsorted(labelsnames, y_test)]

# Map data into word vocabulary
vocab_path = os.path.join(os.path.dirname(FLAGS.checkpoint_file), "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
max_document_length = vocab_processor.max_document_length
x_test = np.array(list(vocab_processor.transform(x_text_test)))

# Map data into POS vocabulary
vocab_pos_path = os.path.join(os.path.dirname(FLAGS.checkpoint_file), "..", "vocab_pos")
vocab_processor_pos = learn.preprocessing.VocabularyProcessor.restore(vocab_pos_path)
pos_test = np.array(list(vocab_processor_pos.transform(pos_tags_test)))

# Distance1 and Distance2
d1_test = np.arange(max_document_length-1, 2*max_document_length-1)[None,:] - distances_test[:,0][:,None]
d2_test = np.arange(max_document_length-1, 2*max_document_length-1)[None,:] - distances_test[:,1][:,None]

# Map data into Type vocabulary
labelstypes = np.load(os.path.join(os.path.dirname(FLAGS.checkpoint_file), "..", "labelstypes.npy")).tolist()
type_test = np.zeros(x_test.shape, dtype=np.int64)
type_test[[range(len(type_test)), np.array(distances_test)[:,0]]] = np.searchsorted(labelstypes, y_test_ids[:,3])+1
type_test[[range(len(type_test)), np.array(distances_test)[:,1]]] = np.searchsorted(labelstypes, y_test_ids[:,4])+1

# Testing
# ==================================================

if os.path.isdir(FLAGS.checkpoint_file):
    checkpoint_file = tf.train.latest_checkpoint(os.path.dirname(FLAGS.checkpoint_file))
else:
    checkpoint_file = FLAGS.checkpoint_file

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, log_device_placement=FLAGS.log_device_placement)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        input_y = graph.get_operation_by_name("input_y").outputs[0]
        input_POS = graph.get_operation_by_name("input_POS").outputs[0]
        input_distance1 = graph.get_operation_by_name("input_distance1").outputs[0]
        input_distance2 = graph.get_operation_by_name("input_distance2").outputs[0]
        input_type = graph.get_operation_by_name("input_type").outputs[0]
        dropout = graph.get_operation_by_name("dropout").outputs[0]
        # Tensor to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Testing loop
        all_predictions = []
        data = np.array(list(zip(x_test, y_test, pos_test, d1_test, d2_test, type_test)))
        for batch_num in range(int((len(y_test) - 1) / FLAGS.batch_size) + 1):
            start_index, end_index = batch_num * FLAGS.batch_size, min((batch_num + 1) * FLAGS.batch_size, len(y_test))
            x_batch, y_batch, pos_batch, d1_batch, d2_batch, type_batch = zip(*data[start_index:end_index])
            batch_predictions = predictions.eval(feed_dict = {input_x: x_batch, input_POS: pos_batch, input_distance1: d1_batch, input_distance2: d2_batch, input_type: type_batch, dropout: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

if y_test is not None:
    # Print statistics
    y_test = np.argmax(y_test, axis=1)
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
    print("F-measure: {:g}%".format(print_results(y_test, all_predictions, labelsnames, verbose=True)[1][3]))
    # Save the evaluation to a csv
    predictions_readable = np.column_stack((np.array(x_text_test), np.array(labelsnames)[y_test], np.array(labelsnames)[all_predictions.astype(int)]))
    predictions_readable = np.vstack((['Sentence', 'Label', 'Prediction'], predictions_readable))
else:
    # Save the prediction to a csv
    predictions_readable = np.column_stack((np.array(x_text_test), np.array(labelsnames)[all_predictions.astype(int)]))
    predictions_readable = np.vstack((['Sentence', 'Prediction'], predictions_readable))
path = os.path.join(os.path.dirname(FLAGS.checkpoint_file), "..", "prediction" + checkpoint_file.split('model')[-1].split('.')[0] + ".csv")
print("Saved prediction labels to {}".format(path))
with open(path, 'w') as f:
    csv.writer(f).writerows(predictions_readable)

# Text output with labels and ids
path = os.path.join(os.path.dirname(FLAGS.checkpoint_file), "..", "prediction" + checkpoint_file.split('model')[-1].split('.')[0] + ".txt")
with open(path, 'w') as f:
    f.write("\n".join([np.array(labelsnames)[all_predictions[t].astype(int)] + '\t' + y_test_ids[t][1] + '\t' + y_test_ids[t][2] for t in range(len(y_test_ids)) if not np.array(labelsnames)[all_predictions[t].astype(int)] == 'None']))