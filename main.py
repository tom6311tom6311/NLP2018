from __future__ import print_function
import sys
import os
import time
import datetime
import json
import math
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


OUTPUT_DIR = 'out/'
TRAINING_DATA_PATH = 'data/training_set.json'
TESTING_DATA_PATH = 'data/test_set.json'
GLOVE_EMBEDDER_PATH = 'data/glove.twitter.27B.50d.txt'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

def progress(count, total, suffix=''):
  bar_len = 60
  filled_len = int(round(bar_len * count / float(total)))
  percents = round(100.0 * count / float(total), 1)
  bar = '#' * filled_len + '-' * (bar_len - filled_len)
  sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
  sys.stdout.flush()

def load_data(path, embed_dict):
  data = json.load(open(path))
  num_data = len(data)
  word_counts = {}
  processed_word_lists = []
  print('preprocessing tweets...')
  for idx, datum in enumerate(data):
    if (type(datum['snippet']) == list ):
      snippet = ' '.join(datum['snippet'])
    else:
      snippet = datum['snippet']
    words = [w for w in snippet.replace(',','').replace('.','').replace('?','').replace('!','').replace(':','').lower().split(' ') if w.isalpha() or '\'' in w]
    processed_word_lists.append(words)
    for w in words:
      if (w in word_counts):
        word_counts[w] += 1
      else:
        word_counts[w] = 1
    progress(idx + 1, num_data)

  print('\nembedding...')
  embedded_word_list = []
  sentiment_list = []
  final_word_list = []
  first_ids = [0]
  for idx, w_l in enumerate(processed_word_lists):
    if (len(sentiment_list) != first_ids[-1]):
      first_ids.append(len(sentiment_list))
    words = [w for w in w_l if w in embed_dict]
    embed_words = [embed_dict[w] for w in words]
    sentiments = [float(data[idx]['sentiment'])] * len(embed_words)
    final_word_list.extend(words)
    embedded_word_list.extend(embed_words)
    sentiment_list.extend(sentiments)
    progress(idx + 1, num_data)
  if first_ids[-1] == len(sentiment_list):
    first_ids = first_ids[:-1]
  print('\ndata loaded')
  return np.array(embedded_word_list), np.array(sentiment_list), final_word_list, first_ids, word_counts

def load_glove():
  print('loading glove dict...')
  lines = []
  with open(GLOVE_EMBEDDER_PATH, 'r') as glove_file:
    lines = glove_file.readlines()
    glove_file.close()
  glove_dict = {}
  num_lines = len(lines)
  for idx, line in enumerate(lines):
    line_arr = line.replace('\n','').split(' ')
    glove_dict[line_arr[0]] = np.array([float(n) for n in line_arr[1:]])
    if (idx % 100 == 0):
      progress(idx + 1, num_lines)
  print('\n')
  return glove_dict

def compute_tfidf(w, word_counts, num_doc):
  # assume a word shows exactly once in each snippet that it shows, i.e. tf = 1
  if w in word_counts:
    tfidf = math.log(num_doc / (1 + word_counts[w]))
  else:
    tfidf = math.log(num_doc / 1)
  return tfidf

if __name__ == '__main__':
  if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
  out_subdir = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d%H%M%S')
  os.makedirs(OUTPUT_DIR + '/' + out_subdir)
  glove_dict = load_glove()
  print('loading training data...')
  embedded_word_list, sentiment_list, raw_word_list, first_ids, word_counts = load_data(TRAINING_DATA_PATH, glove_dict)

  model = Sequential()
  model.add(Dense(1024, activation='relu', input_dim=50))
  # model.add(Dense(128, activation='relu'))
  model.add(Dense(32, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(optimizer='rmsprop', loss='mse')
  model.summary()

  print('training...')
  model.fit(embedded_word_list, (sentiment_list+1)/2, epochs=1000, batch_size=64, callbacks=[EarlyStopping(monitor='loss', patience=5)])
  model.save_weights(OUTPUT_DIR + '/' + out_subdir + '/' + 'model.h5')

  print('loading testing data...')
  embedded_word_list_test, sentiment_list_test, raw_word_list_test, first_ids_test, word_counts_test = load_data(TESTING_DATA_PATH, glove_dict)
  tfidfs_test = [compute_tfidf(w, word_counts, len(first_ids_test)) for w in raw_word_list_test]

  print('predicting...')
  sentiment_list_predicted = model.predict(embedded_word_list_test)

  tse = 0
  for idx, first_id_test in enumerate(first_ids_test):
    word_sentiments = sentiment_list_predicted[first_id_test: (first_ids_test[idx + 1] if idx < len(first_ids_test) - 1 else len(sentiment_list_predicted))]
    weights = np.array(tfidfs_test[first_id_test: (first_ids_test[idx + 1] if idx < len(first_ids_test) - 1 else len(sentiment_list_predicted))]).reshape((-1,1))
    avg_sentiment_predicted = np.average(word_sentiments, weights=weights)
    tse += np.square(((avg_sentiment_predicted*2)-1) - sentiment_list_test[first_id_test])
  mse = tse / len(first_ids_test)
  print('Mean Square Error: ' + str(mse))

  with open(OUTPUT_DIR + '/' + out_subdir + '/' + 'report.txt','w') as fh:
    model.summary(print_fn=lambda x: fh.write(x + '\n'))
    fh.write('Mean Square Error: ' + str(mse))
    fh.close()

  print('#words in training set: ' + str(len(word_counts)))
  print('#words in both training set and embed_dict: ' + str(len([w for w in word_counts if w in glove_dict])))
  print('#words in testing set: ' + str(len(word_counts_test)))
  print('#words in both testing set and embed_dict: ' + str(len([w for w in word_counts_test if w in glove_dict])))
  print('#words in both training set and testing set: ' + str(len([w for w in word_counts if w in word_counts_test])))