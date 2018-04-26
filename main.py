from __future__ import print_function
import sys
import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation


TRAINING_DATA_PATH = 'data/training_set.json'
GLOVE_EMBEDDER_PATH = 'data/glove.twitter.27B.50d.txt'
REDUNDANT_WORD_THRES = 50

def progress(count, total, suffix=''):
  bar_len = 60
  filled_len = int(round(bar_len * count / float(total)))
  percents = round(100.0 * count / float(total), 1)
  bar = '#' * filled_len + '-' * (bar_len - filled_len)
  sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
  sys.stdout.flush()

def load_data(embed_dict):
  print('loading data...')
  data = json.load(open(TRAINING_DATA_PATH))
  num_data = len(data)
  word_counts = {}
  processed_word_lists = []
  print('preprocessing tweets...')
  for idx, datum in enumerate(data):
    # print(datum['tweet'])
    words = [w for w in datum['tweet'].replace(',','').replace('.','').replace('?','').replace('!','').replace(':','').lower().split(' ') if w.isalpha() or '\'' in w]
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
  for idx, word_list in enumerate(processed_word_lists):
    embed_words = [embed_dict[w] for w in word_list if w in embed_dict and word_counts[w] < REDUNDANT_WORD_THRES]
    sentiments = [float(data[idx]['sentiment'])] * len(embed_words)
    embedded_word_list.extend(embed_words)
    sentiment_list.extend(sentiments)
    progress(idx + 1, num_data)
  print('\ndata loaded')
  return np.array(embedded_word_list), np.array(sentiment_list)

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


if __name__ == '__main__':
  glove_dict = load_glove()
  embedded_word_list, sentiment_list = load_data(glove_dict)

  model = Sequential()
  model.add(Dense(32, activation='relu', input_dim=50))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(optimizer='rmsprop', loss='mse')
  model.fit(embedded_word_list, (sentiment_list+1)/2, nb_epoch=100, batch_size=32)