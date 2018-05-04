# NLP2018

* Project Structure  
    NLP2018  
    -- main.py  
    -- data/  
    ---- glove.twitter.27B.50d.txt  
    ---- test_set.json  
    ---- training_set.json  
    -- out/  
    ---- 2018-04-30104022  
    ------ model.h5  
    ------ report.txt  
  
* Usage  
  1. Download GloVe pre-trained word vectors from: http://nlp.stanford.edu/data/glove.twitter.27B.zip and put `glove.twitter.27B.50d.txt` to data/ directory
  2. Download Training/Testing data from TA's slide and put to data/ directory
  3. install tensorflow 1.4 and Keras 2.1.4
  4. $> python main.py this will trigger both training and testing process  
  
