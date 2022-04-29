import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer 
from nltk import word_tokenize
from keras.utils.np_utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
le = LabelEncoder()
one_hot = OneHotEncoder()
lemm = WordNetLemmatizer()
stemm = PorterStemmer()

def cleaning(data):
  train_x = data['examples'] # наши примери
  train_y = data['intents'] #наши намерении

  for ind in range(len(train_x)):
      train_x[ind] = re.sub('[^\w\s]+', '', train_x[ind]) # выражение позволяет удалить все знаки препинания
      train_x[ind] = [word.lower() for word in train_x[ind]] # Нормализующий регистр (нижний регистр)
      train_x[ind] = [lemm.lemmatize(word) for word in train_x[ind]] #нахождение корень слова
      train_x[ind] = ''.join(train_x[ind]) 
      train_x[ind] = word_tokenize(train_x[ind]) #токенизация 
  return train_x, train_y


def encoder(x, y):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)
    train = tokenizer.texts_to_sequences(x)
    x_train = pad_sequences(train, maxlen=len(max(x)))
    
    le = LabelEncoder()
    train_y = le.fit_transform(y)
    
    return x_train, train_y

