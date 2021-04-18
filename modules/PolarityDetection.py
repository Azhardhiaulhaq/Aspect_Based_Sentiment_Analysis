import pandas as pd
import nltk
import numpy as np
import matplotlib.pyplot as plt
import re
import tensorflow as tf
import pickle
import re
from sklearn.utils import shuffle
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM, Bidirectional, GRU, Conv1D,MaxPooling1D
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers.embeddings import Embedding
from sklearn.metrics import f1_score, precision_score, recall_score,confusion_matrix,classification_report

class PolarityDetection :
    def __init__(self):
        self.maxlen = 100
    
    def save_model(self,model,filename):
        model.save(filename)
    
    def load_dataset(self,path):
        data = pd.read_csv(path,encoding='latin-1',names=['sentence','relations','valid','polarity'])
        data = data.drop(columns=['valid'])
        data = data.iloc[1:]
        data = data.dropna().reset_index(drop=True)
        return data

    def load_model(self,filename):
        self.model = tf.keras.models.load_model(filename)
        self.tokenizer = self.load_tokenizer()
        self.encoder = self.load_encoder()

    def load_tokenizer(self):
        with open('modules/model/tokenizer_polarity.pickle', 'rb') as handle:
            Tokenizer = pickle.load(handle)
        return Tokenizer
    
    def load_encoder(self):
        with open('modules/model/encoder_polarity.pickle', 'rb') as handle:
            Encoder = pickle.load(handle)
        return Encoder

    def preprocessing(self,data):
        sentences = data['sentence'] + ' ' + data['relations']
        polarities = data['polarity']

        return sentences,polarities

    def get_tokenizer(self,X_train):
        tk = Tokenizer(num_words=1000,filters='!"#$%&()*+,-./:;<=>?@[\]^_`{"}~\t\n',lower=True, split=" ")
        tk.fit_on_texts(X_train)
        with open('modules/model/tokenizer_polarity.pickle','wb') as handle : 
            pickle.dump(tk,handle,protocol=pickle.HIGHEST_PROTOCOL)
        return tk

    def feature_extraction(self, X_train, X_test):
        tokenizer = self.get_tokenizer(X_train)
        X_train = tokenizer.texts_to_sequences(X_train)
        X_test = tokenizer.texts_to_sequences(X_test)
        X_train = pad_sequences(X_train,maxlen=self.maxlen)
        X_test = pad_sequences(X_test,maxlen=self.maxlen)
        return X_train, X_test

    def encode_target(self, y_train, y_test):
        self.encoder = LabelEncoder()
        y_train = self.encoder.fit_transform(y_train)
        y_test = self.encoder.transform(y_test)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        with open('modules/model/encoder_polarity.pickle','wb') as handle : 
            pickle.dump(self.encoder,handle,protocol=pickle.HIGHEST_PROTOCOL)

        return y_train, y_test

    def get_model(self):
        # model = Sequential([
        #     Embedding(1000,8,input_length=100),
        #     Conv1D(filters=32, kernel_size=3,
        #               strides=1, padding="causal",
        #               activation="relu",
        #               input_shape=[None, 3]),
        #     Dropout(0.1),
        #     Bidirectional(LSTM(200, return_sequences=True)),
        #     Dropout(0.1),
        #     Bidirectional(LSTM(100,)),
        #     Dense(512, activation="relu"),
        #     Dense(3, activation="sigmoid")])

        model = Sequential([
            Embedding(1000,8,input_length=100),
            Bidirectional(LSTM(200, return_sequences=True)),
            Dropout(0.1),
            Bidirectional(GRU(200,)),
            Dense(256, activation="relu"),
            Dense(3, activation="sigmoid")])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def get_prediction(self,list_pred):
        data = np.zeros(shape=(list_pred.shape),dtype=int)
        data[np.where(list_pred == np.max(list_pred))] = 1
        return data.tolist()

    def get_prediction_inference(self,list_pred):
        result = []
        for pred in list_pred:
            pred = pred.tolist()
            result.append(pred.index(max(pred)))
        return result

    def print_evaluation(self, task_name, y_true, y_pred):
        print(task_name)
        print("Precision : ", precision_score(y_true, y_pred, average='macro'))
        print("Recall : ", recall_score(y_true, y_pred, average='macro'))
        print("F1-score : ", f1_score(y_true, y_pred, average='macro'))
        print("Classification Report : \n", classification_report(y_true, y_pred))

    def evaluate(self,model,X,y):
        y_pred = model.predict(X)
        result = []
        for pred in y_pred:
            result.append(self.get_prediction(pred))
        self.print_evaluation("Polarity Detection Evaluation", y, result)

    def train(self):
        data = self.load_dataset('datasets/Relation_Dataset.csv')
        X,y = self.preprocessing(data)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.1,random_state=37)
        X_train, X_test = self.feature_extraction(X_train,X_test)
        y_train, y_test = self.encode_target(y_train,y_test)
        X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size=0.1,random_state=37)
        model = self.get_model()
        print(model.summary())
        model.fit(X_train, y_train, batch_size=32, epochs=40)
        self.save_model(model,"modules/model/PolarityDetection")
        self.evaluate(model,X_test,y_test)

    def predict(self, sentences):
        INDEX2LABEL = {0:'Negative', 1:'Neutral', 2:'Positive'}
        sentences = self.tokenizer.texts_to_sequences(sentences)
        sentences = pad_sequences(sentences,padding='post',maxlen=self.maxlen)
        y_pred = self.model.predict(sentences)
        y_pred = self.get_prediction_inference(y_pred)
        sentimen = []
        for y in y_pred :
            sentimen.append(INDEX2LABEL[y])
        return sentimen

# polarity_detector = PolarityDetection()
# polarity_detector.load_model("modules/model/PolarityDetection")
# polarity_detector.predict(['kamar nyaman tapi makanan tidak enak. kamar nyaman','kamar nyaman tapi makanan tidak enak. makanan tidak enak'])
# polarity_detector.train()
