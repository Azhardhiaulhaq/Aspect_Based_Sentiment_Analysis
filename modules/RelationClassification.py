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

class RelationClassification :
    def __init__(self):
        self.maxlen = 100

    def save_model(self,model,filename):
        model.save(filename)

    def load_dataset(self,path):
        data = pd.read_csv(path,encoding='latin-1',names=['sentence','relations','valid','polarity'])
        data = data.drop(columns=['polarity'])
        data = data.iloc[1:]
        data = data.dropna().reset_index(drop=True)
        return data

    def load_model(self, filename, tokenizer_path, encoder_path):
        self.model = tf.keras.models.load_model(filename)
        self.tokenizer = self.load_tokenizer(tokenizer_path)
        self.encoder = self.load_encoder(encoder_path)

    def load_tokenizer(self, tokenizer_path):
        with open(tokenizer_path, 'rb') as handle:
            Tokenizer = pickle.load(handle)
        return Tokenizer

    def load_encoder(self, encoder_path):
        with open(encoder_path, 'rb') as handle:
            Encoder = pickle.load(handle)
        return Encoder

    def preprocessing(self,data):
        sentences_and_relations = data['sentence'] + ' ' + data['relations']
        valid = data['valid']

        return sentences_and_relations, valid

    def get_tokenizer(self,X_train):
        tk = Tokenizer(num_words=1000,filters='!"#$%&()*+,-./:;<=>?@[\]^_`{"}~\t\n',lower=True, split=" ")
        tk.fit_on_texts(X_train)
        with open('modules/model/tokenizer_relation.pickle','wb') as handle : 
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
        with open('modules/model/encoder_relation.pickle','wb') as handle : 
            pickle.dump(self.encoder,handle,protocol=pickle.HIGHEST_PROTOCOL)

        return y_train, y_test

    def get_model(self):
        # model = Sequential([
        #     Embedding(1000, 8, input_length=self.maxlen),
        #     LSTM(300,),
        #     Dense(2, activation="sigmoid")
        # ])

        # model = Sequential([
        #     Embedding(1000, 8, input_length=self.maxlen),
        #     GRU(300,),
        #     Dense(2, activation="sigmoid")
        # ])

        # model = Sequential([
        #     Embedding(1000, 8, input_length=self.maxlen),
        #     Bidirectional(LSTM(300,)),
        #     Dense(2, activation="sigmoid")
        # ])

        model = Sequential([
            Embedding(1000, 8, input_length=self.maxlen),
            Bidirectional(GRU(300,)),
            Dense(2, activation="sigmoid")
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def get_prediction(self, list_pred) :
        data = np.zeros(shape=(list_pred.shape), dtype=int)
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
        self.print_evaluation("Relation Classification Evaluation", y, result)

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
        self.save_model(model,"modules/model/RelationClassification")
        self.evaluate(model,X_test,y_test)

    def predict(self, sentences):
        print(sentences)
        sentences_processed = self.tokenizer.texts_to_sequences(sentences)
        sentences_processed = pad_sequences(sentences_processed,padding='post',maxlen=self.maxlen)
        y_pred = self.model.predict(sentences_processed)
        print(y_pred)
        y_pred = self.get_prediction_inference(y_pred)

        valid_relations = []
        i = 0
        for y in y_pred :
            if y == 1 :
                valid_relations.append(sentences[i])
            i += 1
        print(valid_relations)

# relation_classification = RelationClassification()
# relation_classification.load_model("modules/model/RelationClassification")
# relation_classification.predict(['kamar nyaman tapi makanan tidak enak. kamar nyaman',
#                                  'kamar nyaman tapi makanan tidak enak. kamar tidak enak',
#                                  'kamar nyaman tapi makanan tidak enak. makanan nyaman',
#                                  'kamar nyaman tapi makanan tidak enak. makanan tidak enak',
#                                  'makanan tidak enak, kamar sempit dan kotor. makanan tidak enak',
#                                  'makanan tidak enak, kamar sempit dan kotor. makanan sempit',
#                                  'makanan tidak enak, kamar sempit dan kotor. makanan kotor',
#                                  'makanan tidak enak, kamar sempit dan kotor. kamar tidak enak',
#                                  'makanan tidak enak, kamar sempit dan kotor. kamar sempit',
#                                  'makanan tidak enak, kamar sempit dan kotor. kamar kotor'])
# relation_classification.train()
