import datetime
import pickle

import streamlit as st
import seaborn as sns
import pandas as pd
import tensorflow as tf
import altair as alt

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model


# from fxns import display_view



@dataclass
class Model:
     
    def __init__(self):
        
        self.classifier = KNN(n_neighbors=5)
        # self.model_type = model_type
        
    # creating a classification model  
    def build_class_model(self, model_name, df, y='y', drop_col=['d_c']):
        
        X = df.drop(columns=drop_col)
        y = df[y]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
         # convert from dataframes to numpy for training
        X_train = X_train.values

        X_test = X_test.values
        y_train = y_train.values

        y_test = y_test.values

        # Building and training model "classifier"
        self.classifier.fit(X_train, y_train)  # training

        new_pred = self.classifier.predict(X_test)

        # Check the accuracy of our model
        accuracy = self.classifier.score(X_test, y_test)
        # acc = accuracy_score(y_test, new_pred)

        # save model
        # save_model(self.classifier, model_name)
        

        return self.classifier, df, accuracy, new_pred

    def load_model(self, filename='my_model'):
        with open(filename, 'rb') as file:
            cl_classifier = pickle.load(file)
        return cl_classifier
    
    def save_model(self, filename='my_model'):
        with open(filename, 'wb') as file:
            pickle.dump(self.classifier, file)
        print('Classification Model saved as: ' + filename)
        st.info(f'Classifier model successfully built and saved{filename}')


            
            

            
            
@dataclass
class Model2:
    
    def __init__(self):
        self.model = Sequential()

    # creating a regression model
    def build_reg_model(self, model_name, df, y='y', drop_col=['d_c']):
        
        
        
        X = df.drop(columns=drop_col)
        y = df[y]
        
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        key = X_train.keys()
        
        # convert from dataframes to numpy for training
        X_train = X_train.values

        X_test = X_test.values
        y_train = y_train.values

        y_test = y_test.values

        self.model.add(Dense(64, activation='relu', input_dim=len(key)))
        
        # model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        
        # model.add(Dropout(0.5))
        self.model.add(Dense(1))

        optimizer = tf.keras.optimizers.RMSprop(0.001)
        # log_dir = f"logs/{model_name}/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        # tensorboard = TensorBoard(log_dir=f"./logs/MODEL/", histogram_freq=1, write_grads=True)
        # tensorboard = TensorBoard(log_dir=log_dir)

        # optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        # logdir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
        # sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

        self.model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse', 'mae', 'mape'])
        # history = self.model.fit(X_train, y_train, epochs=200, validation_split=0.2, callbacks=[tensorboard])
        
        history = self.model.fit(X_train, y_train, epochs=200, validation_split=0.2)

        # saving history information
        # hist_filename = f"trainHistoryDict_{model_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        hist_path = f"\histories"

        # calculate loss and mean absolute error
        loss, *mae = self.model.evaluate(X_test, y_test, verbose=2)
        print(f"Model loss on test set: {loss}")

        # save_model(self.model, model_name)
        with open(hist_path, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        print(f'history of model {model_name}, saved as hist_{model_name} at {hist_path}')
        
        # model_name_h5 = f'{model_name}.h5'
        # mod = 'Models/'
        # '{}'.format("new\nline")
        # self.model.save(f'Models/{model_name}')
        self.model.save(f'{model_name}.h5')
        print(f'Regression Model saved as: {model_name}')
        return self.model, history, loss, mae

    def load_model(self, model_name='my_reg_model'):
        rg_model = load_model(model_name)
        return rg_model
