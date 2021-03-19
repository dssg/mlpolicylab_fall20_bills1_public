import types
import scipy
import keras
import pickle
import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from baselines import *
from metrics import Metric


class Baseline():
    def __init__(self, conn, iter_num=4, iter_val=1):
        self.data = get_sponsor_and_party_data(conn, iter_num, iter_val)
        self.sponsor_pass_rates = 0.0
        self.party_pass_rates = 0.0
    
    def train(self, input_data, params, validation_data=None):
        s, p = get_base_pass_rates(input_data[0], self.data)
        self.sponsor_pass_rates = s
        self.party_pass_rates = p
        
    def predict(self, input_data, params):
        return baseline1_predict(self.data, input_data, self.sponsor_pass_rates, self.party_pass_rates)
    
    def save(self, path):
        model_data = (self.data, self.sponsor_pass_rates, self.party_pass_rates)
        pickle.dump(model_data, open(path+"/baseline.pkl", "wb"))
    
    def load(self, path):
        model_data = pickle.load(open(path+"/baseline.pkl", "rb"))
        self.data = model_data[0]
        self.sponsor_pass_rates = model_data[1]
        self.party_pass_rates = model_data[2]

class LogRegClassifier:
    def __init__(self, **kwargs):
        self.model = LogisticRegression(**kwargs)

    def train(self, data, params, validation_data=None):
        if scipy.sparse.issparse(data[0]):
            data[0].sort_indices()
        if scipy.sparse.issparse(data[1]):
            data[1] = data[1].todense()
        labels = np.array(data[1])
        if len(labels.shape) == 2:
            labels = np.argmax(labels, axis=-1)
        
        self.model.fit(data[0], labels)
        
    def predict_score(self, data, params):
        return self.model.predict_proba(data)

    def predict(self, data, params):
        return self.model.predict(data)

    def save(self, path):
        pickle.dump(self.model, open(path+"/lr.pkl", "wb"))

    def load(self, path):
        self.model = pickle.load(open(path+"/lr.pkl", "rb"))
        

class SVMClassifier:
    def __init__(self, **kwargs):
        self.model = SVC(**kwargs)

    def train(self, data, params, validation_data=None):
        if scipy.sparse.issparse(data[0]):
            data[0].sort_indices()
        if scipy.sparse.issparse(data[1]):
            data[1] = data[1].todense()
        labels = np.array(data[1])
        if len(labels.shape) == 2:
            labels = np.argmax(labels, axis=-1)
        
        self.model.fit(data[0], labels)
        
    def predict_score(self, data, params):
        positive_scores = self.model.decision_function(data)
        prediction_scores = np.vstack([-positive_scores, positive_scores]).T
        return prediction_scores
        
    def predict(self, data, params):
        return self.model.predict(data)

    def save(self, path):
        pickle.dump(self.model, open(path+"/svm.pkl", "wb"))

    def load(self, path):
        self.model = pickle.load(open(path+"/svm.pkl", "rb"))
    

class DTClassifier:
    def __init__(self, **kwargs):
        self.model = DecisionTreeClassifier(**kwargs)

    def train(self, data, params, validation_data=None):
        if scipy.sparse.issparse(data[0]):
            data[0].sort_indices()
        if scipy.sparse.issparse(data[1]):
            data[1] = data[1].todense()
        labels = np.array(data[1])
        if len(labels.shape) == 2:
            labels = np.argmax(labels, axis=-1)
        
        self.model.fit(data[0], labels)
        
    def predict_score(self, data, params):
        return self.model.predict_proba(data)

    def predict(self, data, params):
        return self.model.predict(data)

    def save(self, path):
        pickle.dump(self.model, open(path+"/rf.pkl", "wb"))

    def load(self, path):
        self.model = pickle.load(open(path+"/rf.pkl", "rb"))
        
        
class RFClassifier:
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)

    def train(self, data, params, validation_data=None):
        if scipy.sparse.issparse(data[0]):
            data[0].sort_indices()
        if scipy.sparse.issparse(data[1]):
            data[1] = data[1].todense()
        labels = np.array(data[1])
        if len(labels.shape) == 2:
            labels = np.argmax(labels, axis=-1)
        
        self.model.fit(data[0], labels)
        
    def predict_score(self, data, params):
        return self.model.predict_proba(data)

    def predict(self, data, params):
        return self.model.predict(data)

    def save(self, path):
        pickle.dump(self.model, open(path+"/rf.pkl", "wb"))

    def load(self, path):
        self.model = pickle.load(open(path+"/rf.pkl", "rb"))
        
        
class BinaryLabelNNClassifier:

    def __init__(self, input_dim, num_hidden_nodes_in_layers=None, activation='relu', show=True):
        """
        :param:
            input_dim: Dimension of the input layer
            num_hidden_nodes_in_layers (list of ints): Len of the list will be equal to the number of hidden layers
                in the model, which each hidden layer having the corresponding number of nodes (excluding input and
                output layer)
            activation: activation function to be used in the hidden layers
        """
        if num_hidden_nodes_in_layers is None:
            num_hidden_nodes_in_layers = [50, 20]
        inputs = Input(shape=(input_dim, ))
        layers = [inputs]
        for idx, num_nodes in enumerate(num_hidden_nodes_in_layers):
            layers.append(Dense(num_nodes, activation=activation)(layers[idx]))
        output = Dense(1, activation='sigmoid')(layers[-1])
        self.model = Model(inputs=inputs, outputs=output)
        if show:
            print(self.model.summary())

    def train(self, data, params, validation_data=None):
        self.model.compile(optimizer=params['optimizer'], loss=params['loss'])
        if isinstance(data, types.GeneratorType):
            self.model.fit(data, batch_size=params['batch_size'], epochs=params['epochs'])
        else:
            if scipy.sparse.issparse(data[0]):
                data[0].sort_indices()
            if scipy.sparse.issparse(data[1]):
                data[1].sort_indices()
            self.model.fit(data[0], data[1],  batch_size=params['batch_size'], epochs=params['epochs'],
                           validation_data=validation_data, verbose=params['verbose'], validation_batch_size=validation_data[0].shape[0])
            
    def predict_score(self, data, params):
        return self.model.predict(data, batch_size=params['batch_size'], verbose=params['verbose'])
        
    def predict(self, data, params):
        return np.argmax(self.predict_score(data, params), axis=-1)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = keras.models.load_model(path)
        
        
class MultiLabelNNClassifier:

    def __init__(self, input_dim, num_classes=2, num_hidden_nodes_in_layers=None, activation='relu', show=True):
        """
        :param:
            input_dim: Dimension of the input layer
            num_hidden_nodes_in_layers (list of ints): Len of the list will be equal to the number of hidden layers
                in the model, which each hidden layer having the corresponding number of nodes (excluding input and
                output layer)
            activation: activation function to be used in the hidden layers
        """
        if num_hidden_nodes_in_layers is None:
            num_hidden_nodes_in_layers = [50, 20]
        inputs = Input(shape=(input_dim, ))
        layers = [inputs]
        for idx, num_nodes in enumerate(num_hidden_nodes_in_layers):
            layers.append(Dense(num_nodes, activation=activation)(layers[idx]))
        output = Dense(num_classes, activation='softmax')(layers[-1])
        self.model = Model(inputs=inputs, outputs=output)
        if show:
            print(self.model.summary())

    def train(self, data, params, validation_data=None):
        self.model.compile(optimizer=params['optimizer'], loss=params['loss'])
        if isinstance(data, types.GeneratorType):
            self.model.fit(data, batch_size=params['batch_size'], epochs=params['epochs'])
        else:
            if scipy.sparse.issparse(data[0]):
                data[0].sort_indices()
            if scipy.sparse.issparse(data[1]):
                data[1].sort_indices()
            self.model.fit(data[0], data[1],  batch_size=params['batch_size'], epochs=params['epochs'],
                           validation_data=validation_data, verbose=params['verbose'], validation_batch_size=validation_data[0].shape[0])
            
    def predict_score(self, data, params):
        return self.model.predict(data, batch_size=params['batch_size'], verbose=params['verbose'])
        
    def predict(self, data, params):
        return np.argmax(self.predict_score(data, params), axis=-1)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = keras.models.load_model(path)