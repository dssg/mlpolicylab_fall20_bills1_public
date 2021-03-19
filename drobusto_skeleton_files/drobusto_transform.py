from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import TimeSeriesSplit
from sklearn import preprocessing
import pandas as pd

'''
test dataframe
test = pd.DataFrame({'huh': ['aosw', 'b', 'a', 'c', 'b'], 'model': ['bab', 'ba', 'ba', 'ce', 'bw']})

'''

"""
Skeleton function for column hashing if needed later
def hashColumn(column, n_features):
    '''
    This function takes in a row of categorical data and applies the hash
    trick to it in order to convert it to numerical data

    Arguments: 
    column - an array representing a column to encode values of
    n_features - # of features to 

    returns the arary of arrays of hashed values, one array per row
    '''
    h = FeatureHasher(n_features = n_features, input_type='string')
    f = h.transform(data[column])
    
    return f.toarray()
"""

def oneHot(data):
    '''
    Encodes all categorical variables in the input data via one hot encoding

    data: dataframe of categorical features to be encoded (can just be one column)
    
    returns: an array of arrays with OHE values
    '''

    le = preprocessing.LabelEncoder()
    data_transformed = data.apply(le.fit_transform)
    print(data_transformed)

    encoder = preprocessing.OneHotEncoder()
    encoder.fit(data_transformed)

    oneHotLabels = encoder.transform(data_transformed).toarray()

    return(oneHotLabels)


