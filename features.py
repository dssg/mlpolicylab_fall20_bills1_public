import numpy as np
import scipy
import pandas as pd
import datetime as dt
import mmh3
from math import floor
from sklearn import preprocessing
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation
from queries import read_cols_query, put_condition_on_query
from db_ops import run_sql_query

SECONDARY_COL_OPS = ["time_since_session_start", "time_till_session_end", "sponsor_pass_rate", "sponsor_total_bills"]

# object entries to perform text transformation
# need to created on the whole data column once,
# and can subsequently be used on (batches of) rows.
text_transformation_objects = {
    'bow': None,
    'tfidf': None,
    'lda': None,
    'bert': None
}
database_connection_object = None

def obtain_bow_features(texts):
    texts = texts.values.to_list()
    return np.ones((len(texts),5))


def obtain_tfidf_features(texts):
    texts = texts.values.to_list()
    return np.ones((len(texts),5))


def obtain_lda_features(texts, num_topics=10):
    texts = texts.values.to_list()
    lda = LatentDirichletAllocation(n_components=num_topics)
    lda.fit(texts)
    return np.ones((len(texts), 5))


def obtain_bert_features(texts, params):
    # TODO
    return np.ones((len(texts), 5))


def mark_blanks(data, add_col=False):
    """
    This function allows you to add a dummy column to a dataframe which contains boolean conditions
    for if a cell is missing (True) or not (False)

    Arguments:
    data - a python dataframe
    column - the string name of a column to add the dummy column for
    add_col - (bool) control whether you want a dummy column added that corresponds to the original inputted column and
     indicates whether values were missing or not

    returns either the original imputed column if add_col = False or if it's true, the original imputed column
    and the new dummy
    """

    if add_col:
        bool_series = pd.isnull(data.iloc[:, 0])
        dummy_name = data.columns[0] + "_is_null?"
        print(dummy_name)
        # col_index = data.columns.get_loc(column)
        # col_index += 1
        data.insert(1, column=dummy_name, value=bool_series)

        data = data.fillna("Unknown")

        return data.iloc[:, 0], data.iloc[:, 1]

    else:
        data = data.fillna("Unknown")
        return data


def impute_col(data, impute_method='fixed', val=0, verbose=False):
    """
    Intakes a dataframe and fills and missing values in an inputted column with the mean of the column
    Can also do median imputation by replacing "mean" with "median"

    data: a pandas dataframe or pd.Series
    impute_method: (string) form of imputation: [mean, median, fixed]
    val: the fixed value to impute with

    returns the dataframe after imputation
    """
    assert impute_method in ['mean', 'median', 'fixed']
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:,0]

    if impute_method == 'mean':
        impute_val = data.mean(skipna=True)
        # print(f"\t\tImputing with mean {impute_val}")
    elif impute_method == 'median':
        impute_val = data.median(skipna=True)
        # print(f"\t\tImputing with median {impute_val}")
    else:
        impute_val = val
    
    if str(impute_val) == 'nan':
        if verbose:
            print(f"\t\tImpute value found to be NaN. This happens when all data is missing. Imputing with default value of zero instead.")
        impute_val = 0
    data = data.fillna(impute_val)
    return data, impute_val


def impute_mean(data):
    return impute_col(data, 'mean')


def impute_median(data):
    return impute_col(data, 'median')


def impute_fixed(data, impute_val=0):
    return impute_col(data, 'fixed', val=impute_val)


def one_hot(data, categories='auto'):
    """
    Encodes all categorical variables in the input data via one hot encoding

    data: pd.dataframe or np.array categorical features to be encoded (should just be one column)
    categories: Can be one of the following:
        - auto (default): gets the categories from the data
        - (int) takes the categories as np.arange()
        - (list of ["##from_db", [3-elements]]) (col_name, table_name, schema_name) in the database that has the categories listed
            Note: the definition must match this exactly, else if will be treated as the next way of defining categories
        - (list) explicit list of categories

    returns: an array of arrays with OHE values
    """
    
    if isinstance(categories, int):
        categories = [list(np.arange(categories))]
    elif isinstance(categories, list) and categories[0] == "##from_db":
        assert len(categories[1]) == 3
        col_name, table_name, schema_name = categories[1]
        categories = run_sql_query(database_connection_object, read_cols_query([col_name], table_name, schema_name), 
                                   return_dataframe=True).values.flatten()[None, :].tolist()
    elif isinstance(categories, list):
        categories = [categories]
    else:
        assert categories == 'auto'

    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        data = data.values
    
    data = data[:, None]
    encoder = preprocessing.OneHotEncoder(sparse=False, categories=categories, handle_unknown='ignore')
    one_hot_labels = encoder.fit_transform(data)

    return one_hot_labels, list(encoder.categories_[0])


def normalize(data, params=None, thres=10):
    if params is None:
        min_val, max_val = data.values.min(axis=0), data.values.max(axis=0)
    else:
        min_val, max_val = params
    if np.all(min_val<-thres) or np.all(max_val>thres):
        data = (data - min_val)/(max_val-min_val)
    return data, [min_val, max_val]


def hash_column(data, params):
    """
    This function takes in a row of categorical data and applies the hash
    trick to it in order to convert it to numerical data

    Arguments:
    column - an array representing a column to encode values of
    n_features - # of features to
    params: (e.g  {"n_features": 100, "input_type": "string"})

    returns the arary of arrays of hashed values, one array per row
    """
    h = FeatureHasher()
    if params is not None:
        h.set_params(**params)
    else:
        params= h.get_params(deep=True)

    f = h.transform(data.values)

    return f.toarray(), params


def get_time_diff(end_dates, start_dates):
    """
    To get the time difference in number of days between list of dates
    
    :param end_dates: list of ending dates in "YYYY-MM-DD" or a pd.dataframe column of dates
    :param start_dates: list of starting dates in "YYYY-MM-DD" or a pd.dataframe column of dates
    :returns:
        (pd.dataframe) denoting the difference in number of days
    """
    assert len(end_dates) == len(start_dates)

    date_diff = []
    if not isinstance(end_dates[0], dt.date):
        end_dates = [dt.date(*[int(e) for e in ed.split("-")]) if not ed == '' else 'missing' for ed in end_dates ]
    if not isinstance(start_dates[0], dt.date):
        start_dates = [dt.date(*[int(e) for e in sd.split("-")]) if not sd == '' else 'missing'  for sd in start_dates]
    for i in range(len(end_dates)):
        if end_dates[i] == 'missing' or start_dates[i] == 'missing':
            diff = None
        else:
            diff = (end_dates[i]-start_dates[i]).days
        date_diff.append(diff)
    #print(date_diff)
    return pd.Series(date_diff)
    
    
def time_since_session_start(date, start_date):
    return get_time_diff(date, start_date)
   

def time_till_session_end(date, end_date):
    return get_time_diff(end_date, date)
    

def sponsor_pass_rate(bill_sponsors_data, intro_dates, bill_statuses, return_total_bills=False):
    """
    sponsors_data: pd.dataframe (should just be one column) with comma separated sponsor ids
    """
    bill_sponsors_data = bill_sponsors_data.values.tolist()
    intro_dates = intro_dates.values.tolist()
    bill_statuses = bill_statuses.values.tolist()
    if not isinstance(intro_dates[0], dt.date):
        intro_dates = [dt.date(*[int(e) for e in d.split("-")]) if not d == '' else 'missing' for d in intro_dates ]
    intro_dates, bill_sponsors_data, bill_statuses = zip(*sorted(zip(intro_dates, bill_sponsors_data, bill_statuses)))
    sponsor_data = {}
    intermediate_sponsor_data = {}
    sponsor_pass_rate = []
    sponsor_total_bills = []
    
    for idx, entry in enumerate(bill_sponsors_data):
        sponsors = entry.split(",")
        sponsors = [int(s) for s in sponsors]
        intro_date = intro_dates[idx]
        status = bill_statuses[idx]
        for date in list(intermediate_sponsor_data):
            if (intro_date-date).days > 365:
                for sponsor in intermediate_sponsor_data[date]:
                    passed = intermediate_sponsor_data[date][sponsor]['Passed']
                    total = intermediate_sponsor_data[date][sponsor]['Total']
                    if sponsor in sponsor_data:
                        sponsor_data[sponsor]['Passed'] += passed
                        sponsor_data[sponsor]['Total'] += total
                    else:
                        sponsor_data[sponsor] = {"Passed": passed, "Total": total}
                del intermediate_sponsor_data[date]
                        
        
        if not intro_date in intermediate_sponsor_data:
            intermediate_sponsor_data[intro_date] = {}
        passed = 0
        total = 0
        for sponsor in sponsors:
            if not sponsor in intermediate_sponsor_data[intro_date]:
                # we can change the total to start with to be non-zero 
                # to add some  sort of regularizing effect
                intermediate_sponsor_data[intro_date][sponsor] = {"Passed": 0, "Total": 0}
            intermediate_sponsor_data[intro_date][sponsor]['Total'] += 1
            if status == "Passed":
                intermediate_sponsor_data[intro_date][sponsor]['Passed'] += 1
            if sponsor in sponsor_data:
                passed += sponsor_data[sponsor]['Passed']
                total += sponsor_data[sponsor]['Total']
            else:
                # we can find better baseline numbers as well
                # such as the average number of bills introduced by a member
                passed += 0
                total += 0
        if total > 0:
            spr = 1.0*passed/total
        else:
            spr = None
        sponsor_total_bills.append(total)
        sponsor_pass_rate.append(spr)
    
    if return_total_bills:
        return pd.Series(sponsor_total_bills)
    return pd.Series(sponsor_pass_rate)


def sponsor_total_bills(bill_sponsors_data, intro_dates, bill_statuses):
    return sponsor_pass_rate(bill_sponsors_data, intro_dates, bill_statuses, return_total_bills=True)
  
def generate_label(bill_status, label_type='binary'):
    """generate label from bill status

    Args:
        bill_status_id ([array]): list of bill status
        label_type (str, optional): indicating what type of labelling must be used
            - must be one out of ['binary']

    Returns:
        [array]: label data
    """
    assert label_type in ['binary', 'binary_01', 'ternary']

    label = bill_status.copy()
    if label_type == 'binary' or label_type == 'binary_01':
        for i in range(len(label)):
            # pass_status_id is 4
            label[i] = 1 if bill_status[i] == 'Passed' else 0
        categories = 2
    elif label_type == "ternary":
        for i in range(len(label)):
            if bill_status[i] == "Passed":
                label[i] = 1
            elif bill_status[i] == "Failed":
                label[i] = 0
            else:
                label[i] = 2
        categories = 3
    if label_type == "binary_01":
        return np.asarray(label, dtype='float')
    else:
        return one_hot(np.array(label), categories=categories)[0]


def generate_feature(df, feature_ops, text_transforms, verbose=False):
    """generate feature from dataframe without label, process on each column according to process_types

    Args:
        df ([dataframe]): dataframe including text and processed numerical data
        feature_ops ([dict]): a mapping from dataframe column name to a list of operations
        text_transforms (dict): contains all info about the text transformations to be performed
    Returns:
        [array: float64]: features
    """
    features = []
    # this variable is to store the final transformation applied
    # so we can apply the same transformation on the validation data as well
    applied_feature_ops = {}
    for col_name in feature_ops:
        if verbose:
            print(col_name)
        applied_feature_ops[col_name] = {}
        col_name_to_read = col_name
        if "::" in col_name:
            col_name_to_read = col_name.split("::")[0]
        col = df[col_name_to_read]
        for op in feature_ops[col_name]:
            if verbose:
                print(f"\t{op}")
            # text ops
            if op in ['bow', 'tfidf', 'lda', 'bert']:
                feats = text_transforms[col_name_to_read][op].transform(col.values)
                applied_feature_ops[col_name][op] = text_transforms[col_name_to_read][op]
                col = pd.Series(feats)
            # ops that read more than one column
            elif op in SECONDARY_COL_OPS:
                sec_cols = []
                sec_col_names = feature_ops[col_name][op]
                for sec_col_name in sec_col_names:
                    sec_cols.append(df[sec_col_name])
                applied_feature_ops[col_name][op] = sec_col_names
                col = globals()[op](col, *sec_cols)
            # ops that work with only the one column
            else:
                op_params = feature_ops[col_name][op]
                applied_op = []
                if op_params:
                    col, applied_op = globals()[op](col, op_params)
                else:
                    col, applied_op = globals()[op](col)
                if "impute" in op:
                    applied_feature_ops[col_name]["impute_fixed"] = applied_op
                else:
                    applied_feature_ops[col_name][op] = applied_op

        if isinstance(col, pd.Series):
            col = col.values.tolist()
        elif isinstance(col, np.ndarray):
            col = col.tolist()
        features.append(col)

    isSparse = False
    for idx, f in enumerate(features):
        f = np.array(f)
        if len(f.shape) < 2:
            f = np.expand_dims(f, axis=-1)
        if scipy.sparse.issparse(f[0][0]) and f.shape[1]==1:
            f = scipy.sparse.vstack(f.squeeze(axis=-1))
            isSparse = True
        features[idx] = f
    if isSparse:
        features = scipy.sparse.hstack(features, format='csr')
    else:
        features = np.hstack(features)

    return features, applied_feature_ops


def get_cols_to_read(feature_ops, iter_num):
    # add all keys in feature_ops
    col_names = list(feature_ops.keys())
    # add the iter split column
    col_names += [f"iter_{iter_num}"]
    # to remove the duplicate col names from the list of column names
    for name in col_names: 
        if "::" in name: col_names.remove(name)
    # add the extra columsn needed by SECONDARY_COL_OPS performed
    names_to_add = []
    for col_name in feature_ops:
        for op in feature_ops[col_name]:
            if op in SECONDARY_COL_OPS:
                names_to_add += feature_ops[col_name][op]
    col_names += names_to_add
    return list(set(col_names))


def features_and_labels_iterator(conn, feature_ops, label_info, batch_size, schema_name="sketch", table_name="query_res"):
    col_names = get_cols_to_read(feature_ops, iter_num)
    # add the label column
    col_names += [label_info['label_col_name']]
    cols_query = read_cols_query(col_names, schema_name=schema_name, table_name=table_name)
    count_query = f"SELECT COUNT(bill_id) FROM {schema_name}.{table_name}"
    num_rows = run_sql_query(conn, count_query, return_dataframe=True)[0][0]
    num_batches = floor(num_rows / batch_size)

    for batch_idx in range(num_batches):
        batch_query = f"SELECT * FROM ({cols_query}) LIMIT {batch_size} OFFSET {batch_idx * batch_size}"
        batch_df = run_sql_query(conn, batch_query, return_dataframe=True)
        features = generate_feature(batch_df, feature_ops)
        labels = generate_label(batch_df['bill_status'].to_numpy(), label_info['label_type'])
        yield features, labels
        
    
def obtain_features_and_labels(conn, feature_ops, label_info, return_iterator=False, batch_size=256, schema_name="sketch",
                               table_name="query_res", iter_num=1, iter_val=1, text_transforms=None, index_col_name="bill_id", 
                               verbose=False):
    """
    Obtain features and labels from the data from the table

    :param conn: pSQL connection object
    :param feature_ops: dictionary mapping of the columns to choose in the feature vector, and ops to perform on it
    :param label_info: dict containing `label_type` and `label_col_name`
    :param return_iterator: If true, return an iterator function over features and labels,
        else return two numpy arrays containing all features and labels.
        (TODO: Correct implementation to account for the transformations requiring full column info)
    :param batch_size: Used if return_iterator=True. Specifies size of the batches.
    :param schema_name: Name of the schema holding the data table
    :param table_name: Name of the table from which to read the data
    :param iter_num: which iteration of training_validation is being done (i.e. which temporal block)
    :param iter_val: which data in that temporal block are we looking at
        [1: training, 2: training_buffer, 3:validation, 4:validation_buffer, 0:not included in that temporal block]
    :param text_transforms: The text transform dict, structured as {"col_name": {"op_name": op_object, ...}, ...}
        If None, it is created from the data given. 
        If given, it must have all the required transforms. 

    :return:
        If `return_iterator` is False, return a tuple to two numpy arrays, holding the features and labels
        If `return_iterator` is True, return a iterator that yields a tuple to two numpy arrays,
            holding the features and labels, in batches of size `batch_size`
    """
    global database_connection_object
    database_connection_object = conn
    condition_statement = f"WHERE iter_{iter_num}={iter_val}"
    if text_transforms is None:
        text_tranforms = create_text_transformations(conn, feature_ops, schema_name, table_name, iter_num, iter_val)
    if return_iterator:
        # TODO: Imputation if we read row wise in batches only, not reading all rows at once
        # TODO: Storing applied_feature_ops -- imputation itself is the issue here.
        return features_and_labels_iterator(conn, feature_ops, label_info, batch_size, schema_name, table_name)
    else:
        col_names = get_cols_to_read(feature_ops, iter_num)
        # add the label column
        if label_info['label_col_name'] not in col_names:
            col_names += [label_info['label_col_name']]
        if index_col_name not in col_names:
            col_names += [index_col_name]
        cols_query = read_cols_query(col_names, schema_name=schema_name, table_name=table_name)
        cols_query = put_condition_on_query(cols_query, condition_statement)
        data_df = run_sql_query(conn, cols_query, return_dataframe=True)
        features, applied_feature_ops = generate_feature(data_df, feature_ops, text_transforms, verbose)
        labels = generate_label(data_df[label_info['label_col_name']].to_numpy(), label_info['label_type'])
        ids = data_df[index_col_name].to_numpy()
        return (features, labels, ids), applied_feature_ops


def create_text_transformations(conn, feature_ops, schema_name, table_name, iter_num=None, iter_val=None,
                                text_vocabulary=None, verbose=False):
    """
    To create all the text transfomation objects needed
    
    If either iter_num or iter_val is None, all of the data is used. 
    
    :return:
        a dictionary with mapping from col_name to its vectorizer object (for the text transformations)
    """
    text_transformation_objects = {}
    use_condition = not (iter_num is None or iter_val is None)
    for col_name in feature_ops:
        X = []
        for op in feature_ops[col_name]:
            # if we have to perform a text transformation
            if op in ['bow', 'tfidf', 'lda', 'bert']:
                if not col_name in text_transformation_objects:
                    text_transformation_objects[col_name] = {}
                cols_to_read = [col_name]
                if use_condition:
                    cols_to_read.append(f"iter_{iter_num}")
                col_query = read_cols_query(cols_to_read, schema_name=schema_name, table_name=table_name)
                if use_condition:
                    condition = f"WHERE iter_{iter_num}={iter_val}"
                    col_query = put_condition_on_query(col_query, condition)
                data_df = run_sql_query(conn, col_query, return_dataframe=True)
                data = data_df[col_name].values.tolist()
                
                if text_vocabulary is None:
                    vectorizer = CountVectorizer()
                else:
                    vocab = text_vocabulary[col_name]
                    vectorizer = CountVectorizer(vocabulary=vocab)
                    
                if verbose:
                    print(f"Creating {op} transformation objects on {col_name}")
                if op == 'bow':
                    text_transform_object = vectorizer.fit(data)
                elif op == 'tfidf':
                    text_transform_object = Pipeline([('count', vectorizer), ('tfidf', TfidfTransformer())]).fit(data)
                elif op == 'lda':
                    n_buckets = feature_ops[col_name][op]  
                    text_transform_object = Pipeline([('count', vectorizer), 
                                                      ('lda', LatentDirichletAllocation(n_components=n_buckets))]).fit(data)
                else:
                    text_transform_object = None
                text_transformation_objects[col_name][f'{op}'] = text_transform_object

    return text_transformation_objects

if __name__ == "__main__":
    pass
    # bill_status_id = [4, 1, 2, 4, 0, 3]
    # print(generate_label(bill_status_id))

    # df = pd.DataFrame(data={'text': ["aaa", "bbb"], 'number': [3,4]})
    # features = generate_feature(df)
    # print(features)
