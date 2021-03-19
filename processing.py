import string
import re
import time
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer # or LancasterStemmer, RegexpStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer

from multiprocessing import Pool

from queries import read_cols_query, add_block_val_columns_query
from db_ops import run_sql_query, write_col_in_table, write_df_in_table

"""
#Test dataframe
dict = {'First Score':[100, 90, np.nan, 95], 
        'Second Score': [30, 45, 56, np.nan], 
        'Third Score':[np.nan, 40, 80, 98]} 
  
# creating a dataframe using dictionary 
df = pd.DataFrame(dict) 
"""

default_stemmer = PorterStemmer()
default_lemmatizer = WordNetLemmatizer()
default_stopwords = stopwords.words('english') 

# Function credits:
# https://stackoverflow.com/questions/48865150/pipeline-for-text-cleaning-processing-in-python
def clean_text(text):

    def tokenize_text(text):
        return [w for w in word_tokenize(text)]

    def remove_special_characters(text, characters=string.punctuation+string.digits):
        return text.translate(str.maketrans('', '', characters))

    def stem_text(tokens, stemmer=default_stemmer):
        return [stemmer.stem(t) for t in tokens]
    
    def lemmatize_text(tokens, lemmatizer=default_lemmatizer):
        return [lemmatizer.lemmatize(t) for t in tokens]

    def remove_stopwords(tokens, stop_words=default_stopwords):
        tokens = [w for w in tokens if w not in stop_words]
        return tokens
    
    if text is None or text[0] is None:
        return None
    text = text[0]
    text = text.strip(' ') # strip whitespaces
    text = text.lower() # lowercase
    text = remove_special_characters(text) # remove punctuation and symbols
    text_tokens = tokenize_text(text)
    text_tokens = remove_stopwords(text_tokens) # remove stopwords
    text_tokens = stem_text(text_tokens)    # stemming
    # text_tokens = lemmatize_text(text_tokens) # lemmatizing
    text = " ".join(text_tokens)
    text = text.strip(' ') # strip whitespaces again

    return text


def clean_text_data(texts):
    texts = texts.values.tolist()

    print("Cleaning text data...")
    
    pool = Pool(20)
    start = time.time()
    cleaned_texts = pool.map(clean_text, texts)
    pool.close()
    pool.join()
    print("time: ", time.time()-start)
    return pd.DataFrame(cleaned_texts)


def rename_col(col, new_name):
    """
    renames a pandas column
    
    :param col: a pd.dataframe having a single column
    :param new_name: (str) new name to give to that column
    :return:
        A pd.dataframe having a single column (renamed) 
    """
    col.columns = [new_name]
    return col


def retype_col(col, new_type):
    """
    renames a pandas column
    
    :param col: a pd.dataframe having a single column
    :param new_name: (str) new name to give to that column
    :return:
        A pd.dataframe having a single column (renamed) 
    """
    col = col.astype(new_type)
    return col
    

def preprocess_data(conn, preprocess_ops, input_schema_name, input_table_name, output_schema_name, output_table_name):
    """
    :param conn: a database connection object
    :param preprocess_ops: dict mapping from column_names to operation_type
        - column names must be names of columns present in the table
        - operation type must be one of ['one_hot', 'mean_impute_col', 'add_dummy_col']
    :param schema_name: Name of schema holding the table
    :param table_name: Name of the table from which to read the columns

    :return:
    """
    data = []
    for col_name in preprocess_ops:
        ops = preprocess_ops[col_name]
        read_query = read_cols_query([col_name], table_name=input_table_name, schema_name=input_schema_name)
        col = run_sql_query(conn, read_query, return_dataframe=True)

        for op in ops:
            if "rename" in op:
                op, new_name = op.split("::")
                col = rename_col(col, new_name)
            elif "retype" in op:
                op, new_type = op.split("::")
                col = retype_col(col, new_type)
            else:
                col = globals()[op](col)
        
        # previously write col to table
        # col_type = type_mapping[str(col.iloc[0].dtype)]
        # col_values = col[col_name].to_list()
        # index = list(map(int, col.index.values))
        # write_col_in_table(conn, index, col_values, col_type, col_name, output_schema_name, output_table_name)

        data.append(col)
    df = pd.concat(data, axis=1)
    write_df_in_table(conn, df, output_schema_name, output_table_name)


def create_temporal_blocks(conn, schema_name, table_name, year_col="introduced_date", start_year=2009, end_year=2019,
                            split_list=(2,1,1,1), update_freq=2, type_val=0, verbose=False):
    """
    Define the temporal blocks, and add that information to the database

    :param conn: a database connection object
    :param schema_name: name of schema holding the table
    :param table_name: name of table having the data rows
    :param year_col: The name of the column in table holding the years to create temporal blocks on
    """
    query, num_blocks = add_block_val_columns_query(table_name, schema_name, year_col, start_year=start_year, end_year=end_year,
                                        split_list=split_list, update_freq=update_freq, type_val=type_val, return_num_blocks=True)
    if verbose:
        print(f"Created {num_blocks} temporal splits")
    run_sql_query(conn, query, return_dataframe=True)


if __name__ == "__main__":
    df = pd.DataFrame({'a':[96, 97, 98, 99], 
                       'b': [1.11, 2.22, 3.33, 4.44], 
                       'c':['abc', 'de', 'fg', 'hijk']})