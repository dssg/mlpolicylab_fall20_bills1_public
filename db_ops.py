import pandas as pd
import psycopg2 as pg2
import matplotlib.pyplot as plt
import warnings
import yaml
import seaborn as sns
import io
warnings.filterwarnings('ignore')

'''
dict = {'Index': [1, 2, 3, 4],
        'First Score':[96, 97, 98, 99], 
        'Second Score': [30, 45, 56, 90], 
        'Third Score':[90, 40, 80, 98]} 
  
# creating a dataframe using dictionary 
df = pd.DataFrame(dict)
'''

def open_db_connection(secrets_file="secrets.yaml", verbose=True):
    """
    Opens connection to psql db

    :return:
        connection object
    """
    try:
        with open(secrets_file, 'r') as f:
            # loads contents of secrets.yaml into a python dictionary
            secret_config = yaml.safe_load(f.read())
            db_params = secret_config['db']
    except FileNotFoundError:
        print("Cannot establish connection to database. Please provide db_params in secrets.yaml file.")
        exit(1)

    conn = pg2.connect(
        host=db_params['host'],
        port=db_params['port'],
        dbname=db_params['dbname'],
        user=db_params['user'],
        password=db_params['password']
    )
    if verbose:
        print(f"Connection opened to database {db_params['dbname']}")
    return conn

# connection = open_db_connection()


def clear_table(conn, schema_name="sketch", table_name="query_res"):
    """
    Drop the table if it exists

    :param conn: pSQL connection object
    :param schema_name: (str) Name of the schema
    :param table_name: (str) Name of the table
    :return:
    """
    cur = conn.cursor()
    drop_command = f"DROP TABLE IF EXISTS {schema_name}.{table_name};"
    print(drop_command)
    cur.execute(drop_command)
    conn.commit()


def run_sql_query(conn, query, return_dataframe=False, schema_name="sketch", table_name="query_result"):
    """
    Run the given SQL query on the database connection object provided

    :param conn: pSQL DB connection object
    :param query: (str) The query to be run on the database
    :param return_dataframe: (bool) whether to return the output of the query as a DataFrame object, or to write it as
        another table in the database
    :param schema_name: (str) Name of the schema to which the table needs to be added
    :param table_name: (str) Name of the table on which we want to write the output of the query
        (If the table already exists, it is dropped, and overwritten)

    :return:
        If `return_data` is True, a pandas.DataFrame object is returned with the output from the query,
        else returns None.
    """
    cur = conn.cursor()

    if not return_dataframe:
        # creating schema
        schema_command = f"CREATE SCHEMA IF NOT EXISTS {schema_name};"
        cur.execute(schema_command)

        # check if this table exists
        # check_existence_command = f"SELECT EXISTS (SELECT * FROM information_schema.tables \
        #                                 WHERE table_schema='{schema_name}' AND table_name='{table_name}');"
        check_drop_command = f"DROP TABLE IF EXISTS {schema_name}.{table_name};"
        cur.execute(check_drop_command)
        
        # remove the ending semicolon from the query
        if query.strip()[-1] == ";":
            query = query.strip()[:-1]
        appended_query = f"CREATE TABLE {schema_name}.{table_name} AS ({query});"

        # give permission to all
        # permission_query = f"GRANT ALL PRIVILEGES ON {schema_name}.{table_name} TO PUBLIC;"

        # change table owner
        permission_query = f"ALTER TABLE {schema_name}.{table_name} OWNER TO bills1;"
        
        print(appended_query)
        cur.execute(appended_query)
        cur.execute(permission_query)
        conn.commit()

    if return_dataframe:
        cur.execute(query)
        try:
            rows = cur.fetchall()
            col_names = [desc[0] for desc in cur.description]
            data_df = pd.DataFrame(rows)
            data_df.columns = col_names
        except pg2.ProgrammingError:
            data_df = None
        conn.commit()
        cur.close()
        return data_df
    cur.close()


def status_info(object_name, state_id, schema_name="sketch", table_name="query_res"):
    """
    Summarize status information for a state

    :param object_name: either a pSQL DB connection object or a pandas.DataFrame object
        - if it is a pandas.DataFrame object, it should hold the data for the state, having columns of
            `state_id`, `status`, `count_year`.
        - if it is a pSQL connection object, it reads the data from the schema and table provided
            that table should then have the corresponding columns.
    :param state_id: (int) integer ID of the state we want the summary for
    :param schema_name: (str) Name of the schema having the table
    :param table_name: (str) Name of the table to read the state-bill data from

    :return: tuple of 3 pd.Series (a1, a2, a3)
        - a1 is distribution of all labels in state x (including all progress levels of all bills)
        - a2 is distribution of final status label for the bills
        - a3 is distribution of relabeled final status labels for the bills
    """

    if type(object_name) == pd.DataFrame:
        df = object_name
    else:
        read_query = f"SELECT state_id, status, count_year FROM {schema_name}.{table_name}"
        df = run_sql_query(object_name, read_query, return_dataframe=True)

    df_x = df[df['state_id'] == state_id]
    df_x['status_1'] = df_x['status']
    df_x['status_1'][(df_x['count_year'] >= 2) & (df_x['status_1'] != 'Passed') & (df_x['status_1'] != 'Failed')] = 'AssignFail'

    label_dist_type1 = df_x.groupby('status').size()
    label_dist_type2 = df_x[df_x['seq'] == 1].groupby('status').size()
    label_dist_type3 = df_x[df_x['seq'] == 1].groupby('status_1').size()
    return label_dist_type1, label_dist_type2, label_dist_type3


def write_col_in_table(conn, id, col, col_type, col_name, schema_name, table_name):
    """
    Updates the column values in the given column of the table with the provided col values

    :param id: a list of values that will be mapped to the values
        of the col parameter for adding to the db
    :param col: a list of values to be written to the provided column of the table
    :param conn: a pSQL databse connection object
    :param col_name: name of the column to write
        - if the column exists, we will get an error for now (#TODO: overwrite)
        - if doesnt exist, we create and add to it
    :param schema_name: name of the schema for the table
    :param table_name: name of the table

    :return:

    """
    cur = conn.cursor()
    # create the table write to if not exist, add id column
    cur.execute(f"""CREATE TABLE IF NOT EXISTS {schema_name}.{table_name} (id integer PRIMARY KEY);""")

    # if id column doesn't has values, add sequential id values from id
    cur.execute(f"""select exists (select * from {schema_name}.{table_name} where id>=0);""")
    has_id = cur.fetchone()[0]
    if not has_id:
        ids = zip(id)
        cur.executemany(f"""INSERT INTO {schema_name}.{table_name} (id) VALUES(%s)""", ids)
    
    # add coloumn into the table write to 
    cur.execute(f"""ALTER table {schema_name}.{table_name} ADD COLUMN {col_name} {col_type.upper()};""")
    rows = zip(id, col)
    cur.execute(f"""CREATE TEMP TABLE codelist(id integer,
     col {col_type.upper()}) ON COMMIT DROP""")
    cur.executemany("""INSERT INTO codelist (id, col) VALUES(%s, %s)""", rows)
    cur.execute(f"""
        UPDATE {schema_name}.{table_name}
        SET {col_name} = codelist.col
        FROM codelist
        WHERE {schema_name}.{table_name}.id = codelist.id;
    """)
    conn.commit()
    cur.close()


def write_df_in_table(conn, df, schema_name, table_name):
    """write pandas dataframe in table

    Args:
        conn: a pSQL databse connection object
        df: a pandas dataframe to write to the database
        schema_name: name of the schema for the table
        table_name: name of the table
    """
    # write df to memory buffer
    SEP = "~"
    buffer = io.StringIO()
    df.to_csv(buffer, index_label='id', header=False, sep=SEP)
    buffer.seek(0)

    type_mapping = {'int64': 'integer', 'float64': 'double precision', 'object': 'varchar'}
    cur = conn.cursor()

    cur.execute(f"DROP TABLE IF EXISTS {schema_name}.{table_name};")
    cur.execute(f"CREATE TABLE {schema_name}.{table_name} (id integer PRIMARY KEY);")
    # cur.execute(f"GRANT ALL PRIVILEGES ON {schema_name}.{table_name} TO bills1;")
    cur.execute(f"ALTER TABLE {schema_name}.{table_name} OWNER TO bills1;")
    # create table column
    for col_name, col_type in zip(df.columns, df.dtypes):
        print(col_name)
        col_type = type_mapping[str(col_type)]
        cur.execute(f"ALTER table {schema_name}.{table_name} ADD COLUMN {col_name} {col_type};")

        # hard-coded for now, may be made dynamic later
        # TODO: need to figure out how to change NULL values to date as well
        if col_name == "introduced_date":
            cur.execute(f"""ALTER table {schema_name}.{table_name} ALTER COLUMN {col_name}
                            TYPE date using to_date({col_name}, 'YYYY-MM-DD');""")
    # copy data from buffer to table
    cur.copy_from(buffer, f'{schema_name}.{table_name}', sep=SEP)

    conn.commit()
    cur.close()


def visualize_float_columns(object_name, float_cols, schema_name="sketch", table_name="query_res"):
    """
    visualizes continous variables from box plots

    :param object_name: either a pSQL DB connection object or a pandas.DataFrame object
        - if it is a pandas.DataFrame object, it should include numerical variables that we are 
            interested in the distribution
        - if it is a pSQL connection object, it reads the data from the schema and table provided
    :param float_cols: (list) list of numerical variables we want the summary for
    :param schema_name: (str) Name of the schema having the table
    :param table_name: (str) Name of the table to read the state-bill data from

    :return: boxplots

    """
    if type(object_name) == pd.DataFrame:
        df = object_name
    else:
        read_query = f"SELECT * FROM {schema_name}.{table_name}"  
        df = run_sql_query(object_name, read_query, return_dataframe=True)

    for i in float_cols:
        df.boxplot(i)
        plt.show()


def missing_value_count(object_name, schema_name="sketch", table_name="query_res"):
    """
    identify missing data

    :param object_name: either a pSQL DB connection object or a pandas.DataFrame object
        - if it is a pandas.DataFrame object, it should include variables that we are 
            interested in identifying missing values
        - if it is a pSQL connection object, it reads the data from the schema and table provided
    :param schema_name: (str) Name of the schema having the table
    :param table_name: (str) Name of the table to read the state-bill data from

    :return: a list of tuples, each tuple includes column name, total number, number of missing value
    """

    if type(object_name) == pd.DataFrame:
        df = object_name
    else:
        read_query = f"SELECT * FROM {schema_name}.{table_name}"  
        df = run_sql_query(object_name, read_query, return_dataframe=True)

    return [(x, df.shape[0], df[x].isna().sum()) for x in df.columns]


def plot_heatmap(object_name, float_cols, schema_name="sketch", table_name="query_res"):
    """
    visualizes correlation of continous variables 

    :param object_name: either a pSQL DB connection object or a pandas.DataFrame object
        - if it is a pandas.DataFrame object, it should include numerical variables that we are 
            interested in their correlation
        - if it is a pSQL connection object, it reads the data from the schema and table provided
    :param float_cols: (list) list of numerical variables we want the summary for
    :param schema_name: (str) Name of the schema having the table
    :param table_name: (str) Name of the table to read the state-bill data from

    :return: heatmap
    """

    if type(object_name) == pd.DataFrame:
        df = object_name
    else:
        read_query = f"SELECT * FROM {schema_name}.{table_name}"  
        df = run_sql_query(object_name, read_query, return_dataframe=True)

    correlation=df[float_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation, ax=ax)
    plt.show()

