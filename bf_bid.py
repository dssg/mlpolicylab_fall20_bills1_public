import pandas as pd
import psycopg2 as pg2
import yaml
import io
import ohio.ext.pandas
from sqlalchemy import create_engine 




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

connection = open_db_connection()

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
        #if col_name == "introduced_date":
        #    cur.execute(f"""ALTER table {schema_name}.{table_name} ALTER COLUMN {col_name}
        #                   TYPE date using to_date({col_name}, 'YYYY-MM-DD');""")
    # copy data from buffer to table
    cur.copy_from(buffer, f'{schema_name}.{table_name}', sep=SEP)

    conn.commit()
    cur.close()


# If you need to recreate the SQL tables for whatever reason

object = pd.read_pickle(r'/data/groups/bills1/mlpolicylab_fall20_bills1/bid_groups.pkl')
white_df = pd.DataFrame(object['white'], columns=['bill_id'])
write_df_in_table(conn=connection, df=white_df, schema_name="sketch", table_name="reference_bills_w")
"""
black_df = pd.DataFrame(object['black'], columns=['bill_id'])
asian_df = pd.DataFrame(object['asian'], columns=['bill_id'])

write_df_in_table(conn=connection, df= black_df, schema_name="sketch", table_name="protected_bills_b")

write_df_in_table(conn=connection, df= asian_df, schema_name="sketch", table_name="protected_bills_a")
"""


