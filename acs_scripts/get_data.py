import censusdata
from gather_data import *
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--search', action='store_true',
                    help="To perform a search for variables")
parser.add_argument('--get', action='store_true',
                    help="To load variables data from the CENSUS data and store to csv")
parser.add_argument('--store', action='store_true',
                    help="To load data from csv into the database")
args = parser.parse_args()

if args.search:
    # to search for variables in CENSUS data
    vars = censusdata.search('acs5', 2018, 'label', 'race', tabletype='detail')
    print(f"Found {len(vars)} matching variables.")
    # prints all retrieved census data variables to file
    with open("search_results.txt", "w") as f:
        for v in vars:
            f.write(str(v)+"\n")

if args.get:
    # to download the data from the CENSUS
    df = download_data('useful_variables.txt')
    # saves the retrieved data to a csv
    df.to_csv('data.csv', index=False)

if args.store:
    table_name = "acs_upper_chamber_illinois"
    schema_name = "sketch"

    conn = open_db_connection()
    cur = conn.cursor()

    # creating schema
    schema_command = f"CREATE SCHEMA IF NOT EXISTS {schema_name};"
    print(schema_command)
    cur.execute(schema_command)

    # drop table if it exists already
    drop_command = f"DROP TABLE IF EXISTS {schema_name}.{table_name};"
    print(drop_command)
    cur.execute(drop_command)

    # creating table
    create_command = os.popen(f"cat data.csv | tr [:upper:] [:lower:] | tr ' ' '_' | sed 's/#/num/' | " + \
                       f"csvsql -i postgresql --db-schema {schema_name} --tables {table_name} ").read()
    print(create_command)
    cur.execute(create_command)

    # building primary index for table
    with open("data.csv", "r") as f:
        header = f.readlines()[0].split(",")

    if header[0] == 'state':
        keys = 'state'
    else:
        keys = 'us'
    if header[1] in ['county', 'state_legislative_district_upper_chamber', 'state_legislative_district_lower_chamber']:
        keys += f', {header[1]}'
    if header[2] == 'tract':
        keys += ', tract'
    if header[3] == 'block_group':
        keys += ", block_group"
    elif header[3] == 'block':
        keys += ', block'

    index_command = f"ALTER TABLE {schema_name}.{table_name} ADD PRIMARY KEY ({keys})"
    print(index_command)
    cur.execute(index_command)

    # copying data to table
    with open("data.csv", "r") as f:
        next(f) #ignore header
        cur.copy_from(f, f"{schema_name}.{table_name}", sep=',')

    # Close communication with the database
    cur.close()
    conn.commit()
    conn.close()

