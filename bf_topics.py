import pandas as pd
import psycopg2 as pg2
import yaml
import io
import ohio.ext.pandas
from sqlalchemy import create_engine 
from collections import Counter
import nltk
from nltk.corpus import stopwords
import json



with open('secrets.yaml', 'r') as f:
    secrets = yaml.safe_load(f)

db_params = secrets['db']
engine = create_engine('postgres://{user}:{password}@{host}:{port}/{dbname}'.format(
  host=db_params['host'],
  port=db_params['port'],
  dbname=db_params['dbname'],
  user=db_params['user'],
  password=db_params['password']    
))

"""
il_bills_table = 'il_bills'
table_schema = 'sketch'

il_bills = pd.DataFrame.pg_copy_from(il_bills_table, engine, schema=table_schema)
il_bills = il_bills.apply(lambda x: x.astype(str).str.lower())
keyword_test = il_bills['bill_description'].str.contains(r'immigrant|immigration|visa|undocumented|sanctuary cit|deferred action for childhood arrivals|daca|h-1b|illegal alien|non-citizen|noncitizen|first gen|1st gen|deport|asylum|migrant|foreign worker')
il_bills['keyword_check'] = keyword_test
count_keywords = keyword_test.sum()
print(count_keywords)

il_bills.pg_copy_to('il_bills2', engine, schema=table_schema, index=False, if_exists='append')
"""

# Keyword count for asian- and african-american high proportion district bills

protected_bills_a = 'protected_bills_a'
protected_bills_b = 'protected_bills_b'
table_schema = 'sketch'

"""
#8,804 rows
protected_df_a = pd.DataFrame.pg_copy_from(protected_bills_a, engine, schema=table_schema)
protected_df_a = protected_df_a.apply(lambda x: x.astype(str).str.lower())
keyword_test_a = protected_df_a['doc'].str.contains(r'china|chinese|asia|asian|korea|japan|india')
count_keywords_a = keyword_test_a.sum()
print(count_keywords_a)

#14,863 rows
protected_df_b = pd.DataFrame.pg_copy_from(protected_bills_b, engine, schema=table_schema)
protected_df_b = protected_df_b.apply(lambda x: x.astype(str).str.lower())
keyword_test_b = protected_df_b['doc'].str.contains(r'africa|african american')
count_keywords_b = keyword_test_b.sum()
print(count_keywords_b)
"""
reference_bills_w = 'reference_bills_w'
stop = stopwords.words('english')

# Get data from db
reference_df = pd.DataFrame.pg_copy_from(reference_bills_w, engine, schema=table_schema)
reference_df['bill_description'] = reference_df['bill_description'].astype(str)

# Remove stopwords
reference_df['bill_description'] = reference_df['bill_description'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))

# Count the occurances of each word
results = Counter()
reference_df['bill_description'].str.lower().str.split().apply(results.update)

# Sort the dict
results = dict(sorted(results.items(), key=lambda item: item[1]))

# Dump dict to a json since it's so big
"""
json = json.dumps(results)
f = open("dict.json","w")
f.write(json)
f.close()
"""

# PROTECTED A
# Get data from db
protected_df_a = pd.DataFrame.pg_copy_from(protected_bills_a, engine, schema=table_schema)
protected_df_a['bill_description'] = protected_df_a['bill_description'].astype(str)

# Remove stopwords
protected_df_a['bill_description'] = protected_df_a['bill_description'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))

# Count the occurances of each word
results_a = Counter()
protected_df_a['bill_description'].str.lower().str.split().apply(results_a.update)

# Sort the dict
results_a = dict(sorted(results_a.items(), key=lambda item: item[1]))

# Dump dict to a json since it's so big
"""
json = json.dumps(results_a)
f = open("protected_dict_a.json","w")
f.write(json)
f.close()
"""

#print({k:v for k,v in results.items() if k not in results_a})
#print({k:v for k,v in results_a.items() if k not in results})

# PROTECTED B
# Get data from db
protected_df_b = pd.DataFrame.pg_copy_from(protected_bills_b, engine, schema=table_schema)
protected_df_b['bill_description'] = protected_df_b['bill_description'].astype(str)

# Remove stopwords
protected_df_b['bill_description'] = protected_df_b['bill_description'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))

# Count the occurances of each word
results_b = Counter()
protected_df_b['bill_description'].str.lower().str.split().apply(results_b.update)

# Sort the dict
results_b = dict(sorted(results_b.items(), key=lambda item: item[1]))

# Dump dict to a json since it's so big

json = json.dumps(results_b)
f = open("protected_dict_b.json","w")
f.write(json)
f.close()


#print({k:v for k,v in results.items() if k not in results_a})
print({k:v for k,v in results_a.items() if k not in results_b})
