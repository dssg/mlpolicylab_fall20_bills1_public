from queries import read_cols_query 
from db_ops import run_sql_query, open_db_connection
import numpy as np
from tqdm import tqdm 
import pickle as pkl

conn = open_db_connection("../secrets.yaml", verbose=False)
# query = "select * from sketch.bill_district"
# query = """select s.bill_id, s.sponsor_id, district from ml_policy_class.bill_sponsors s
#     inner join 
#     (
#         select person_id, state_id, district from ml_policy_class.sessions_people sp
#         where sp.state_id = 13
#     ) as sp1
#     on sp1.person_id = s.sponsor_id"""
print("running query")

query = "select sponsor_id, bs.bill_id, session_id from ml_policy_class.bill_sponsors bs join ml_policy_class.bills b on b.bill_id=bs.bill_id"
data1 = run_sql_query(conn, query, return_dataframe=True).values
print('got data1', data1.shape)
query = "select session_id, person_id, district from ml_policy_class.sessions_people where state_id = 13"
data2 = run_sql_query(conn, query, return_dataframe=True).values
print('got data2', data2.shape)

data_state = {}
for d in tqdm(data2):
    sessid, pid, dis = d
    data_state[(sessid, pid)] = dis

data = []
for d in tqdm(data1):
    spid, bid, sessid = d
    if (sessid, spid) not in data_state:
        continue
    data.append([bid, data_state[(sessid, spid)]])
    
    
#exit(0)
#data = run_sql_query(conn, query, return_dataframe=True)
#print("converting to np")
#data = data.values
#print("data ready")

print("data_dict creation started")
data_dict = {}

for d in tqdm(data):
    #print(d)
    bid, dis = d
    if not bid in data_dict:
        data_dict[bid] = {"upper": [], "lower":[]}
    house, num = dis.strip().split("-")
    num = int(num)
    if house == "SD":
        data_dict[bid]["upper"].append(num)
    elif house == "HD":
        data_dict[bid]["lower"].append(num)

print("data_dict created")

dist_dict = {
    "black": {
        "upper": [17,15,13,14,5,16,4,3], 
        "lower": [29,33,34,26,27,38,5,30,9,8,31,25,10,28,32,7]
    },
    "asian": {
        "upper": [8,23,28,1,9],
        "lower": [16,2,15,44,17]
    }
}

output_bids = {}
for race in dist_dict:
    print(race)
    output_bids[race] = []
    for bid in tqdm(data_dict):
        if len(set(data_dict[bid]["upper"]).intersection(set(dist_dict[race]["upper"]))) > 0:
            output_bids[race].append(bid)
        if len(set(data_dict[bid]["lower"]).intersection(set(dist_dict[race]["lower"]))) > 0:
            output_bids[race].append(bid)
    
output_bids["white"] = list(data_dict.keys())
print("white")
for bid in tqdm(data_dict):
    if (bid in output_bids["black"]) or (bid in output_bids["asian"]):
        output_bids["white"].remove(bid)

print("writing to disk")
with open("bid_groups.pkl","wb") as f:
    pkl.dump(output_bids, f)