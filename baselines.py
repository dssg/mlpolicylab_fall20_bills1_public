from queries import read_cols_query, put_condition_on_query
from db_ops import run_sql_query
import numpy as np
from tqdm import tqdm

def get_sponsor_and_party_data(conn, iter_num, iter_val, table_name="bill_sponsor_status", schema_name="sketch"):
    query = read_cols_query(['sponsor_id', 'party_id', 'bill_id', 'status_new'], table_name, schema_name)
    query = put_condition_on_query(query, f" WHERE iter_{iter_num}={iter_val}")
    data_df = run_sql_query(conn, query, return_dataframe=True)
    return data_df.values


def get_sponsor_and_party_info(input_x, data):
    sponsor_info = {}
    party_info = {}

    n, c = data.shape

    for i in range(n):
        sponsor_id, party, bill_id, status = data[i]
        if not bill_id in input_x:
            continue
        if not sponsor_id in sponsor_info:
            sponsor_info[sponsor_id] = {"passed": 0, "failed": 0}
        if not party in party_info:
            party_info[party] = {"passed": 0, "failed": 0}

        if status == "Passed":
            sponsor_info[sponsor_id]["passed"] += 1
            party_info[party]["passed"] += 1
        else:
            sponsor_info[sponsor_id]["failed"] += 1
            party_info[party]["failed"] += 1

    return sponsor_info, party_info
    
    
def get_base_pass_rates(input_x, data):
    """
        data  is output of get_sponsor_and_party_data()
        input_x is a list of bill_ids that are available to use
    """
    sponsor_info, party_info = get_sponsor_and_party_info(input_x, data)
    sponsor_pass_rates = {}
    party_pass_rates = {}
    for sponsor_id in sponsor_info:
        pass_rate = sponsor_info[sponsor_id]['passed']/(sponsor_info[sponsor_id]['passed']+sponsor_info[sponsor_id]['failed'])
        sponsor_pass_rates[sponsor_id] = pass_rate
    for party in party_info:
        pass_rate = party_info[party]['passed']/(party_info[party]['passed']+party_info[party]['failed'])
        party_pass_rates[party] = pass_rate
    return sponsor_pass_rates, party_pass_rates


def baseline1_predict(data, input_x, sponsor_pass_rates, party_pass_rates):
    """
        data  is output of get_sponsor_and_party_data()
        input_x is just bill_ids of the bills to be predicted on
    """
    mean_spr = np.mean(list(sponsor_pass_rates.values()))
    mean_ppr = np.mean(list(party_pass_rates.values()))
    
    y_pred = []
    y_true = []
    for bill_id in tqdm(input_x):
        bill_data = data[data[:, 2]==bill_id]
        if bill_data[0][3] == "Passed":
            y_true.append(1)
        else:
            y_true.append(0)
        spr = 0.0
        ppr = 0.0
        for entry in bill_data:
            sponsor, party, _, _ = entry
            if sponsor in sponsor_pass_rates:
                spr += sponsor_pass_rates[sponsor]
            else:
                spr += mean_spr
            if party in party_pass_rates:
                ppr += party_pass_rates[party]
            else:
                ppr += mean_ppr
        spr /= bill_data.shape[0]
        ppr /= bill_data.shape[0]
        score = np.sqrt(spr*ppr)
        y_pred.append(score)
    return y_pred             
    

    
    
    