def map_bill_to_state_query(params):
    """
    To obtain SQL query to map all the bills to their state IDs.
    Params is a dict with the following parameters.
    :param start_year: (int)
    :param end_year: (int)
        The years between which we get all the bills
    :return:
    A string have the SQL query for the task
    """
    query = f"""
        select state_id, bill_id
        from ml_policy_class.bills b
        join ml_policy_class.sessions s
        on b.session_id = s.session_id
        where to_char(introduced_date, 'YYYY') >= '{params['start_year']}'  and  to_char(introduced_date, 'YYYY') <= '{params['end_year']}'
        order by state_id;
        """
    return query

def bill_date_difference(params):
    """
     To obtain SQL query to create table which includes attributes bill_id and date since introduced

    :param 

    :return:
    A string have the SQL query for the task   
    """
    query = f"""
        select bill_id, d1-d2 as diff from
        (select bill_id,table3.progress_date as d1,table7.progress_date as d2  from(
        select * from(
        select *, row_number() over(
        partition by bill_id
        order by progress_date desc, index desc) as seq
        from(select *, ROW_NUMBER() over() as index from ml_policy_class.bill_progress) as table1) as table2
        where seq = 1) as table3
        join
        (select * from(
        select * from(
        select *, row_number() over(
        partition by bill_id
        order by progress_date, index desc) as seq_reverse
        from(select *, ROW_NUMBER() over() as index from ml_policy_class.bill_progress) as table4) as table5
        where seq_reverse = 1) as table6) as table7
        using (bill_id)) as table8;
        """
    return query


def rank_by_bill_count_query(params):
    """
     To obtain SQL query to rank states based on number of bills.

    :param start_year: (str)
    :param end_year: (str)
        The years between which we get all the bills
    :return:
    A string have the SQL query for the task   
    """
    query = f"""
        select st.state_id, state_abbreviation, count(bill_id), ROW_NUMBER() over(
        order by count(bill_id) desc)as rank from
        (select bill_id, b.session_id, state_id, to_char(introduced_date, 'YYYY') as year
        from ml_policy_class.bills b
        join ml_policy_class.sessions s
        on b.session_id = s.session_id
        where to_char(introduced_date, 'YYYY') >= {params['start_year']} and to_char(introduced_date, 'YYYY') <= {params['end_year']}) as table1
        join catalogs.states st
        on table1.state_id = st.state_id
        group by st.state_id,state_abbreviation
        order by count desc;
        """
    return query


def bill_detail_query(params):
    """
     To obtain SQL query to get detailed information about bill, including bill_id, and corresponding 
     state_id, progress_date, bill_status, status, count_year
    
     Params is a dict with the following parameter.
    :param state_id: (int)

    :return:
    A string have the SQL query for the task   
    """
    query = """
        select bill_id, state_id, progress_date, bill_status, status, count_year, row_number() over(
        partition by bill_id order by progress_date desc, index desc) as seq
        from (
        select state_id, state_abbreviation, bill_id, progress_date, bill_status, status, ROW_NUMBER() over() as index
        from ml_policy_class.bill_progress
        left join bill_state using(bill_id)
        left join catalogs.bill_status
        on status_id = bill_status
        join catalogs.states using(state_id)
        where state_id = {params['state_id']} ) as table1;
        """
    return query

#TODO: add WHERE clause to bill_final_text so that we only have bill type_id = 1

def bill_final_query(params):
    """
     To obtain SQL query to get final label and other attributes about bill, including bill_id, state_id, 
     progress_date, bill_status, status, diff, party_id, bill_type, introduced_date, introduced_body, status_new
    
    Params is a dict with the following parameter
    :param state_id: (int)
    
    :return:
    A string have the SQL query for the task   
    """
    query = f"""
    with 
    bill_final_status as (
        select * from(
        select *, row_number() over( 
        partition by bill_id order by progress_date desc, index desc) as seq 
        from ( 
            select state_id, state_abbreviation, bill_id, progress_date, bill_status, status, ROW_NUMBER() over() as index 
            from ml_policy_class.bill_progress  
            left join sketch.bill_state using(bill_id)
            left join catalogs.bill_status  
            on status_id = bill_status 
            join catalogs.states using(state_id) 
            where state_id = {params['state_id']} ) as table1
            join sketch.date_diff using(bill_id)
            join ml_policy_class.bill_sponsors bs using(bill_id)
            ) as table2
        where seq = 1), 
    bill_final_text as(
        select * from (
        select *, row_number() over(
            partition by bill_id
            order by doc_date desc) from ml_policy_class.bill_texts
        where type_id = 1) as table5
        where row_number = 1
    ),
    bill_sponsor_agg as(
    select bill_id, string_agg(sponsor_id::TEXT,', ' order by sponsor_id) as sponsors
    from ml_policy_class.bill_sponsors
    group by bill_id)
    select * from(
    select *, case 
        when diff <= 365 and bill_status = 6 then 'Failed'
        when diff <= 365 and bill_status = 4 then 'Passed'
        else 'Failed_Final' end as status_new
    from(
        select session_id,start_date,end_date,bill_id, state_id, bill_status, status, bill_description, doc, diff, party_id,bill_type,
        introduced_date, progress_date as final_date, introduced_body,sponsors, cast(to_char(introduced_date,'YYYY') as int) as year
        from bill_final_status
        join ml_policy_class.bills using(bill_id)
        join bill_final_text using(bill_id)
        join newdata.session_date_illinois using(session_id)
        join bill_sponsor_agg using(bill_id)
        where bill_type = 'B') as table4
        ) as table5 join acs.acs_illinois using(year)
        join (select bill_id, dem_senate_perc, rep_senate_perc, dem_house_perc, rep_house_perc,
        sponsor_count, rep_sponsor_count, sen_sponsor_count from sketch.one_year_sessions) as table6 using(bill_id);
        """
    return query


def bill_final_query_processed_text(params):
    """
     To obtain SQL query to get final label and other attributes about bill, including bill_id, state_id, 
     progress_date, bill_status, status, diff, party_id, bill_type, introduced_date, introduced_body, status_new
    
    Params is a dict with the following parameter
    :param state_id: (int)
    
    :return:
    A string have the SQL query for the task   
    """
    query = f"""
    with
    bill_final_status as (
        select * from(
        select *, row_number() over(
        partition by bill_id order by progress_date desc, index desc) as seq
        from (
            select state_id, state_abbreviation, bill_id, progress_date, bill_status, status, ROW_NUMBER() over() as index
            from ml_policy_class.bill_progress
            left join sketch.bill_state using(bill_id)
            left join catalogs.bill_status
            on status_id = bill_status
            join catalogs.states using(state_id)
            where state_id = {params['state_id']} ) as table1
            join sketch.date_diff using(bill_id)
            join ml_policy_class.bill_sponsors bs using(bill_id)
            ) as table2
        where seq = 1),
    bill_final_text as(
        select * from (
        select *, row_number() over(
            partition by bill_id
            order by doc_date desc) from ml_policy_class.bill_texts
        where type_id = 1) as table5
        where row_number = 1
    ),
    bill_sponsor_agg as(
        select bill_id, string_agg(sponsor_id::TEXT,', ' order by sponsor_id) as sponsors\
        from ml_policy_class.bill_sponsors
        group by bill_id)
    select * from(
    select *, case
        when diff <= 365 and bill_status = 6 then 'Failed'
        when diff <= 365 and bill_status = 4 then 'Passed'
        else 'Failed_Final' end as status_new
    from(
        select session_id,start_date,end_date,bill_id, state_id, bill_status, status, bill_description, bill_text, diff, party_id,bill_type,
        introduced_date, progress_date as final_date, introduced_body, sponsors, cast(to_char(introduced_date,'YYYY') as int) as year
        from bill_final_status
        join ml_policy_class.bills using(bill_id)
        join sketch.cleaned_bill_texts using(bill_id)
        join newdata.session_date_illinois using(session_id)
        join bill_sponsor_agg using(bill_id)
        where bill_type = 'B') as table4
        ) as table5 join acs.acs_illinois using(year)
        join (select bill_id, dem_senate_perc, rep_senate_perc, dem_house_perc, rep_house_perc,
        sponsor_count, rep_sponsor_count, sen_sponsor_count from sketch.one_year_sessions) as table6 using(bill_id);
    """
    return query


def bill_sponsor(params):
    """
     To obtain SQL query to get spaonsor_id, party_id, bill_id, and final label 
    
    Params is a dict with the following parameter
    :param state_id: (int)
    
    :return:
    A string have the SQL query for the task   
    """
    query = f"""
        select sponsor_id, bs.party_id, bill_id, status_new
        from (
        select *, case
        when diff <= 365 and bill_status = 6 then 'Failed'
        when diff <= 365 and bill_status = 4 then 'Passed'
        else 'Failed_Final'
        end as status_new
        from(
        select session_id,start_date,end_date,bill_id, state_id, progress_date, bill_status, status, bill_description,diff, party_id,bill_type, introduced_date, introduced_body from(
        select * from(
        select *, row_number() over(
        partition by bill_id order by progress_date desc, index desc) as seq
        from (
        select state_id, state_abbreviation, bill_id, progress_date, bill_status, status, ROW_NUMBER() over() as index
        from ml_policy_class.bill_progress
        left join bill_state using(bill_id)
        left join catalogs.bill_status
        on status_id = bill_status
        join catalogs.states using(state_id)
        where state_id = {params['state_id']} ) as table1 join date_diff using(bill_id)
        join ml_policy_class.bill_sponsors bs using(bill_id)
        ) as table2
        where seq = 1) as table3
        join ml_policy_class.bills using(bill_id)
        join ml_policy_class.bill_texts using(bill_id)
        join newdata.session_date_illinois using(session_id)
        ) as table4) as table_final
        join ml_policy_class.bill_sponsors bs using(bill_id);
        """
    return query


def read_cols_query(col_names, table_name, schema_name):
    """
    Obtain the query to read the columns from a table

    :param col_names: (list of strs) list of cols names to read
    :param table_name: (str) table name to read from
    :param schema_name: (str) schema name of the table

    :return:
        A string having the pSQL query for the task
    """
    cols = ", ".join(col_names)
    query = f"SELECT {cols} FROM {schema_name}.{table_name}"
    return query


def add_block_val_columns_query(table_name, schema_name, year_col, start_year=2009, end_year=2019,
                                split_list=(2,1,1,1), update_freq=2, type_val=0, return_num_blocks=False):
    """
    Creates four new columns, one for each iteration of block CV

    :param table_name: table to add columns to
    :param schema_name: schema containing table given in table_name
    :param year_col: column containing the date the bill was introduced
    :param type_val: specifies which typ eof splitting to be done
        - 0: rolling window splitting, moves the start and end dates
        - 1: increasing window splitting, moves the end date, but start date remains fixed
    
    :return: a string containing the pSQL query
    """
    # ALTER table {schema_name}.{table_name}  ALTER COLUMN {year_col} TYPE date using to_date({year_col}, 'YYYY-MM-DD');

    query = f""
    iter_num = 1
    while True:
        num_start_zeros = (iter_num-1)*update_freq
        if num_start_zeros+sum(split_list) > end_year-start_year+1:
            break
        query += f"ALTER TABLE {schema_name}.{table_name} add if not exists iter_{iter_num} int;\n"
        for idx, year in enumerate(range(start_year, end_year+1)):
            if idx < num_start_zeros:
                query += f"update {schema_name}.{table_name} set iter_{iter_num} = {type_val} WHERE EXTRACT(YEAR FROM {year_col}) = {year};\n"
            elif num_start_zeros <= idx < num_start_zeros+split_list[0]:
                query += f"update {schema_name}.{table_name} set iter_{iter_num} = 1 WHERE EXTRACT(YEAR FROM {year_col}) = {year};\n"
            elif num_start_zeros+split_list[0] <= idx < num_start_zeros+sum(split_list[0:2]):
                query += f"update {schema_name}.{table_name} set iter_{iter_num} = 2 WHERE EXTRACT(YEAR FROM {year_col}) = {year};\n"
            elif num_start_zeros+sum(split_list[0:2]) <= idx < num_start_zeros+sum(split_list[0:3]):
                query += f"update {schema_name}.{table_name} set iter_{iter_num} = 3 WHERE EXTRACT(YEAR FROM {year_col}) = {year};\n"
            elif num_start_zeros+sum(split_list[0:3]) <= idx < num_start_zeros+sum(split_list):
                query += f"update {schema_name}.{table_name} set iter_{iter_num} = 4 WHERE EXTRACT(YEAR FROM {year_col}) = {year};\n"
            else:
                query += f"update {schema_name}.{table_name} set iter_{iter_num} = 0 WHERE EXTRACT(YEAR FROM {year_col}) = {year};\n"
        iter_num += 1
    
    if return_num_blocks:
        return query, iter_num-1
    return query


def put_condition_on_query(query, condition):
    """
    To generate a query that wraps the output of some query with some given conditions

    :param query: (str) base query
    :param condition: (str) where condition
    :return:
        (str) new query
    """
    query = query.strip()
    if query[-1] == ";":
        query = query[:-1]
    return f"SELECT * FROM ({query}) as cond_table {condition}"
