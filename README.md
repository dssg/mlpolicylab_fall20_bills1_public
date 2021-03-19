# Bill Passage Team 1

## Code Pipeline Structure

## Requirements

- Python 3.6
- PostgreSQL

#### Python Libaries   

- `sklearn`
- `numpy`
- `keras`
- `scipy`
- `pickle`
- `psycopg2`
- `mmh3`
- `seaborn`
- `yaml`
- `pandas`
- `joblib`

## Usage: 

1. `config.json` can be used to specify the various parameters you want to use. 
This includes the following details.
    - state name for the state you are analysing
    - `model_type`: the model that you want to use for classification
    - `model_info`: defines the setup and hyperparams to be used for evey model type
    - `query`: Contains the details about what queries to run on the database in order. 
    Also takes in the schema name and table name where we want to store the output of those queries
    - `transform`: Contains a list of transformations that need to applied on each column 
    of the final database table from the queries section, and the table name for where you want
    to save the resulting table at. 
    - `split`: Contains the arguments for the temporal splitting
        * `split_type`: 0 for fixed window length, and 1 for expanding windows. 
        * `update_freq`: how much forward in time do we move with each split
        * `split_list`: a list of length 4, having the time for training, training buffer, validation, validation buffer. 
        * `num_temporal_blocks`: How many of the temporal blocks to be used for training/validation. 
        (E.g. If you say 1, then we will only do the analysis on the first temporal split.
         This parameter helps while debugging.)
    - `feature_type`: the type of feature vector to be used
    - `feature_info`: contains the details for all the different feature types
        * `index_col_name`: name of the column holding the indexing value, for our case the bill id
        * `label_col_name`: name of the column holding the prediction label data
        * `feature_ops`: contains the column name along with the list of operations to be performed
        on the column to obtain the feature from the data in the column. 
        If a column is used more than once, then you can append `::i` after the column name.
        The operation you can perform are the functions in the `features.py` file. You can also 
        specify the various parameters taken in by the operations as a list. 
        You can read through the parameters taken, and the operation themselves in the docstrings
        in the `features.py` file. 
    - `feature_names`: human readable names to be used as the interpretable output 
        (if different from column names)
    - `label_type`: the type of labels to create from the label information
    - `label_info`: info to be used for the various label types (contains size only for now)
    - `train_params`: specifies the training parameters for neural network training 
    - `model_path`: path where we store the model while training, or the path from where we
    load the model while prediction
    - `evaluate_params`:  specifies what evaluation metrics and parameters to use
     and other evaluation analysis to be performed such as feature importances etc.
            
2. `run.py` is the driver code that excutes all the code.
 It can take in the following parameters to control what task you want to performs
    - `--db_ops`: to run the query operations
    - `--transform`: to run the transform operations
    - `--split`: to create temporal splits
    - `--train`: to train a model of the pre-created temporal splits, 
        and save the models at `model_path`.
    - `--predict`: to predict using the model at `model_path` on the validation splits
    - `--evaluate`: to predict and evaluate the model at `model_path` on the validation splits.

   Usage: 
   ```
        python run.py <parameters>
   ```
   
3. `run_model_grid.py` contains the code to run the model grid over a range of model types
and a range over multiple hyperparameters for each model type. It uses `joblib` to run paralle 
processes to execute multiple configurations in parallel. 
    - `feature_types`: specifies which feature types need to be run
    - `model_types`: specifies which model types need to be run
    - `model_grid_params`: contains the range of hyperparameters to run for each model type
    - `model_folder`: the target location where all the results from the model grid are stored
    
    Usage: 
   ```
        python run_model_grid.py 
   ```
## ACLU Data Notes

American Civil Liberties Union's (ACLU) mission is protecting civil liberties of all citizens through advocacy efforts against legislation that infringe on civil liberties, and ensuring that necessary statutes are in place to protect civil liberties.  To that end, one of the most important aspects of their work is monitoring legislation passed into law at national level and state level. Currently, this process of bill monitoring entails signifcant manual effort. Therefore, ACLU's state affiliates either, spend their time reading through all bills to identify legilation that they should oppose (or support), or end up having to pick and choose bills they read using their domain knowledge (e.g. picking bills sponsored by certain legislators/committees). The former takes away their time that could be spent on devising interventions (e.g. drafting lawsuits) and the latter can result in missing important legislation pieces.  

The goal of this project is to help ACLU better target their resources by identifying bills that are likely to be passed into law. The data for this project comes from LegiScan, an organization that collects and diseminates data about national and state level legislation and the legislative processes. LegiScan provides an API to acquire continuous updates to the legislative sessions for all 52 states and districts/territories of the US. Therefore, this project uses data that is publicly available. The provided database contains legilative session and bill data for the last ~10 years. The initial raw data was provided by LegiScan as a collection of JSON files. A cleaned schema from the JSON files was created, and includes the following infomation:
- *State legislative session information*: Information about legislative sessions for each session is available in `.sessions` with a unique session identifier `session_id`. Note that legislative sessions can be either periodic regular sessions or special sessions convened by the Governor. The boolean column `special` indicates whether the session is a special session or not. 

- *Bill information*: Data about bills introduced in each session is provided in `ml_policy_class.bills`. Each bill has a unique identifier `bill_id`. Information about the type of the bill, the session it belongs to, bill subjects, the date and the chamber/ body (House/Senate) where the bill was introduced are provided. The `url` field contains the link to the LegiScan webpage that shows the bill, if needed. 

- *Bill text information* : Bills typically get amended and go through different versions when it moves through the legislative process. The table `ml_policy_class.bill_texts` contains text and information about different versions of a bill. Each version has a unique identifier named `doc_id`. Each bill version/text has a type to indicate which form the bill is in (e.g. is it as it was introduced? amendments adopted?) and is indicated by the `type_id`. The title of the bill and a summary of the bill is given in the fields `bill_title` and `bill_description`, respectively. The `url` fields point to the LegiScan page and the state's webpage for the bill text.

- *People information*: Information about congress members (both senate and house) related to each session is available in `ml_policy_class.session_people`. Each member has a unique identifier `person_id` and information such as their party affiliation, role is provided. Further, several identifiers for external data sources such as _follow the money_, _votesmart_, and _opensecrets_ are available as well. In addition, `ml_policy_class.bill_sponsors` table contains information about people who sponsored each bill.

- *Voting information*: Voting information for roll calls of bills is available in the table `ml_policy_class.bill_votes`. Each roll call has a unique identifier `vote_id` and contains a summary of the votes. Note that if individual person votes for each roll call can be obtained through the LegiScan API (API info provided below) using the roll call id. 

- *Bill event information*: Events that a bill goes through and the status progress of a bill are available in the tables `ml_policy_class.bill_events` and `ml_policy_class.bill_progress` respectively. 

- *Amendment information*: Information about amendments made to the bill from the house and the seante is available in the table `ml_policy_class.bill_amendments`. The links to find the text for the amendment (from the Legiscan website and the State website) are given in the columns `legiscan_url, state_url`. 

- *catalog tables*: In addition the the `ml_policy_class` schema, a `catalogs` schema that contains mappings from LegiScan identifiers to descriptions. For instance, identifiers such as party ids, bill text type ids, status codes are mapped to their descriptions in this schema.

- *Legiscan API for additional data*: LegiScan API (https://legiscan.com/legiscan) can be used to acquire more data about bills if necessary and the user manual for the API can be found here: https://legiscan.com/gaits/documentation/legiscan. 

We've added some information about the dates of regular legislative sessions in `regular_session_dates.csv` in this repo that you may want to load into your database as well. Note that this file covers regular sessions (but not special sessions) and may be missing some dates (online sources like ballotopedia may help fill these out). Additionally, some states (such as CA) sometimes start their sessions in the December before the year associated with the legislative session, so you may want to look out for potential inconsistencies.

