import os, time
import sys
import argparse
import json, pickle as pkl
from models import *
import queries
from sklearn import tree
from db_ops import run_sql_query, open_db_connection
from processing import preprocess_data, create_temporal_blocks
from features import obtain_features_and_labels, create_text_transformations
from metrics import Metric, plot_PR_k, plot_score_distribution, plot_feature_importance


parser = argparse.ArgumentParser()
parser.add_argument("--db_ops", action='store_true', help="To combine the data from the multiple tables into one")
parser.add_argument("--transform", action='store_true', help="To run transformations on the data table")
parser.add_argument("--split", action="store_true", help="To add temporal block validation columns/values")
parser.add_argument("--train", action='store_true', help="To run feature creation on the data table")
parser.add_argument("--predict", action="store_true", help="To generate predictions on the data table")
parser.add_argument("--evaluate", action='store_true', help="To evaluate the prediction on the data table")
parser.add_argument('--secrets', default="secrets.yaml", help="YAML file holding the DB credentials")


def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))
        return ret
    return wrap
    
#@timing
def main(params=None, raw_args=sys.argv[1:], verbose=True):
    args = parser.parse_args(raw_args)

    if params is None:
        # load parameters from config file
        with open("config.json", "r") as f:
            params = json.load(f)

    model_type = params['model_type']

    conn = open_db_connection(args.secrets, verbose=verbose)
    query_dict = params['query']
    transform_dict = params['transform']
    split_dict = params['split']
    feature_dict = params['feature_info'][params['feature_type']]
    label_info = params['label_info'][params['label_type']]
    eval_dict = params['evaluate_params']
    
    if args.evaluate:
        args.predict = True

    load_data = (args.train or args.predict)

    if args.db_ops:
        # If we're predicting, use the test set. Otherwise, train set.
        query_params = query_dict['test'] if args.predict else query_dict['train']
        # Loop through different queries in query_dict
        # Store the query name and its parameters in query
        # Run the query on the specified table
        for idx, query_name in enumerate(query_dict['name']):
            query = getattr(queries, query_name)(query_params)
            run_sql_query(conn, query, schema_name=query_dict['schema_name'][idx], table_name=query_dict['table_name'][idx])

    if args.transform:
        # preprocess operations (cleaning etc.)
        preprocess_data(conn, transform_dict['ops'], input_schema_name=query_dict['schema_name'][-1],
                        input_table_name=query_dict['table_name'][-1], output_schema_name=transform_dict['schema_name'],
                        output_table_name=transform_dict['table_name'])


    if args.split:
        # run add_block_val_columns to create blocking temporal val columns in the desired table
        split_list = [int(i) for i in split_dict['split_list']]
        create_temporal_blocks(conn, schema_name=transform_dict['schema_name'],
                               table_name=transform_dict['table_name'], year_col=transform_dict['year_col'],
                               start_year=int(query_dict['train']['start_year']), end_year=int(query_dict['train']['end_year']),
                               update_freq=split_dict['update_freq'], split_list=split_list, type_val=split_dict['split_type'])

    if load_data:
        # we only do batching at DataBase level if training,
        # assuming test/prediction data would be small enough to sit in memory

        # iterate through the temporal blocks
        num_temporal_blocks = split_dict["num_temporal_blocks"]
        TRAINING_IDX = 1
        VALIDATION_IDX = 3

        # create text vocabulary indexing from ALL the training data
        # Alternatively, we can create this from some other much larger corpus of data as well 
        # load vocab if exist
        text_vocab_path = os.path.join(os.path.dirname(os.path.dirname(params['model_path'])), "text_vocab.json")
        if os.path.exists(text_vocab_path):
            # load the text vocabulary from the saved file
            with open(text_vocab_path, "r") as f:  
                text_vocab = json.load(f) 
        else:
            vectorizer_feature_ops = {}
            for col_name in feature_dict['feature_ops']:
                if "::" in col_name:
                    col_name = col_name.split("::")[0]
                if col_name in vectorizer_feature_ops:
                    continue
                for op in feature_dict['feature_ops'][col_name]:
                    if op in ['bow', 'lda', 'tfidf', 'bert']:
                        vectorizer_feature_ops[col_name] = {}
                        vectorizer_feature_ops[col_name]['bow'] = []
                        break           
            text_vectorizer = create_text_transformations(conn, vectorizer_feature_ops, schema_name=transform_dict['schema_name'],
                                                         table_name=transform_dict['table_name'])
            text_vocab = {}
            for col_name in text_vectorizer:
                for op in text_vectorizer[col_name]:
                    text_vocab[col_name] = text_vectorizer[col_name][op].vocabulary_
            with open(text_vocab_path, "w") as f:  
                json.dump(text_vocab, f) 

        results = {}
        # training and validation from each of the temporal blocks
        for temporal_block_id in range(1,num_temporal_blocks+1):
            ##debug
#             if temporal_block_id in [1,2,3]:
#                 continue
            ##end debug
            
            if verbose:
                print(f"\nRUNNING FOR TEMPORAL BLOCK {temporal_block_id}/{num_temporal_blocks}")
            train_params = params['train_params']
            label_info = {"label_type": params['label_type'], "label_col_name": feature_dict["label_col_name"]}
            return_iterator = train_params['return_iterator']
            
            if verbose:
                print("Obtaining Text Transformations")
            text_transforms = create_text_transformations(conn, feature_dict['feature_ops'], schema_name=transform_dict['schema_name'],
                                                          table_name=transform_dict['table_name'], iter_num=temporal_block_id, 
                                                          iter_val=TRAINING_IDX,text_vocabulary=text_vocab)
            if verbose:
                print("Obtaining training features")
            training_data, applied_feature_ops = obtain_features_and_labels(conn, feature_dict['feature_ops'], label_info,
                                              return_iterator=return_iterator, batch_size=train_params['batch_size'],
                                              schema_name=transform_dict['schema_name'], table_name=transform_dict['table_name'],
                                              iter_num=temporal_block_id, iter_val=TRAINING_IDX, text_transforms=text_transforms,
                                              index_col_name=feature_dict['index_col_name'], verbose=False)
            if verbose:
                print("Obtaining validation features")
            validation_data, _ = obtain_features_and_labels(conn, applied_feature_ops, label_info,
                                              return_iterator=False, schema_name=transform_dict['schema_name'], 
                                              table_name=transform_dict['table_name'], iter_num=temporal_block_id, 
                                              iter_val=VALIDATION_IDX, text_transforms=text_transforms,
                                              index_col_name=feature_dict['index_col_name'], verbose=False)

            if verbose:
                print("Training data:")
                print(training_data[0].shape, training_data[1].shape)
                print(f"Bills passed: {np.sum(training_data[1][:, 1])} ({np.sum(training_data[1][:,1])/training_data[1].shape[0]*100}%)")
                print("Validation data:")
                print(validation_data[0].shape, validation_data[1].shape)
                print(f"Bills passed: {np.sum(validation_data[1][:,1])} ({np.sum(validation_data[1][:,1])/validation_data[1].shape[0]*100}%)")

            assert model_type in ["BinaryLabelNNClassifier", "MultiLabelNNClassifier", "SVMClassifier",
                                  "RFClassifier", "DTClassifier", "LogRegClassifier","Baseline"], \
                        "model_type must be one out of ['BinaryLabelNNClassifier', 'MultiLabelNNClassifier', \
                        'SVMClassifier', 'RFClassifier', 'DTClassifier', 'LogRegClassifier','Baseline']"
            if model_type == "BinaryLabelNNClassifier":
                input_dim = training_data[0].shape[1]
                model = BinaryLabelNNClassifier(input_dim=input_dim, show=True, **params["model_info"][model_type])
            elif model_type == "MultiLabelNNClassifier":
                input_dim = training_data[0].shape[1]
                model = MultiLabelNNClassifier(input_dim=input_dim, num_classes=params['label_info'][params['label_type']]['size'],
                                               show=True, **params["model_info"][model_type])
            elif model_type == "SVMClassifier":
                model = SVMClassifier(**params["model_info"][model_type])
            elif model_type == "RFClassifier":
                model = RFClassifier(**params["model_info"][model_type])
            elif model_type == "DTClassifier":
                model = DTClassifier(**params["model_info"][model_type])
            elif model_type == "LogRegClassifier":
                model = LogRegClassifier(**params["model_info"][model_type])
            else:
                model = Baseline(conn, iter_num=temporal_block_id, iter_val=1)

            if args.train:
                if verbose:
                    print("Started Training")
                model.train(training_data, train_params, validation_data)
                model_path = os.path.join(params['model_path'], f"temporal_block_{temporal_block_id}")
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                model.save(model_path)

            if args.predict:
                model.load(os.path.join(params['model_path'], f"temporal_block_{temporal_block_id}"))
                train_predictions = model.predict_score(training_data[0], params['predict_params'])[:,1]
                validation_predictions = model.predict_score(validation_data[0], params['predict_params'])[:,1]
                # predictions = model.predict(validation_data[0], params['predict_params'])
                # print(predictions)

            if args.evaluate:
                # currently evaluation only works for one-hot labels
                # can be extended to 0-1 labels later quite simply
                plots_path = os.path.join(params['model_path'], "plots")
                if not os.path.exists(plots_path):
                    os.makedirs(plots_path)
                if verbose:
                    print("Evaluation Metrics:")
                metrics = []
                results[temporal_block_id] = {}
                metric_names = eval_dict['metrics']   
                
                # get the feature names for visualising feature importances
                if eval_dict['feature_importances']['use']:
                    feature_names = []
                    for f_name in feature_dict['feature_ops']:
                        if f_name in ['bill_text', 'bill_description']:
                            words = text_vocab[f_name].keys()
                            indices = text_vocab[f_name].values()
                            order = np.argsort(indices)
                            sorted_words = [f_name+"::"+words[i] for i in order]
                            feature_names += sorted_words
                        elif f_name in feature_dict["feature_names"]:
                            f_label = feature_dict["feature_names"][f_name]
                            if isinstance(f_label, list):
                                feature_names += f_label
                            else:    
                                feature_names.append(f_label)
                        else:
                            feature_names.append(f_name)
                    
                    top_n_features_names, top_n_features_ids = plot_feature_importance(model, feature_names, top_n=eval_dict['feature_importances']['top_n'], file_path=os.path.join(plots_path, f"feature_importance_{temporal_block_id}.png"))

#                 validation_data = training_data
#                 validation_predictions = train_predictions
                for metric_name in metric_names:
                    metric = Metric(metric_name, average=None, threshold=eval_dict['threshold'],
                                    k=eval_dict['top_k_fraction'])
                    # y_pred_one_hot = np.zeros(validation_data[1].shape)
                    # y_pred_one_hot[[np.arange(validation_data[1].shape[0]), predictions]] = 1
                    if args.train:
                        train_score, _ = metric.evaluate_top_k(y_true=np.argmax(training_data[1], axis=-1), y_pred=train_predictions)
                    validation_score, val_pred_binary = metric.evaluate_top_k(y_true=np.argmax(validation_data[1], axis=-1), y_pred=validation_predictions)
                    if verbose:
                        if args.train:
                            print(f"{metric_name}:: Train: {train_score}, Validation: {validation_score}")
                        else:
                            print(f"{metric_name}:: Validation: {validation_score}")
                    if args.train:
                        results[temporal_block_id][metric_name] = {"train": train_score, "validation": validation_score}
                    else:
                        results[temporal_block_id][metric_name] = {"validation": validation_score}
                    if eval_dict['cross_tabs']['use']:
                        if eval_dict['feature_importances']['use'] and not eval_dict['cross_tabs']['feature_indices']:
                            cross_tab_features = top_n_features_ids
                        else:
                            cross_tab_features = eval_dict['cross_tabs']['feature_indices']
                        if args.train:
                            train_top_vals, train_bottom_vals = metric.cross_tabs_at_k(X=training_data[0], y_pred=train_predictions,
                                                                                   feature_indices=cross_tab_features)
                        validation_top_vals, validation_bottom_vals = metric.cross_tabs_at_k(X=validation_data[0],
                                                                                             y_pred=validation_predictions,
                                                                                             feature_indices=cross_tab_features)
                        if verbose:
                            if eval_dict['feature_importances']['use'] and not eval_dict['cross_tabs']['feature_indices']:
                                print(f"Top features: {top_n_features_names}")
                            if args.train:
                                print(f"Training: {metric_name} Cross Tabs: \n Top vals: {train_top_vals} \n Bottom Vals: {train_bottom_vals}")
                            print(f"Validation: {metric_name} Cross Tabs: \n Top vals: {validation_top_vals} \n Bottom Vals: {validation_bottom_vals}")
                
                if eval_dict['bias_analysis']['use']:
                    with open("bid_groups.pkl","rb") as f:
                        bill_ids_by_group = pkl.load(f)
                    results[temporal_block_id]['bias_analysis'] = {}
                    for group in eval_dict['bias_analysis']['groups']:
                        group_bill_ids = bill_ids_by_group[group]
                        idx = np.isin(validation_data[2], group_bill_ids) # this step takes long time
                        results[temporal_block_id]['bias_analysis'][group] = {}
                        for metric_name in eval_dict['bias_analysis']['metrics']:
                            metric = Metric(metric_name)
                            score = metric.evaluate(y_true=np.argmax(validation_data[1][idx], axis=-1), y_pred=val_pred_binary[idx])
                            results[temporal_block_id]['bias_analysis'][group][metric_name] = score
                        
                plot_PR_k(y_true=np.argmax(validation_data[1], axis=-1), y_pred=validation_predictions, 
                          file_path=os.path.join(plots_path, f"prk_{temporal_block_id}.png"))
                
                plot_score_distribution(y_pred=validation_predictions, 
                                        file_path=os.path.join(plots_path, f"distribution_{temporal_block_id}.png"))
                    
        return results


if __name__ == "__main__":
    main()