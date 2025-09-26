# -*- coding: utf-8 -*-,
"""
Created on Thu Aug 15 14:16:58 2024

@author: stopfer
"""
import os
import pdb
import json
import random
import numpy as np
import pandas as pd

from src.utils.error_utils import CustomizedError

def create_random_markowitz_problem_config_files(number_of_new_files: int,
                                         number_of_assets: int,
                                         base_config_file_path: str,
                                         stock_market: str,
                                         opt_goal: str):
    '''
    creates a certain number of config files for the Markowitz portfolio optimization 
    problem with a certain opt goal from a certain stock market
    '''
    # %% read base_config_file
    if os.path.exists(base_config_file_path):
        with open(base_config_file_path) as json_file:
            try:
                markowitz_config_data = json.load(json_file)
            except:
                print("ATTENTION: decoding Error in config.json-file; check that the json is formatted like a dictionary. Often errors occur because of wrong comma placements")
                raise CustomizedError("DecodingError in config.json-file; check the following: \n \
                                        - has to be formatted like a dictionary \n \
                                        - comma placements between key-value-pairs \n \
                                        - no comma after the last key-value-pair \n \
                                        - boolean variables have to written small (->e.g. false instead of False) \n \
                                        - no tuples")
    
    for _ in range(number_of_new_files):
        for asset_number in number_of_assets:
            asset_number = 500
            # %% take some random assets from the stock market
            current_file_directory = os.path.dirname(os.path.abspath(__file__))
            asset_returns_df = pd.read_csv(os.path.join(current_file_directory, f'{stock_market}_annual_returns.csv'), delimiter='\t')
            covariance_matrix_df = pd.read_csv(os.path.join(current_file_directory, f'{stock_market}_annualized_covariance_matrix.csv'), delimiter='\t')
            all_asset_names = asset_returns_df.columns.to_list()
            chosen_assets = random.sample(all_asset_names, min(len(all_asset_names), asset_number))
            asset_returns = asset_returns_df[chosen_assets]
            asset_covariances = covariance_matrix_df.loc[chosen_assets, chosen_assets]
            
            # %% write to new config file
            markowitz_config_data['problem_config']['asset_returns'] = []
            markowitz_config_data['problem_config']['asset_limits'] = []
            markowitz_config_data['problem_config']['asset_covariance_matrix'] = [[]]
            markowitz_config_data['problem_config'][f'choose_{stock_market}_assets'] = True
            markowitz_config_data['problem_config'][f'{stock_market}_assets'] = chosen_assets
            markowitz_config_data['problem_config']['opt_goal'] = opt_goal
            markowitz_config_data['problem_config']['min_return'] = np.percentile(asset_returns, 70)
            if opt_goal == "max_return_with_constrained_volatility":
                markowitz_config_data['problem_config']['min_return'] = 1.0 # irrelevant
                markowitz_config_data['problem_config']['max_volatility'] = np.percentile([calc_volatility(get_random_portfolio(len(chosen_assets)), asset_covariances) for i in range(100)], 30)
                markowitz_config_data['problem_config']['delta_risk_aversion'] = 1.0 # irrelevant
            elif opt_goal == "min_volatility_with_constrained_return":
                markowitz_config_data['problem_config']['min_return'] = np.percentile(asset_returns, 70)
                markowitz_config_data['problem_config']['max_volatility'] = 1.0 # irrelevant
                markowitz_config_data['problem_config']['delta_risk_aversion'] = 1.0 # irrelevant
            elif opt_goal == "max_return_min_volatility":
                markowitz_config_data['problem_config']['min_return'] = 1.0 # irrelevant
                markowitz_config_data['problem_config']['max_volatility'] = 1.0 # irrelevant
                avg_abs_magnitude_vola = np.mean([calc_volatility(get_random_portfolio(len(chosen_assets)), asset_covariances) for i in range(100)])
                avg_abs_magnitude_return = np.mean([calc_return(get_random_portfolio(len(chosen_assets)), asset_returns) for i in range(100)])
                # aavola=[calc_volatility(get_random_portfolio(len(chosen_assets)), asset_covariances) for i in range(100)]
                # aaret=[calc_return(get_random_portfolio(len(chosen_assets)), asset_returns) for i in range(100)]
                # pdb.set_trace()
                fraction_ret_vola = avg_abs_magnitude_return / avg_abs_magnitude_vola
                markowitz_config_data['problem_config']['delta_risk_aversion'] = avg_abs_magnitude_return / avg_abs_magnitude_vola
            else:
                raise Exception(f"unknown opt_goal {opt_goal}")
            
            # %% save new config file
            path = os.path.join(os.path.dirname(base_config_file_path), f"from_{stock_market}", f"{asset_number}assets")
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                
            opt_goal_str = 'minvola' if opt_goal == 'min_volatility_with_constrained_return' else 'maxret' if opt_goal == 'max_return_with_constrained_volatility' else 'multiobj'
            filename = f"random_MarkowitzPortfolio_from{stock_market}_{opt_goal_str}_{asset_number}assets_"
            counter=0
            while os.path.exists(os.path.join(path, filename+str(counter)+".json")):
                counter += 1                
            json_file_path = os.path.join(path, filename+str(counter)+".json")
            
            with open(json_file_path, "w") as file:
                json.dump(markowitz_config_data, file, indent=4) # indent=4 adds for new keys of the dict 4 spaces -> improves readability of the json

def calc_volatility(asset_weights: list, asset_cov_matrix: pd.DataFrame):
    weights = np.array(asset_weights)
    covariance = asset_cov_matrix.values
    return float(np.dot(weights, np.dot(covariance, weights)))


def calc_return(asset_weights: list, asset_returns: pd.DataFrame):
    weights = np.array(asset_weights)
    returns = np.array(asset_returns.iloc[0,:])
    return float(np.dot(weights, returns))


def get_random_portfolio(number_of_assets: int):
    random_values = np.random.uniform(0, 1, number_of_assets-1)
    random_values = np.append(random_values, [0, 1])
    sorted_values = np.sort(random_values)
    portfolio_weights = list(np.diff(sorted_values))
    return portfolio_weights


def delete_config_files(path_root: str, config_file_substring: str):
    ''' deletes all config files with a substring in its names from a root path '''
    assert os.path.exists(path_root)
    for root, dirs, files in os.walk(path_root):
        for file in files:
            if config_file_substring in file:
                file_path = os.path.join(root, file)
                os.remove(file_path)


def change_attribute_in_config_files(path_root: str, attribute_name: str, new_attribute_value: any):
    ''' changes attribute new_attribute_value (->structured like "path/to/attribute/inside/json/dict" 
    to a new value for all json files in root_path '''
    assert os.path.exists(path_root)
    attribute_name_dict_path = attribute_name.split('/')
    for root, dirs, files in os.walk(path_root):
        for filename in files:
            if ".json" in filename:
                file_path = os.path.join(root, filename)
                with open(file_path, "r") as file:
                    config_data = json.load(file)
                    try:
                        if len(attribute_name_dict_path) == 1:
                            pathpart1 = attribute_name_dict_path[0]
                            config_data[pathpart1] = new_attribute_value 
                        elif len(attribute_name_dict_path) == 2:
                            pathpart1 = attribute_name_dict_path[0]
                            pathpart2 = attribute_name_dict_path[1]
                            config_data[pathpart1][pathpart2] = new_attribute_value
                        else:
                            raise ValueError
                        with open(file_path, "w") as file:
                            json.dump(config_data, file, indent=4) # indent=4 adds for new keys of the dict 4 spaces -> improves readability of the json
                    except ValueError:
                        raise Exception('attribute_name_dict_path was only configured for length 1 and 2')
                
                
if __name__ == "__main__":
    
    # configs_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "config"), "config_files")
    # delete_config_files(path_root = configs_path, 
    #                     config_file_substring = 'fromnasdaq_multiobj')
    
    # markowitz_configs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "config", "config_files", "problem_configs", "MarkowitzPortfolio")
    # change_attribute_in_config_files(
    #     path_root = markowitz_configs_path,
    #     attribute_name = 'problem_mapping_config/qubo_penalty_factor',
    #     new_attribute_value = 1000
    #     )
    
    create_random_markowitz_problem_config_files(
        number_of_new_files = 10,
        number_of_assets = \
                           [3, 5, 8] + 
                           list(range(10,100,10)) +
                           list(range(100,300,50)) +
                           list(range(300,1000,100)) +
                           [1000, 2000], 
        # number_of_assets = [15,25],
        base_config_file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) + 
                                "/config/config_files/problem_configs/MarkowitzPortfolio/markowitz_3assets_minvola.json",
        stock_market = "nasdaq",
        # opt_goal = "min_volatility_with_constrained_return",
        # opt_goal = "max_return_with_constrained_volatility"
        opt_goal = "max_return_min_volatility"
        )