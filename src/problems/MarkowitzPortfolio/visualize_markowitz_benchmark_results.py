# -*- coding: utf-8 -*-
"""
Created on 05.09.2024

@author: stopfer
"""
import os
import json
import pdb
import numpy as np
from matplotlib import pyplot as plt



def visualize_markowitz_benchmark_results(opt_results: list, result_folder: str, only_best_sols: bool = True):
    ''' 
    function that visualizes the found feasible solutions of the opt_method runs. 
    It plots their return vs volatility inside of a coordinate system with different 
    colours for the opt_methods
    '''
    
    # %% create a dictionary with the returns and volatilities for the opt_methods
    return_volatility_combinations = {}
    for [problem_config, solver_config, _], report_obj, feas_sol_dict in opt_results:
        used_opt_method = report_obj.solve_method
        used_device = report_obj.solve_method_device
        opt_goal = report_obj.problem_config['opt_goal']
        calc_name = used_opt_method + '_' + used_device
        return_volatility_combinations[calc_name] = {}
        return_volatility_combinations[calc_name]['expected_returns'] = []
        return_volatility_combinations[calc_name]['expected_volatilities'] = []
        for bitstr, bitstr_dict in feas_sol_dict.items():
            expected_ret = bitstr_dict['expected_return']
            expected_vola = bitstr_dict['expected_volatility']
            return_volatility_combinations[calc_name]['expected_returns'].append(expected_ret)
            return_volatility_combinations[calc_name]['expected_volatilities'].append(expected_vola)
        
        if only_best_sols == True:   
            # keep only the best solution
            if opt_goal == "max_return_with_constrained_volatility":                    
                if return_volatility_combinations[calc_name]['expected_returns'] != []:
                    index_max_return = np.argmax(return_volatility_combinations[calc_name]['expected_returns'])
                    return_volatility_combinations[calc_name]['expected_returns'] = return_volatility_combinations[calc_name]['expected_returns'][index_max_return]
                    return_volatility_combinations[calc_name]['expected_volatilities'] = return_volatility_combinations[calc_name]['expected_volatilities'][index_max_return]
            elif opt_goal == "min_volatility_with_constrained_return":
                if return_volatility_combinations[calc_name]['expected_volatilities'] != []:
                    index_min_vola = np.argmin(return_volatility_combinations[calc_name]['expected_volatilities'])
                    return_volatility_combinations[calc_name]['expected_returns'] = return_volatility_combinations[calc_name]['expected_returns'][index_min_vola]
                    return_volatility_combinations[calc_name]['expected_volatilities'] = return_volatility_combinations[calc_name]['expected_volatilities'][index_min_vola]
                    
    # %% create the plot
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.xlabel('volatility')
    plt.ylabel('return')
    if only_best_sols == False:
        plt.title('solutions for each solve_method')
    else:
        plt.title('best solution for each solve_method')
    colours = ['red', 'blue', 'green', 'yellow', 'black', 'orange', 'brown']
    counter = 0
    for opt_method, opt_method_dict in return_volatility_combinations.items():
        plt.scatter(x = opt_method_dict['expected_volatilities'], 
                    y = opt_method_dict['expected_returns'], 
                    color = colours[counter], 
                    marker = 'o',       # draw points in the graphic
                    facecolors='none',  # don't fill out the points
                    label = opt_method,
                    s = 50 + counter*40,# radius of the points
                    alpha=1             # factor for transparency of the points
                    )
        counter+=1
    plt.legend(loc='upper right')
    filename = 'MarkowitzPortfolio_feasible_solutions_found_all' if only_best_sols==False else 'MarkowitzPortfolio_feasible_solutions_found_only_best_ones'
    plt.savefig(os.path.join(result_folder, filename+".png"), dpi=200)            
    plt.clf()
    plt.close()
            
    return return_volatility_combinations
            