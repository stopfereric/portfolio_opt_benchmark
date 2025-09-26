# -*- coding: utf-8 -*-
"""
Created on 07.03.2024

@author: Eric Stopfer
"""
import pdb
import time
import os
import random
import copy
import numpy as np

from config.config import Config
from src.report import Report
from src.utils.error_utils import CustomizedError
from src.utils.qc_utils import calculate_expectation_value_random_sampling_qubo
from src.utils.solver_result_utils import analyse_feasibility_of_solver_solutions, get_all_feasible_solutions, write_kpis_to_report
from src.utils.visualisation_utils import create_barplot_obj_values_with_number_of_violated_constraints


class Opt_Heuristics:
    '''
    ---- Definition from https://en.wikipedia.org/wiki/Heuristic -----
    
    A heuristic is any approach to problem solving that employs a pragmatic method that is 
    not fully optimized, perfected, or rationalized, but is nevertheless "good enough" as an 
    approximation or attribute substitution. Where finding an optimal solution 
    is impossible or impractical, heuristic methods can be used to speed up the process
    of finding a satisfactory solution
    '''
    
    def __init__(self, config: Config, report: Report):
        """
        Function that creates an Greedy-Algorithm solver instance
        """
        config.logger.info("we are using the following solving method: %s" %config.solve_method)
        
        
    def run(self, 
            opt_problem, 
            config: Config, 
            report: Report
            ) -> dict:
        '''
        Function that runs the optimization heuristic for a problem
        '''
        self.qubo_dict = opt_problem.problem_mapping[0]
        self.qubo = opt_problem.problem_mapping[1]
        self.mip_for_feasibility_analysis = opt_problem.problem_mapping[2]
        
        config.logger.info("the solving with a heuristic starts right now")
        start_time = time.time()
        
        # %% read the necessary attributes from the configuration
        try:
            opt_problem_type = config.problem_name
            number_of_samples = config.solve_method_config['number_of_samples']
            max_calculation_time_exists = config.solve_method_config['max_calculation_time_exists']
            max_calculation_time = config.solve_method_config['max_calculation_time_in_min']
            heuristic_name = config.solve_method_config['heuristic_name']
        except:
            raise CustomizedError(error_message="wrong or missing configuration of the Opt_Heuristics. check in config.py and config.json")
        
        if opt_problem_type == 'MarkowitzPortfolio':
            samples = self.run_markowitzportfolio_heuristic(opt_problem, 
                                                          number_of_samples, 
                                                          heuristic_name,
                                                          max_calculation_time_exists, 
                                                          max_calculation_time, 
                                                          config, 
                                                          report)
            
        elif opt_problem_type == 'TSP':
            samples = self.run_tsp_heuristic(opt_problem, 
                                            number_of_samples, 
                                            max_calculation_time_exists, 
                                            max_calculation_time, 
                                            config, 
                                            report)
            
        elif opt_problem_type == 'BinPacking':
            samples = self.run_binpacking_heuristic(opt_problem, 
                                                  number_of_samples, 
                                                  max_calculation_time_exists, 
                                                  max_calculation_time, 
                                                  config, 
                                                  report)
            
        elif opt_problem_type == 'CVRP':
            samples = self.run_cvrp_heuristic(opt_problem, 
                                            number_of_samples, 
                                            max_calculation_time_exists, 
                                            max_calculation_time, 
                                            config, 
                                            report)
        
        else:
            raise CustomizedError(error_message=f"wrong configuration of the Opt_Heuristics. For the problem type {opt_problem_type} there is no optimization heuristic implemented")
                
        # %% 
        config.logger.info("The OptHeuristic finished. \n")  
        
        report.time_measurements['OptHeuristic_run_total_time_consumption'] = time.time() - start_time
        report.time_measurements['opt_method_run_total_time_consumption'] = time.time() - start_time
        return samples
    
    
    def run_markowitzportfolio_heuristic(self, 
                                         opt_problem: any, 
                                         number_of_samples: int,
                                         chosen_heuristic: str,
                                         max_calculation_time_exists: bool, 
                                         max_calculation_time: int, 
                                         config: Config,
                                         report: Report
                                         ) -> list:
        ''' 
        function that tries to solve the MarkowitzPortfolio optimization problem heuristically
        '''
        start_time = time.time()
        assert opt_problem.problem_instance.opt_goal == "min_volatility_with_constrained_return", "the MarkowitzPortfolio Heuristic was implemented only for the 'min_volatility_with_constrained_return'-formulation."
        assert all(x == opt_problem.problem_instance.asset_limits[0] for x in opt_problem.problem_instance.asset_limits), "for the implemented MarkowitzPortfolio Heuristic to work, all asset limits have to be the same"
        
        if chosen_heuristic == "default": 
            chosen_heuristic = "steepestdescent_weightadding"
            
        variable_discretization_n = config.problem_config['variable_discretization_n']
        lowest_asset_share = opt_problem.problem_instance.asset_limits[0] / (2**variable_discretization_n)
        return_improving_asset_indices = [i for i, ret in enumerate(opt_problem.problem_instance.asset_returns) if ret > opt_problem.problem_instance.min_return]
        number_of_assets = len(opt_problem.problem_instance.asset_returns)
        
        if max_calculation_time_exists:
            config.logger.info(f"we search for samples with time limit of {max_calculation_time} min.") 
        else:
            config.logger.info(f"no time limit. we search for {number_of_samples} samples")
        
        
        markowitz_samples = []
        if chosen_heuristic == "steepestdescent_weightadding":
            self.num_reads = number_of_assets
            for asset_idx in range(number_of_assets):
                sample = np.array([0.0] * len(opt_problem.problem_instance.asset_returns))
                sample[asset_idx] += lowest_asset_share
                while round(sum(sample), 8) < 1 - lowest_asset_share / 2: # 1 - x , because this then is the best option regarding the qubo-penalty of the asset-weights-sum-up-to-1-constraint
                    sample = self.add_new_asset_weight_steepestdesc(
                                    sample, 
                                    opt_problem.problem_instance.asset_returns,
                                    opt_problem.problem_instance.asset_limits,
                                    opt_problem.problem_instance.asset_covariance_matrix,
                                    lowest_asset_share, 
                                    opt_problem.problem_instance.min_return)
                if not any(np.array_equal(sample, existing_sample) for existing_sample in markowitz_samples):
                    markowitz_samples.append(sample)
                
                if max_calculation_time_exists and time.time() - start_time >= 60*max_calculation_time:
                    self.num_reads = len(markowitz_samples)
                    break 
        elif chosen_heuristic == "randomized_weightadding":
            self.num_reads = number_of_samples
            while len(markowitz_samples) < number_of_samples:
                sample = np.array([0.0] * len(opt_problem.problem_instance.asset_returns))
                while sum(sample) < 1 - lowest_asset_share / 2: # 1 - x , because this then is the best option regarding the qubo-penalty of the asset-weights-sum-up-to-1-constraint
                    sample = self.add_new_asset_weight_randomized(
                                    sample, 
                                    opt_problem.problem_instance.asset_returns,
                                    return_improving_asset_indices,
                                    lowest_asset_share, 
                                    opt_problem.problem_instance.min_return)
                if not any(np.array_equal(sample, existing_sample) for existing_sample in markowitz_samples):
                    markowitz_samples.append(sample)
                
                if max_calculation_time_exists and time.time() - start_time >= 60*max_calculation_time:
                    self.num_reads = len(markowitz_samples)
                    break 
        else:
            raise CustomizedError(f"the chosen heuristic '{chosen_heuristic}' doesn't exist")
                
        config.logger.info(f"we found {len(markowitz_samples)} samples")
            
        report.solution_quality['number_of_shots'] = len(markowitz_samples)
        report.time_measurements['MarkowitzPortfolio_heuristic_samples_creation'] = time.time() - start_time
        
        markowitz_samples_discretized = self.discretize_markowitz_samples(markowitz_samples, variable_discretization_n, opt_problem.problem_instance.asset_limits[0])
        markowitz_samples_discretized_var_dict = self.transform_samples_list_to_var_dicts(markowitz_samples_discretized, opt_problem.var_names)
        
        return markowitz_samples, markowitz_samples_discretized, markowitz_samples_discretized_var_dict

    
    @staticmethod
    def add_new_asset_weight_steepestdesc(asset_weights: np.array, 
                                          asset_returns: np.array, 
                                          asset_limits: np.array,
                                          cov_matrix: np.array,
                                          lowest_asset_share: float, 
                                          minreturn: float):
        ''' 
        function that adds the lowest_asset_share to one of the asset_weights.
        Criterium on which asset gets chosen: 
        -> the one that improves the volatility the most while still satisfying the minreturn
        '''
        new_volas_list = []
        new_returns_list = []
        vola_big_M = 1000
        return_small_M = -1000
        for asset_idx in range(len(asset_weights)):
            new_weights = copy.deepcopy(asset_weights)
            new_weights[asset_idx] += lowest_asset_share
            if new_weights[asset_idx] <= asset_limits[asset_idx]:
                new_return = np.dot(new_weights, asset_returns) / np.sum(new_weights)# * (1-lowest_asset_share/2)
                new_returns_list.append(new_return)
                if new_return >= minreturn:
                    new_vola = np.dot(new_weights, np.dot(cov_matrix, new_weights))
                    new_volas_list.append(new_vola)
                else: # hopeless case, where return drops too much
                    new_volas_list.append(vola_big_M) 
            else: # hopeless case, asset limit is violated
                new_returns_list.append(return_small_M)
                new_volas_list.append(vola_big_M) 
        
        if min(new_volas_list) == vola_big_M:
            most_improv_asset = np.argmax(new_returns_list)
        else:
            most_improv_asset = np.argmin(new_volas_list)
            
        asset_weights[most_improv_asset] += lowest_asset_share
        return asset_weights
                
    
    @staticmethod
    def add_new_asset_weight_randomized(asset_weights: np.array, 
                             asset_returns: np.array, 
                             return_improving_asset_indices: list,
                             lowest_asset_share: float, 
                             minreturn: float,
                             max_tries: int = 20):
        ''' 
        function that adds the lowest_asset_share to one of the asset_weights.
        Criterium on which assets gets chosen, depends on the current average return
        of the so far selected assets
        '''
        if np.sum(asset_weights) + lowest_asset_share < 1: # if other weight shares will be added afterwards because sum of asset weights still < 1
            current_average_return = np.dot(asset_weights, asset_returns) / np.sum(asset_weights)
            if current_average_return < minreturn:
                new_sample_index = random.sample(return_improving_asset_indices, 1)[0]
            else:
                new_sample_index = random.sample(list(range(len(asset_returns))), 1)[0]            
            asset_weights[new_sample_index] += lowest_asset_share
        
        else: # if this the last adding of a weight share    
            new_asset_weights = copy.copy(asset_weights)
            counter = 0
            while counter <= max_tries:
                new_asset_weights = copy.copy(asset_weights)
                new_sample_index = random.sample(list(range(len(asset_returns))), 1)[0]
                new_asset_weights[new_sample_index] += lowest_asset_share
                if np.dot(new_asset_weights, asset_returns) / np.sum(new_asset_weights) >= minreturn:
                    asset_weights[new_sample_index] += lowest_asset_share
                    break
                counter += 1  
            else:
                #we land here if the upper while loop isn't broken by 'break'
                most_improv_asset = np.argmax(asset_returns)
                asset_weights[most_improv_asset] += lowest_asset_share                    
                
        return asset_weights
        
    
    @staticmethod 
    def discretize_markowitz_samples(samples: list, variable_discretization_n: int, asset_limit: float):
        ''' 
        function to discretize the portfolio weight samples that were created with the heuristic
        '''
        discr_samples = []
        for sample in samples:
            discr_sample = []
            for weight in sample:
                discr_weight = []
                for exponent in list(range(variable_discretization_n)) + [variable_discretization_n-1]:
                    discretized_factor = asset_limit * (1 / 2**(exponent+1))
                    if round(weight, 6) >= round(discretized_factor, 6):
                        discr_weight.append(1)
                        weight -= discretized_factor
                    else:
                        discr_weight.append(0)
                    
                discr_sample.extend(discr_weight)
            discr_samples.append(np.array(discr_sample))
        return discr_samples
    
    
    def transform_samples_list_to_var_dicts(self, 
                                            samples_list: list, 
                                            var_names: list
                                            ) -> list:
        list_of_var_dicts = []
        for sample in samples_list:
            list_of_var_dicts.append(dict(zip(var_names, sample)))
        return list_of_var_dicts      
    
    
    def run_tsp_heuristic(self, 
                        problem_instance: tuple, 
                        number_of_samples: int, 
                        max_calculation_time_exists: bool, 
                        max_calculation_time: int, 
                        config: Config,
                        report: Report) -> list:
        ''' 
        function that tries to solve the TSP optimization problem heuristically
        '''
        raise CustomizedError("currently not implemented")
    
    
    
    def run_cvrp_heuristic(self, 
                        problem_instance: tuple, 
                        number_of_samples: int, 
                        max_calculation_time_exists: bool, 
                        max_calculation_time: int, 
                        config: Config,
                        report: Report) -> list:
        ''' 
        function that tries to solve the CVRP optimization problem heuristically
        '''
        raise CustomizedError("currently not implemented")
    
    
    
    def run_binpacking_heuristic(self, 
                        problem_instance: tuple, 
                        number_of_samples: int, 
                        max_calculation_time_exists: bool, 
                        max_calculation_time: int, 
                        config: Config,
                        report: Report) -> list:
        ''' 
        function that tries to solve the BinPacking optimization problem heuristically
        '''
        raise CustomizedError("currently not implemented")
    
    
    def analyse_solver_result(self, solver_result, config: Config, report: Report) -> (dict, dict):
        '''
        function to analyse the result of the Optimization Heuristics: 
            - get the best sample
            - feasibility analysis
            - comparison with random sampling
        '''
        start_time = time.time()
        solver_result_discretized = solver_result[1]
        # %% feasibility analysis of samples
        variable_names_list = [] # first we get the original variable names that were in the MIP
        for variable in self.mip_for_feasibility_analysis.variables:
            variable_names_list.append(variable.name)
            
        result_dict = self.map_heuristic_to_result_dict(solver_result_discretized)
        result_dict_feasibility_analysed = analyse_feasibility_of_solver_solutions(result_dict, self.qubo, self.mip_for_feasibility_analysis, config, report)
        if config.report_config["visualize_distribution_of_obj_values"] == True:
            create_barplot_obj_values_with_number_of_violated_constraints(result_dict_feasibility_analysed, config.EXPORT_PATH, "barplot_obj_values_GreedyAlg")
        feas_sol_dict, feas_sol_df = get_all_feasible_solutions(result_dict_feasibility_analysed, variable_names_list, report)
        if config.report_config["save_feasible_sols_to_excel"] == True:
            feas_sol_df.to_excel(os.path.join(config.EXPORT_PATH, "feasible_solutions.xlsx"), index=False)
        write_kpis_to_report(feas_sol_dict, self.num_reads, config, report)
        
        # %% expectation value comparison with random sampling
        expec_value_opt_heuristic = np.mean([bitstr_dict["mip_obj_value"] for bitstr_dict in feas_sol_dict.values() if "mip_obj_value" in bitstr_dict])
        expec_value_rand_sampling = calculate_expectation_value_random_sampling_qubo(self.qubo_dict, self.qubo.objective.constant)
        report.solution_quality['expectation_obj_value_opt_heuristic'] = expec_value_opt_heuristic
        report.solution_quality['expectation_obj_value_random_sampling'] = expec_value_rand_sampling
        # %%
        
        report.time_measurements['opt_method_result_analysis_total_time_consumption'] = time.time() - start_time
        return feas_sol_dict, result_dict_feasibility_analysed
            
        
    @staticmethod 
    def map_heuristic_to_result_dict(solver_result: list):
        solver_result_dict = {}
        for bitarray in solver_result:
            bitstr = ', '.join(map(str, bitarray))
            solver_result_dict[bitstr] = {}
            solver_result_dict[bitstr]['count'] = 1
        return solver_result_dict
    