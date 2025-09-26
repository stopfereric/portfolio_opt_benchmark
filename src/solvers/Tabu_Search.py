# -*- coding: utf-8 -*-
"""
Created on 07.03.2024

@author: Eric Stopfer
"""
import time
import pdb
import os
import math
import numpy as np

from qiskit_optimization import QuadraticProgram
from tabu import TabuSampler #from DWave 

from config.config import Config
from src.report import Report
from src.utils.error_utils import CustomizedError
from src.utils.qc_utils import calculate_expectation_value_random_sampling_qubo
from src.utils.solver_result_utils import process_dwave_response, analyse_feasibility_of_solver_solutions, get_all_feasible_solutions, write_kpis_to_report
from src.utils.visualisation_utils import create_barplot_obj_values_with_number_of_violated_constraints
from src.utils.opt_utils import execute_heuristic_for_initial_states, execute_steepest_descent_on_bitstrings



class Tabu_Search:
    '''
    ---- Definition from https://www.javatpoint.com/what-is-a-tabu-search -----
    
    - Tabu search is a metaheuristic algorithm employed to solve optimization issues. 
    Its name is derived from the Arabic word "tabu," which denotes anything banned. 
    By keeping a short-term memory of the search process and utilizing this knowledge 
    to direct the search toward promising regions, the Tabu search is made to explore 
    the solution space efficiently.
    - Starting with one answer, the algorithm iteratively examines the surrounding 
    solutions by making certain adjustments or modifications. It assesses each 
    neighbour's quality based on an objective function or assessment criteria 
    particular to the current issue. The goal is to select the optimal solution 
    or to maximize a particular objective function.
    - Using a tabu list is one of the tabu search's distinguishing characteristics. 
    The search is prevented from returning to the exact solutions or being bogged 
    down in loops by this list, which maintains track of the movements or 
    transformations that have recently been used. The tabu list ensures the diverse 
    search process, enabling the exploration of a larger solution space.
    
    ---- Pseudocode ----
    
    current_solution = create_first_solution()
    while termination-criterium is not fulfilled:
        neighborhood = create_neighborhood(current_solution)
        evaluate(neighborhood)
        neighborhood = delete_tabu_moves_from(neighborhood)
        current_solution = make_best_move_from(neighborhood)
        add_to_tabu_list(current_solution, tabu_time)
    end while
    
    '''
    def __init__(self, config: Config, report: Report):
        """
        Function that creates an Tabu-Search solver instance
        """
        self.device_name = config.solve_method_device
        self.device = self.get_device(self.device_name, report)
        
        config.logger.info("we are using the following solving method: %s" %config.solve_method)
        config.logger.info("with the following device: %s" %config.solve_method_device)
        
        return
    
    
    def get_device(self, device_name: str, report: Report) -> any:
        '''
        Function that gets the Tabu-Search device by a name-string
        '''
        start_time = time.time()
        if device_name == "Dwave_TabuSampler":
            #the Dwave-Tabu-Sampler works with one-bit-flip-neighborhoods
            #e.g. if we are at solution (0,1,1) --> the neighborhood is chosen as: (#1#,1,1), (0,#0#,1), (0,1,#0#)
            device = TabuSampler()
            report.time_measurements['connect_to_DwaveTabuSearchSampler'] = time.time() - start_time
        else:
            raise CustomizedError(error_message="Wrong configuration of the device. Check in config.py or config.json")
        return device
        
        
    def run(self, 
            opt_problem, 
            config: Config, 
            report: Report,
            max_reads: int = 100000 # 1 million
            ) -> dict:
        '''
        Function that runs the Tabu-Search for a problem that is mapped 
        to a QUBO dictionary and solving configuration.
        '''
        start_time_overall = time.time()
        # %% read the necessary attributes from the configuration
        try:
            self.num_reads = config.solve_method_config['number_of_reads']
            time_limit = config.solve_method_config['time_limit']
            max_calculation_time_exists = config.solve_method_config['max_calculation_time_exists']
            calc_initial_states_heuristically = config.solve_method_config['calc_initial_states_with_heuristic']
            number_of_initial_states = config.solve_method_config['number_of_initial_states']
            
        except:
            raise CustomizedError(error_message="wrong or missing configuration of the Tabu-Search. check in config.py and config.json")
        
        self.opt_problem = opt_problem
        self.qubo_dict = opt_problem.problem_mapping[0]
        self.qubo = opt_problem.problem_mapping[1]
        self.mip_for_feasibility_analysis = opt_problem.problem_mapping[2]
                
        config.logger.info("the Tabu Search starts right now")
        
        tabu_responses = []
        
        if calc_initial_states_heuristically == True:
            initial_states = execute_heuristic_for_initial_states(opt_problem, number_of_initial_states, config, report)
            tabu_response = self.device.sample_qubo(
                Q = self.qubo_dict,
                num_reads = len(initial_states),
                initial_states = initial_states,                
                timeout = time_limit)
            tabu_responses.append(tabu_response)
        else:
            if max_calculation_time_exists == True:
                # we will run the simulated anneal in batches of n reads until the set time limit is exceeded
                max_time_in_sec = 45 * config.solve_method_config['max_calculation_time_in_min']
                config.logger.info(f"the tabu search will have a time limit of {max_time_in_sec} sec")
                
                # determine the batch size of sampling to not exceed the time limit
                time_elapsed_during_test = 0
                num_reads_for_batches = 1
                counter = 0
                start_time = time.time()
                while time_elapsed_during_test < 1:
                    time_meas = time.time()
                    num_reads_for_batches *= 2
                    tabu_response = self.device.sample_qubo(
                        Q = self.qubo_dict,
                        num_reads = num_reads_for_batches,
                        timeout = time_limit)
                    time_elapsed_during_test = time.time() - time_meas
                    counter += 1                
                report.time_measurements[f'tabusearch_execution_to_find_number_of_shots_for_approx_{max_time_in_sec}_sec'] = time.time() - start_time 
                config.logger.info(f"adjusted the number of reads per batch to {num_reads_for_batches} by doing {counter} method executions")
                
                #while not exceeding the timelimit we will execute the tabu search
                counter = 0
                time_elapsed = 0
                start_time = time.time()
                while time_elapsed < max_time_in_sec and counter * num_reads_for_batches < max_reads:
                    tabu_response = self.device.sample_qubo(
                        Q = self.qubo_dict,
                        num_reads = num_reads_for_batches,
                        timeout = time_limit)
                    tabu_responses.append(tabu_response)
                    time_elapsed = time.time() - start_time + 0.001
                    config.logger.info("finished a batch")
                    counter+=1
                
                self.num_reads = counter * num_reads_for_batches
                config.logger.info(f"We did {self.num_reads} reads")
                report.time_measurements['tabu_search_total'] = time.time() - start_time
            
            else:
                tabu_response = self.device.sample_qubo(
                    Q = self.qubo_dict,
                    num_reads = self.num_reads,                
                    timeout = time_limit)
                tabu_responses.append(tabu_response)
            
                        
        # %% 
        config.logger.info("The Tabu-Search finished. \n")
        report.time_measurements['opt_method_run_total_time_consumption'] = time.time() - start_time_overall
        return tabu_responses
    
        
    def analyse_solver_result(self, solver_result, config: Config, report: Report) -> (dict, dict):
        '''
        function to analyse the result of the Greedy Algorithm solver: 
            - get the best sample
            - feasibility analysis
            - comparison with random sampling
        '''
        start_time = time.time()
        # %% feasibility analysis of samples
        variable_names_list = [] # first we get the original variable names that were in the MIP
        for variable in self.mip_for_feasibility_analysis.variables:
            variable_names_list.append(variable.name)
        result_dict = process_dwave_response(solver_result, variable_names_list, config, report)
        result_dict_feasibility_analysed = analyse_feasibility_of_solver_solutions(result_dict, self.qubo, self.mip_for_feasibility_analysis, config, report)
        if config.report_config["visualize_distribution_of_obj_values"] == True:
            create_barplot_obj_values_with_number_of_violated_constraints(result_dict_feasibility_analysed, config.EXPORT_PATH, "barplot_obj_values_TabuSearch")
        feas_sol_dict, feas_sol_df = get_all_feasible_solutions(result_dict_feasibility_analysed, variable_names_list, report)
        if config.report_config["save_feasible_sols_to_excel"] == True:
            feas_sol_df.to_excel(os.path.join(config.EXPORT_PATH, "feasible_solutions.xlsx"), index=False)
        report.solution_quality['feasible_solutions_how_many'] = len(feas_sol_dict.keys())
        write_kpis_to_report(feas_sol_dict, self.num_reads, config, report)
        
        if config.solve_method_config['postprocess_with_steepest_descent'] == True:
            feas_sol_dict, _ = execute_steepest_descent_on_bitstrings(self.opt_problem, feas_sol_dict, config, report)
        
        # %% expectation value comparison with random sampling
        all_energies = []
        for dwave_response in solver_result:
            all_energies.extend(list(dwave_response.data_vectors['energy']))
        all_obj_values = [energy + self.qubo.objective.constant for energy in all_energies]
        expec_value_tabu_search = np.mean(all_obj_values)
        expec_value_rand_sampling = calculate_expectation_value_random_sampling_qubo(self.qubo_dict, self.qubo.objective.constant)
        report.solution_quality['expectation_obj_value_tabu_search'] = expec_value_tabu_search
        report.solution_quality['expectation_obj_value_random_sampling'] = expec_value_rand_sampling
        # %%
        report.time_measurements['opt_method_result_analysis_total_time_consumption'] = time.time() - start_time
        return feas_sol_dict, result_dict_feasibility_analysed