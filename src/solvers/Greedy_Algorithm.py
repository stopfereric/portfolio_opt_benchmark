# -*- coding: utf-8 -*-
"""
Created on 07.03.2024

@author: Eric Stopfer
"""
import pdb
import time
import os
import math
import numpy as np

from greedy import SteepestDescentSolver #from DWave 
from qiskit_optimization import QuadraticProgram

from config.config import Config
from src.report import Report
from src.utils.error_utils import CustomizedError
from src.utils.qc_utils import calculate_expectation_value_random_sampling_qubo
from src.utils.solver_result_utils import process_dwave_response, analyse_feasibility_of_solver_solutions, get_all_feasible_solutions, write_kpis_to_report
from src.utils.visualisation_utils import create_barplot_obj_values_with_number_of_violated_constraints


class Greedy_Algorithm:
    '''
    ---- Definition from https://en.wikipedia.org/wiki/Greedy_algorithm -----
    
    A greedy algorithm is any algorithm that follows the problem-solving heuristic 
    of making the locally optimal choice at each stage [-->takes the solution with minimal
    energy in the neighborhood of the current solution]. In many problems, 
    a greedy strategy does not produce an optimal solution, but a greedy heuristic 
    can yield locally optimal solutions that approximate a globally optimal solution 
    in a reasonable amount of time.
    '''
    
    def __init__(self, config: Config, report: Report):
        """
        Function that creates an Greedy-Algorithm solver instance
        """
        self.device_name = config.solve_method_device
        self.device = self.get_device(self.device_name, report)
        
        config.logger.info("we are using the following solving method: %s" %config.solve_method)
        config.logger.info("with the following device: %s" %config.solve_method_device)
        
    
    def get_device(self, device_name: str, report: Report) -> any:
        '''
        Function that gets the Greedy-Algorithm device by a name-string
        '''
        start_time = time.time()
        if device_name == "Dwave_GreedyAlgorithm":
            #the Dwave-GreedyAlgorihm works with one-bit-flip-neighborhoods
            #choses the bitstring in a neighborhood that improves the objective funciton the most
            #e.g. if we are at solution (0,1,1) --> the neighborhood is chosen as: (#1#,1,1), (0,#0#,1), (0,1,#0#)
            #if no initial states specified they are chosen as random bitstrings.
            device = SteepestDescentSolver()
            report.time_measurements['connect_to_DwaveGreedyAlgorithm'] = time.time() - start_time
        else:
            raise CustomizedError(error_message="Wrong configuration of the device. Check in config.py or config.json")
        return device
        
        
    def run(self, 
            opt_problem, 
            config: Config, 
            report: Report, 
            initial_states: np.array = None,
            max_reads: int = 5000000 # 1 million
            ) -> dict:
        '''
        Function that runs the Greedy-Algorithm for a problem that is mapped 
        to a QUBO dictionary and solving configuration.
        '''
        start_time_overall = time.time()
        # %% read the necessary attributes from the configuration
        try:
            self.num_reads = config.solve_method_config['number_of_reads'] if initial_states == None else len(initial_states)
            max_calculation_time_exists = config.solve_method_config['max_calculation_time_exists']
        except:
            raise CustomizedError(error_message="wrong or missing configuration of the Greedy-Algorithm. check in config.py and config.json")
        
        self.opt_problem = opt_problem
        self.qubo_dict = opt_problem.problem_mapping[0]
        self.qubo = opt_problem.problem_mapping[1]
        self.mip_for_feasibility_analysis = opt_problem.problem_mapping[2]
        
        config.logger.info("the Greedy Algorithm starts right now")
        greedy_responses = []
        if max_calculation_time_exists == True:
            # we will run the simulated anneal in batches of n reads until the set time limit is exceeded
            max_time_in_sec = 45 * config.solve_method_config['max_calculation_time_in_min']
            config.logger.info(f"the greedy algorithm will have a time limit of {max_time_in_sec} sec")
            
            # determine the batch size of sampling to not exceed the time limit
            time_elapsed_during_test = 0
            num_reads_for_batches = 1
            counter = 0
            start_time = time.time()
            while time_elapsed_during_test < 1:
                time_meas = time.time()
                num_reads_for_batches *= 2
                greedy_response = self.device.sample_qubo(
                    Q = self.qubo_dict,
                    num_reads = num_reads_for_batches
                    )
                time_elapsed_during_test = time.time() - time_meas
                counter += 1                
            report.time_measurements[f'greedyalgorithm_execution_to_find_number_of_shots_for_approx_{max_time_in_sec}_sec'] = time.time() - start_time 
            config.logger.info(f"adjusted the number of reads per batch to {num_reads_for_batches} by doing {counter} method executions")
            
            #while not exceeding the timelimit we will execute the simulated anneal
            counter = 0
            time_elapsed = 0
            start_time = time.time()
            while time_elapsed < max_time_in_sec and counter * num_reads_for_batches < max_reads:
                greedy_response = self.device.sample_qubo(
                    Q = self.qubo_dict,
                    num_reads = num_reads_for_batches
                    )
                greedy_responses.append(greedy_response)
                time_elapsed = time.time() - start_time + 0.001
                config.logger.info("finished a batch")
                counter+=1
            
            self.num_reads = counter * num_reads_for_batches
            config.logger.info(f"We did {self.num_reads} reads")
            report.time_measurements['greedy_algorithm_total'] = time.time() - start_time
        
        else:
            greedy_response = self.device.sample_qubo(
                Q = self.qubo_dict,
                num_reads = self.num_reads,
                initial_states = initial_states
                )
            greedy_responses.append(greedy_response)
                
        # %% 
        config.logger.info("The Greedy-Algorithm finished. \n")  
        
        report.time_measurements['GreedyAlgorithm_total_time_consumption'] = time.time() - start_time_overall
        report.time_measurements['opt_method_run_total_time_consumption'] = time.time() - start_time_overall
        return greedy_responses
    
    
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
            create_barplot_obj_values_with_number_of_violated_constraints(result_dict_feasibility_analysed, config.EXPORT_PATH, "barplot_obj_values_GreedyAlg")
        feas_sol_dict, feas_sol_df = get_all_feasible_solutions(result_dict_feasibility_analysed, variable_names_list, report)
        if config.report_config["save_feasible_sols_to_excel"] == True:
            feas_sol_df.to_excel(os.path.join(config.EXPORT_PATH, "feasible_solutions.xlsx"), index=False)
        write_kpis_to_report(feas_sol_dict, self.num_reads, config, report)
        
        # %% expectation value comparison with random sampling
        all_energies = []
        for dwave_response in solver_result:
            all_energies.extend(list(dwave_response.data_vectors['energy']))
        all_obj_values = [energy + self.qubo.objective.constant for energy in all_energies]
        expec_value_greedy_alg = np.mean(all_obj_values)
        expec_value_rand_sampling = calculate_expectation_value_random_sampling_qubo(self.qubo_dict, self.qubo.objective.constant)
        report.solution_quality['expectation_obj_value_greedy_alg'] = expec_value_greedy_alg
        report.solution_quality['expectation_obj_value_random_sampling'] = expec_value_rand_sampling
        # %%
        
        report.time_measurements['opt_method_result_analysis_total_time_consumption'] = time.time() - start_time
        return feas_sol_dict, result_dict_feasibility_analysed
            
          
        