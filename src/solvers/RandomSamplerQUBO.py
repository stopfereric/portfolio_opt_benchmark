# -*- coding: utf-8 -*-
"""
Created on 07.03.2024

@author: Eric Stopfer
"""
import pdb
import time
import os
import numpy as np

from config.config import Config
from src.report import Report
from src.utils.error_utils import CustomizedError
from src.utils.qc_utils import calculate_expectation_value_random_sampling_qubo, calc_ising_energy_of_bitstring_array
from src.utils.solver_result_utils import analyse_feasibility_of_solver_solutions, get_all_feasible_solutions, write_kpis_to_report
from src.utils.visualisation_utils import create_barplot_obj_values_with_number_of_violated_constraints


class RandomSamplerQUBO:
    '''
    RandomSamplerQUBO samples the binary variables of a QUBO randomly
    '''
    
    def __init__(self, config: Config, report: Report):
        """
        Function that creates an Greedy-Algorithm solver instance
        """
        config.logger.info("we are using the following solving method: %s" %config.solve_method)
        self.samples = [] #initialize
        self.energies = [] #initialize
        
        
    def run(self, 
            opt_problem, 
            config: Config, 
            report: Report,
            ) -> dict:
        '''
        Function that runs the Greedy-Algorithm for a problem that is mapped 
        to a QUBO dictionary and solving configuration.
        '''
        start_time_overall = time.time()
        # %% read the necessary attributes from the configuration
        try:
            self.num_reads = config.solve_method_config['number_of_reads'] 
            max_calculation_time_exists = config.solve_method_config['max_calculation_time_exists']
            max_calc_time = config.solve_method_config['max_calculation_time_in_min']
        except:
            raise CustomizedError(error_message="wrong or missing configuration of the Greedy-Algorithm. check in config.py and config.json")
        
        self.opt_problem = opt_problem
        self.qubo_dict = opt_problem.problem_mapping[0]
        self.qubo = opt_problem.problem_mapping[1]
        self.mip_for_feasibility_analysis = opt_problem.problem_mapping[2]
        self.ising_matrix = opt_problem.problem_mapping[3]
        self.ising_vector = opt_problem.problem_mapping[4]
        self.ising_offset = opt_problem.problem_mapping[5]
        self.qubo_num_vars = self.qubo.get_num_vars()
        
        config.logger.info("the Random Sampling starts right now")
        if max_calculation_time_exists == True:
            max_time_in_sec = 60 * max_calc_time
            config.logger.info(f"the random sampling will have a time limit of {max_time_in_sec} sec")
            
            time_elapsed = 0 #initialize
            start_time = time.time()
            time_sampling = 0
            time_evaluating = 0
            while time_elapsed < max_time_in_sec:
                a = time.time()
                ising_sample = np.random.choice([-1, 1], size=self.qubo_num_vars)
                bitlist = [1 if i==-1 else 0 for i in ising_sample]
                time_sampling += time.time() - a
                b = time.time()
                energy_of_sample = calc_ising_energy_of_bitstring_array(self.ising_matrix, self.ising_vector, self.ising_offset, ising_sample)
                time_evaluating += time.time() - b
                self.samples.append(bitlist)
                self.energies.append(energy_of_sample)
                time_elapsed = time.time() - start_time
            self.num_reads = len(self.samples)
            config.logger.info(f"we were able to generate {self.num_reads} samples within {time_elapsed} sec")
            config.logger.info(f"during random sampling we took {round(time_sampling, 2)}s for sampling and {round(time_evaluating, 2)}s for evaluating \n")
        else:
            a = time.time()
            random_ising_samples_array = np.random.choice([-1, 1], size=(self.num_reads, self.qubo_num_vars))
            self.samples = (random_ising_samples_array == -1).astype(int).tolist()
            time_sampling = time.time() - a
            b = time.time()
            self.energies = [calc_ising_energy_of_bitstring_array(self.ising_matrix, self.ising_vector, self.ising_offset, sample) for sample in random_ising_samples_array]
            time_evaluating = time.time() - b             
            config.logger.info(f"during random sampling we took {round(time_sampling, 2)}s for sampling and {round(time_evaluating, 2)}s for evaluating \n")
        
        report.time_measurements['opt_method_run_total_time_consumption'] = time.time() - start_time_overall
        return self.samples, self.energies
    
    
    def analyse_solver_result(self, solver_result, config: Config, report: Report) -> (dict, dict):
        '''
        function to analyse the result of the Greedy Algorithm solver: 
            - get the best sample
            - feasibility analysis
            - comparison with random sampling
        '''
        start_time = time.time()
        # %% feasibility analysis of samples
        # first we get the original variable names that were in the MIP
        variable_names_list = [] 
        for variable in self.mip_for_feasibility_analysis.variables:
            variable_names_list.append(variable.name)
            
        result_dict = {} 
        for idx, bitlist in enumerate(self.samples):
            bitstring = ', '.join([str(item) for item in bitlist])
            if bitstring not in result_dict.keys():
                result_dict[bitstring] = {}
                result_dict[bitstring]['energy'] = self.energies[idx]
                result_dict[bitstring]['count'] = 1
            else:
                result_dict[bitstring]['count'] += 1            
        
        if config.report_config["analyse_only_best_sols"] == True:
            sorted_items = sorted(result_dict.items(), key=lambda item: item[1]['energy'])
            number_of_sols_to_keep = config.report_config["number_of_sols_to_be_analysed"]
            result_dict = dict(sorted_items[:number_of_sols_to_keep])
            config.logger.info(f"We only keep the {number_of_sols_to_keep} best solutions")
        
        result_dict_feasibility_analysed = analyse_feasibility_of_solver_solutions(result_dict, self.qubo, self.mip_for_feasibility_analysis, config, report)
        if config.report_config["visualize_distribution_of_obj_values"] == True:
            create_barplot_obj_values_with_number_of_violated_constraints(result_dict_feasibility_analysed, config.EXPORT_PATH, "barplot_obj_values_GreedyAlg")
        feas_sol_dict, feas_sol_df = get_all_feasible_solutions(result_dict_feasibility_analysed, variable_names_list, report)
        if config.report_config["save_feasible_sols_to_excel"] == True:
            feas_sol_df.to_excel(os.path.join(config.EXPORT_PATH, "feasible_solutions.xlsx"), index=False)
        write_kpis_to_report(feas_sol_dict, self.num_reads, config, report)
        
        # %% expectation value comparison with random sampling
        expec_value_randomsamplerqubo = np.mean(self.energies)
        expec_value_rand_sampling = calculate_expectation_value_random_sampling_qubo(self.qubo_dict, self.qubo.objective.constant)
        report.solution_quality['expectation_obj_value_randomsamplerqubo'] = expec_value_randomsamplerqubo
        report.solution_quality['expectation_obj_value_random_sampling'] = expec_value_rand_sampling
        # %%
        
        report.time_measurements['opt_method_result_analysis_total_time_consumption'] = time.time() - start_time
        return feas_sol_dict, result_dict_feasibility_analysed
            
          
        