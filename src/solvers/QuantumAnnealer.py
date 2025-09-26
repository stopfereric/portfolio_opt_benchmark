# -*- coding: utf-8 -*-
"""
Created on 07.03.2024

@author: Eric Stopfer
"""
import time
import pdb
import json
import os
import numpy as np

from dwave.system import DWaveSampler, DWaveCliqueSampler, EmbeddingComposite, AutoEmbeddingComposite, FixedEmbeddingComposite #for Quantum Annealing
from dwave.cloud import Client 
from dimod import BinaryQuadraticModel

from config.config import Config
from src.report import Report
from src.utils.error_utils import CustomizedError
from src.utils.qc_utils import calculate_expectation_value_random_sampling_qubo
from src.utils.solver_result_utils import process_dwave_response, analyse_feasibility_of_solver_solutions, get_all_feasible_solutions, write_kpis_to_report
from src.utils.visualisation_utils import create_barplot_obj_values_with_number_of_violated_constraints
from src.utils.opt_utils import execute_steepest_descent_on_bitstrings


class QuantumAnnealer:
    '''
    ---- from Wikipedia -----
    
    The word ANNEALING describes the process of slowly cooling hot glass objects after they have been formed, 
    to relieve residual internal stresses introduced during manufacture. (https://en.wikipedia.org/wiki/Annealing_(glass) )
    
    QUANTUM ANNEALING (QA) is an optimization process for finding the global minimum of a given objective 
    function over a given set of candidate solutions (candidate states), by a process using quantum fluctuations.
    (https://en.wikipedia.org/wiki/Quantum_annealing )
    '''
    def __init__(self, config: Config, report: Report):
        """
        Function that creates a Quantum Annealing solver instance
        """
        self.device_name = config.solve_method_device
        if config.problem_name == "MarkowitzPortfolio":
            self.sampler = 'DWaveCliqueSampler' # 'EmbeddingComposite' # 
        else:
            self.sampler = 'EmbeddingComposite' # 'AutoEmbeddingComposite' # 'FixedEmbeddingComposite' # 
        self.device = self.get_device(self.device_name, self.sampler, config, report)
        
        config.logger.info("we are using the following solving method: %s" %config.solve_method)
        config.logger.info("with the following device: %s" %config.solve_method_device)
        
    
    def get_device(self, device_name: str, sampler: str, config: Config, report: Report) -> any:
        '''
        Function that gets the Quantum Annealing device by a name-string
        '''
        start_time = time.time()
        if device_name == "Quantum_Annealer_Dwave":
            solver_api_tokens_path = config.BASE_PATH_CONFIG + "\\config_files\\" + "solver_api_tokens.json"
            with open(solver_api_tokens_path, "r") as json_file:
                solver_api_tokens_data = json.load(json_file)
            dwave_token = solver_api_tokens_data['Dwave_Token']
            dwave_processor = solver_api_tokens_data['Dwave_QuantumComputer']
            if dwave_token == "type_your_dwave_api_token_here": # default value
                raise CustomizedError("for Quantum Annealer of Dwave a valid API token is necessary. it has to be put inside the file: \n C:\...\quopt\config\config_files\solver_api_tokens.json")
            if sampler == 'DWaveCliqueSampler':
                device = DWaveCliqueSampler(token=dwave_token, 
                                            solver=dwave_processor)
            elif sampler == 'AutoEmbeddingComposite':
                device = AutoEmbeddingComposite(DWaveSampler(token=dwave_token, 
                                                             solver=dwave_processor))
            # elif sampler == 'FixedEmbeddingComposite':
            #     device = FixedEmbeddingComposite(DWaveSampler(token=dwave_token,
            #                                                   solver=dwave_processor))
            elif sampler == 'EmbeddingComposite':
                device = EmbeddingComposite(DWaveSampler(token=dwave_token, 
                                                         solver=dwave_processor))
            else:
                raise CustomizedError("unknown dwave sampler: {sampler}")
            config.logger.info(f"We are using the {sampler} on {dwave_processor}")
            report.time_measurements['connect_to_dwave_annealer'] = time.time() - start_time
        else:
            raise CustomizedError(error_message="Wrong configuration of the device. Check in config.py or config.json")
        return device
        
        
    def run(self,
            opt_problem, 
            config: Config, 
            report: Report
            ) -> dict:
        '''
        Function that runs a Quantum Annealing process for a problem that is mapped 
        to a QUBO dictionary and solving configuration.
        '''
        start_time = time.time()
        # %% read the necessary attributes from the configuration
        try:
            self.num_reads = config.solve_method_config['number_of_reads']
            annealing_time = config.solve_method_config['annealing_time']
            use_custom_chain_strength = config.solve_method_config['use_custom_chain_strength']
            chain_strength = (config.solve_method_config['chain_strength'] 
                              if use_custom_chain_strength 
                              else None)
            max_calculation_time_exists = config.solve_method_config['max_calculation_time_exists']
            max_calc_time_in_min = config.solve_method_config['max_calculation_time_in_min']
        except:
            raise CustomizedError(error_message="wrong or missing configuration of the Quantum Annealer. check in config.py and config.json")
        
        self.opt_problem = opt_problem
        self.qubo_dict = opt_problem.problem_mapping[0]
        self.qubo = opt_problem.problem_mapping[1]
        self.mip_for_feasibility_analysis = opt_problem.problem_mapping[2]
        
        # %% do the anneal
        annealer_responses = []
        if max_calculation_time_exists:
            if self.sampler == 'DWaveCliqueSampler':
                max_num_reads = self.device.properties['qpu_properties']['num_reads_range'][1]
                max_qpu_time = self.device.properties['qpu_properties']['problem_run_duration_range'][1]
            elif self.sampler in ['EmbeddingComposite', 'AutoEmbeddingComposite']:
                max_num_reads = self.device.properties['child_properties']['num_reads_range'][1]
                max_qpu_time = self.device.properties['child_properties']['problem_run_duration_range'][1]
            else:
                raise CustomizedError(f"unknown dwave sampler: {self.sampler}")
                
            batches_of_reads = self.get_optimal_num_reads_for_timelimit(
                timelimit_in_min = max_calc_time_in_min,
                annealing_time = annealing_time,
                max_num_reads_per_job = max_num_reads,
                max_qpu_time_per_job = max_qpu_time,
                config = config) 
            self.num_reads = sum(batches_of_reads)
            
            number_of_batches = len(batches_of_reads)
            config.logger.info(f"The Quantum anneal starts right now with {self.num_reads} reads and {annealing_time} µs annealing time at chain strength {chain_strength} and is split up into {number_of_batches} batches")
            for idx, num_of_reads in enumerate(batches_of_reads):
                annealer_response = self.device.sample_qubo(
                    Q = self.qubo_dict,
                    num_reads = num_of_reads,
                    annealing_time = annealing_time,
                    chain_strength = chain_strength)
                annealer_responses.append(annealer_response)
                config.logger.info(f"submitted job {idx+1}/{number_of_batches}")
        else:            
            config.logger.info(f"The Quantum anneal starts right now with {self.num_reads} reads and {annealing_time} µs annealing time at chain strength {chain_strength}")
            annealer_response = self.device.sample_qubo(
                Q = self.qubo_dict,
                num_reads = self.num_reads,
                annealing_time = annealing_time,
                chain_strength = chain_strength
                )
            annealer_responses.append(annealer_response)
                        
            report.time_measurements['quantum_annealing_job_sumission'] = time.time() - start_time
            config.logger.info("Finished submitting the quantum anneal jobs.  \n")
            # %% 
            report.time_measurements['opt_method_run_total_time_consumption'] = 'will be filled later, here only job submission'
        return annealer_responses
    
    
    def get_optimal_num_reads_for_timelimit(self, 
                                            timelimit_in_min: float, 
                                            annealing_time: float, 
                                            max_num_reads_per_job: int,
                                            max_qpu_time_per_job: int,
                                            config: Config):
        """ 
        calculates the number of reads that are possible to run within a timelimit
        # results from the equation from https://docs.dwavequantum.com/en/latest/quantum_research/operation_timing.html
        # qpu_access_time = preprocessing_time + num_of_reads * total_sampling_time
        # we create two equations with two estimation results and solve the equation system
        # returns a list of the number of reads in the batches
        """
        assert max_num_reads_per_job > 1 and isinstance(max_num_reads_per_job, int)
        assert max_qpu_time_per_job > 1
        
        solver_api_tokens_path = config.BASE_PATH_CONFIG + "\\config_files\\" + "solver_api_tokens.json"
        with open(solver_api_tokens_path, "r") as json_file:
            solver_api_tokens_data = json.load(json_file)
        dwave_token = solver_api_tokens_data['Dwave_Token']
        dwave_processor = solver_api_tokens_data['Dwave_QuantumComputer']
        
        client = Client(token=dwave_token)
        solver = client.get_solver(name=dwave_processor)
        
        num_of_qubits = len(solver.nodes)
        estimated_qputime_1_read = solver.estimate_qpu_access_time( # in microseconds
            num_qubits = num_of_qubits,
            num_reads = 1,
            annealing_time = annealing_time)
        
        # determine maximum number of reads so that max qpu time per job is not exceeded
        estimated_qputime_xxx_reads = 100000000 #initialize in microseconds
        max_num_reads = max_num_reads_per_job * 2 #initialize --> will possibly get smaller
        while estimated_qputime_xxx_reads > max_qpu_time_per_job:
            max_num_reads /= 2
            estimated_qputime_xxx_reads = solver.estimate_qpu_access_time( # in microseconds
                num_qubits = num_of_qubits,
                num_reads = max_num_reads,
                annealing_time = annealing_time)
        
        # determine the blocks with their number of reads
        timelimit_in_microseconds = timelimit_in_min * 60 * 1000000
        if estimated_qputime_xxx_reads < timelimit_in_microseconds:
            number_of_full_blocks = timelimit_in_microseconds // estimated_qputime_xxx_reads
            batches_of_reads = [int(max_num_reads) for i in range(int(number_of_full_blocks))]
            # last_batch = int((timelimit_in_microseconds % estimated_qputime_xxx_reads) / timelimit_in_microseconds * max_num_reads)
            # batches_of_reads.append(last_batch)
            
            return batches_of_reads
        else:
            total_sampling_time = (estimated_qputime_xxx_reads - estimated_qputime_1_read) / (max_num_reads - 1)
            preprocessing_time = estimated_qputime_1_read - total_sampling_time
            
            optimal_num_reads = (60 * 1000000 * timelimit_in_min - preprocessing_time) / total_sampling_time
            
            return [int(optimal_num_reads)]
        
        
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
            create_barplot_obj_values_with_number_of_violated_constraints(result_dict_feasibility_analysed, config.EXPORT_PATH, "barplot_obj_values_QuantumAnneal")
        feas_sol_dict, feas_sol_df = get_all_feasible_solutions(result_dict_feasibility_analysed, variable_names_list, report)
        if config.report_config["save_feasible_sols_to_excel"] == True:
            feas_sol_df.to_excel(os.path.join(config.EXPORT_PATH, "feasible_solutions.xlsx"), index=False)
        write_kpis_to_report(feas_sol_dict, self.num_reads, config, report)
        
        # %% potentially postprocess
        if config.solve_method_config['postprocess_with_steepest_descent'] == True:
            feas_sol_dict, _ = execute_steepest_descent_on_bitstrings(self.opt_problem, feas_sol_dict, config, report)
        
        # %% expectation value comparison with random sampling
        all_energies = []
        for dwave_response in solver_result:
            all_energies.extend(list(dwave_response.data_vectors['energy']))
        all_obj_values = [energy + self.qubo.objective.constant for energy in all_energies]
        expec_value_qua_anneal = np.mean(all_obj_values)
        expec_value_rand_sampling = calculate_expectation_value_random_sampling_qubo(self.qubo_dict, self.qubo.objective.constant)
        report.solution_quality['expectation_obj_value_quantum_anneal'] = expec_value_qua_anneal
        report.solution_quality['expectation_obj_value_random_sampling'] = expec_value_rand_sampling
        # %%
        report.time_measurements['opt_method_result_analysis_total_time_consumption'] = time.time() - start_time
        return feas_sol_dict, result_dict_feasibility_analysed
    

def retrieve_dwave_solver_result(problem_id: str):
    
    solver_api_tokens_path = Config.BASE_PATH_CONFIG + "\\config_files\\" + "solver_api_tokens.json"
    with open(solver_api_tokens_path, "r") as json_file:
        solver_api_tokens_data = json.load(json_file)
    dwave_token = solver_api_tokens_data['Dwave_Token']
    
    with Client(token=dwave_token) as client:
        future = client.retrieve_answer(problem_id)
        result = future.result()
    
if __name__ == '__main__':
    retrieve_dwave_solver_result(
        problem_id = '0685aa49-50d1-42ed-9961-adccd7cc12be'
        )