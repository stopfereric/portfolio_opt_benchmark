# -*- coding: utf-8 -*-
"""
Created on 07.03.2024

@author: Eric Stopfer
"""
import time
import pdb
import os
import json
import copy
import math
import pickle
import numpy as np
from matplotlib import pyplot as plt

from qiskit import QuantumCircuit
# from qiskit.compiler import transpile
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator # Basic Quantum Simulator
# from mqt.ddsim import DDSIMProvider # other Quantum Simulator from the Munich Quantum Toolkit
from qiskit_aer.noise import NoiseModel # for Noisy Quantum Simulation
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler # for real Quantum device
from scipy.optimize import minimize

from config.config import Config
from src.report import Report
from src.utils.error_utils import CustomizedError
from src.utils.callback_utils import Callback_StopScipyMinimizer, OptimizationTimeout
from src.utils.visualisation_utils import visualize_qaoa_parameter_optimization_process, visualize_qaoa_distribution_of_energies, create_barplot_obj_values_with_number_of_violated_constraints
from src.utils.solver_result_utils import analyse_feasibility_of_solver_solutions, get_all_feasible_solutions, write_kpis_to_report
from src.utils.opt_utils import execute_steepest_descent_on_bitstrings
from src.utils.qc_utils import calc_ising_energy_of_bitstring_array


class QAOA:
    '''
    ----- from Pelofske, Bärtschi, Eidenbenz (https://www.nature.com/articles/s41534-024-00825-w) -----
    
    The Quantum Alternating Operator Ansatz (QAOA) is a hybrid quantum-classical algorithm 
    for sampling combinatorial optimization problems, the parameterized quantum component 
    of which is executed on a programmable gate-based universal quantum computer. The
    Quantum Approximate Optimization Algorithm is the original algorithm of this type, 
    which was then generalized to the Quantum Alternating Operator Ansatz algorithm
    under the same acronym of QAOA. The classical component of QAOA involves learning 
    the best parameters of the circuit to obtain low energy solutions of the combinatorial 
    optimization problem.
    '''
    def __init__(self, config: Config, report: Report, ibm_session = None, abort_code_after_job_submission = False):
        """
        Function that creates a QAOA solver instance (-> Quantum Approximate Optimization Algorithm)
        """
        self.config = config
        self.read_config()
        self.ibm_session = ibm_session
        self.abort_code_after_job_submission = abort_code_after_job_submission
        self.ibm_job_ids = []
        
        #initialize stuff that will be filled during QAOA run
        self.qaoa_results = [] 
        self.number_of_reads = 0
        
        #get the sampling device
        self.device_name_sampling = config.solve_method_device
        self.device_sampling = self.get_device(self.device_name_sampling, config, report, self.ibm_session)
        
        #get the training device
        if config.solve_method_config["training_with_local_simulator"] == 1 :
            self.device_name_training = "Local_Simulator"
        else:
            self.device_name_training = self.device_name_sampling
        self.device_training = self.get_device(self.device_name_training, config, report)
                
        config.logger.info("we are using the following solving method: %s" %config.solve_method)
        config.logger.info("with the following device for training: %s" %self.device_name_training)
        config.logger.info("with the following device for sampling: %s" %self.device_name_sampling)
        
    def read_config(self):
        try:
            self.circ_depth = self.config.solve_method_config['number_of_layers']  # circuit depth for QAOA
            self.shots_for_sampling = self.config.solve_method_config['shots_for_sampling']
            self.optimize_params_classically = self.config.solve_method_config['optimize_params_classically']
            self.opt_method = self.config.solve_method_config['opt_method']  # SLSQP, COBYLA, Nelder-Mead, BFGS, Powell, ...
            self.training_with_local_simulator = self.config.solve_method_config['training_with_local_simulator']
            self.shots_for_training = self.config.solve_method_config['shots_for_training']
            self.optimize_params_with_grid_search  = self.config.solve_method_config['optimize_params_with_grid_search']
            self.grid_search_size = self.config.solve_method_config['grid_search_size'] # depth of the grid of QAOA parameters
            self.sample_with_linearramp_without_training = self.config.solve_method_config['sample_with_linearramp_without_training']
            self.sample_randomly = self.config.solve_method_config['sample_randomly']
            self.max_calculation_time_exists = self.config.solve_method_config['max_calculation_time_exists']
            self.max_training_time_in_sec = self.config.solve_method_config['max_training_time_in_sec']
            self.max_sampling_time_in_sec = self.config.solve_method_config['max_sampling_time_in_sec']
        except:
            self.config.logger.error("wrong or missing configuration of the QAOA.")
            raise CustomizedError(error_message="wrong or missing configuration of the QAOA. check in config.py and config.json")
        
    
    def get_device(self, device_name: str, config: Config, report: Report, ibm_session = None) -> any:
        '''
        Function that gets the QAOA device by the name given in the configuration
        '''
        start_time = time.time()
        if device_name == "Local_Simulator":
            device = AerSimulator()
            report.time_measurements['connect_to_local_simulator_Aer'] = time.time() - start_time
            
        elif device_name == "Noisy_Local_Simulator":
            solver_api_tokens_path = config.BASE_PATH_CONFIG + "\\config_files\\" + "solver_api_tokens.json"
            with open(solver_api_tokens_path, "r") as json_file:
                solver_api_tokens_data = json.load(json_file)
            ibm_token = solver_api_tokens_data['IBM_Token']
            if ibm_token == "type_your_ibm_api_token_here": # default value
                raise CustomizedError("for IBM Quantum Computer a valid API token is necessary. it has to be put inside the file: \n C:\...\quopt\config\config_files\solver_api_tokens.json")
            service = QiskitRuntimeService(channel='ibm_quantum', token=ibm_token)
            quantum_backend = service.least_busy(operational=True, simulator=False)
            device = NoiseModel.from_backend(quantum_backend)
            report.time_measurements['connect_to_noisy_simulator'] = time.time() - start_time
            
        # elif device_name == "MQT_Simulator":
        #     device = DDSIMProvider().get_backend("qasm_simulator")
        #     report.time_measurements['connect_to_local_simulator_MQT'] = time.time() - start_time
            
        elif device_name == "IBM_Quantum_Computer":
            config.logger.info("Connecting to IBM Quantum Computer...")
            #get the ibm token
            solver_api_tokens_path = config.BASE_PATH_CONFIG + "\\config_files\\" + "solver_api_tokens.json"
            with open(solver_api_tokens_path, "r") as json_file:
                solver_api_tokens_data = json.load(json_file)
            ibm_token = solver_api_tokens_data['IBM_Token']
            ibm_backend_name = solver_api_tokens_data['IBM_QuantumComputer']
            if ibm_token == "type_your_ibm_api_token_here": # default value
                raise CustomizedError("for IBM Quantum Computer a valid API token is necessary. it has to be put inside the file: \n C:\...\quopt\config\config_files\solver_api_tokens.json")
            service = QiskitRuntimeService(channel='ibm_quantum', token=ibm_token)
            if ibm_session != None:
                device = Sampler(mode=ibm_session)  
                backend_name = ibm_session.details()['backend_name']
                self.backend = service.backend(name=backend_name)
                config.logger.info("using IBM Session")
            else:
                # self.backend = service.least_busy(operational=True, simulator=False)
                self.backend = service.backend(name=ibm_backend_name)
                device = Sampler(mode=self.backend)  
            config.logger.info("\n As the Quantum device for the QAOA the following backend is used: " + self.backend.name)
            report.time_measurements['connect_to_IBM_QC'] = time.time() - start_time
            
        else:
            config.logger.error("Wrong configuration of the device. Check in config.py or config.json")
            raise CustomizedError(error_message="Wrong configuration of the device. Check in config.py or config.json")
            
        return device
    
        
    def run(self, opt_problem, config: Config, report: Report, early_run_exit=False) -> (any, any, any):
        """
        Function that executes the QAOA algorithm on a certain problem on a 
        certain device for a certain configuration
        """
        start_time_overall = time.time()
        # Ising-problem objective can be gotten with obj = x J x^T + h x + c
        self.opt_problem = opt_problem
        self.qubo_dict = opt_problem.problem_mapping[0]
        self.qubo = opt_problem.problem_mapping[1]
        self.mip_for_feasibility_analysis = opt_problem.problem_mapping[2]
        self.ising_matrix = opt_problem.problem_mapping[3]
        self.ising_vector = opt_problem.problem_mapping[4]
        self.ising_offset = opt_problem.problem_mapping[5]
        self.ising = (self.ising_matrix, self.ising_vector, self.ising_offset)
        
        if early_run_exit == True:
            return
        
        if self.optimize_params_classically == True:
            # %% run QAOA classical parameter optimization
            start_time = time.time()
            # create a QAOA tracker to keep track of results
            qaoa_tracker = {
                'count': 1,  # Elapsed optimization steps in the training
                'training_start_time': time.time(), # start time of the training
                'per_cycle_energy_expectation': [], # energy expectation at each step
                'per_cycle_params': [], # state of the parameters (-->beta, gamma) after each iteration
                'time_passed_after_cycle': [], # contains times passed in the training cycles since the training start 
                } 
                    
            # train the parameters beta and gamma
            params_qaoa, qaoa_tracker \
                = self.train(self.device_training, 
                             self.circ_depth, 
                             self.ising, 
                             self.shots_for_training, 
                             self.opt_method, 
                             qaoa_tracker)
            report.time_measurements['qaoa_parameter_training'] = time.time() - start_time
            report.solution_quality['qaoa_parameters'] = params_qaoa
            
            # visualize the parameter optimization process
            visualize_qaoa_parameter_optimization_process(qaoa_tracker['per_cycle_energy_expectation'], config.EXPORT_PATH, "qaoa_classical_parameter_optimization_process_of_expectation_value")
        
            # sample QAOA circuit with classically trained parameters to hopefully get good solutions
            start_time = time.time()
            qaoa_sampling_results, qc_or_not = self.sample_qaoa_circuit(params_qaoa, self.device_sampling, self.shots_for_sampling, self.ising, self.abort_code_after_job_submission)
            report.time_measurements['qaoa_sampling'] = time.time() - start_time
            for counter, result in enumerate(qaoa_sampling_results):
                self.qaoa_results.append((f"QAOA_with_classical_param_opt_sample_{counter}", result, qc_or_not))
            
        if self.optimize_params_with_grid_search == True: 
            # %% run QAOA grid search parameter optimization for circuit depth p=1
            start_time = time.time()
            if self.max_calculation_time_exists == True:
                # train the parameters beta and gamma in a certain time limit
                grid_search_beta, grid_search_gamma \
                    = self.grid_search_qaoa_one_layer(self.grid_search_size, self.ising, config.EXPORT_PATH, config, self.max_training_time_in_sec)
            else:    
                # train the parameters beta and gamma without a time limit
                grid_search_beta, grid_search_gamma \
                    = self.grid_search_qaoa_one_layer(self.grid_search_size, self.ising, config.EXPORT_PATH, config)
            params_grid_search = [grid_search_gamma, grid_search_beta]
            report.time_measurements['qaoa_grid_search'] = time.time() - start_time
            
            # grid search circuit sampling --> run the QAOA circuit with the best parameters from the grid search
            start_time = time.time()
            grid_search_sampling_results, qc_or_not = self.sample_qaoa_circuit(params_grid_search, self.device_sampling, self.shots_for_sampling, self.ising, self.abort_code_after_job_submission)
            report.time_measurements['qaoa_grid_search_sampling'] = time.time() - start_time
            for counter, result in enumerate(grid_search_sampling_results):
                self.qaoa_results.append((f"QAOA_with_grid_search_param_opt_sample_{counter}", result, qc_or_not))
                        
        if self.sample_with_linearramp_without_training == True:
            # %% sampling with linear ramp schedule without parameter training
            start_time = time.time()
            # initialize gamma and beta with formula from Linear-Ramp-QAOA-Paper: http://arxiv.org/abs/2405.09169
            Delta_gamma = 0.6
            Delta_beta = 0.3
            gamma_initial = [((i+1) / self.circ_depth) * Delta_gamma for i in range(self.circ_depth)]
            beta_initial = [(1 - i/self.circ_depth) * Delta_beta for i in range(self.circ_depth)]
            lr_qaoa_params = gamma_initial + beta_initial
            lr_qaoa_results, qc_or_not = self.sample_qaoa_circuit(lr_qaoa_params, self.device_sampling, self.shots_for_sampling, self.ising, self.abort_code_after_job_submission)
            report.time_measurements['qaoa_lr_qaoa_sampling'] = time.time() - start_time
            for counter, result in enumerate(lr_qaoa_results):
                self.qaoa_results.append((f"QAOA_with_LinearRamp_sampling_sample_{counter}", result, qc_or_not))
            
        if self.sample_randomly == True:
            # %% random sampling: run the QAOA with parameters [0, 0] --> no rotations
            start_time = time.time()
            random_sampling_results, qc_or_not = self.sample_qaoa_circuit([0, 0], self.device_sampling, self.shots_for_sampling, self.ising, self.abort_code_after_job_submission)
            report.time_measurements['qaoa_random_sampling'] = time.time() - start_time
            for counter, result in enumerate(random_sampling_results):
                self.qaoa_results.append((f"QAOA_with_random_sampling_sample_{counter}", result, qc_or_not))
            
        # %% end of QAOA            
        config.logger.info("The opt-problem-solving with QAOA has finished. \n")
        report.time_measurements['opt_method_run_total_time_consumption'] = time.time() - start_time_overall
        return self.qaoa_results

    
    def train(self, 
              device: any, 
              circ_depth: int, 
              ising: tuple, 
              n_shots: int, 
              opt_method: str, 
              tracker: dict,
              parameter_initialization: str='LR-QAOA-formula'
              ) -> ((float, float), dict):
        """
        function that trains the circuit parameters of a QAOA circuit corresponding 
        to an ising formulation with a certain optimization method
        """
        self.config.logger.info("Starting the training of the parameters of the QAOA circuit.")
    
        if parameter_initialization == 'random':
            # randomly initialize variational parameters within appropriate bounds
            gamma_initial = np.random.uniform(0, 2 * math.pi, circ_depth).tolist()
            beta_initial = np.random.uniform(0, 1/2 * math.pi, circ_depth).tolist()
        elif parameter_initialization == 'LR-QAOA-formula':
            # initialize gamma and beta with formula from Linear-Ramp-QAOA-Paper: http://arxiv.org/abs/2405.09169
            Delta_gamma = 0.6
            Delta_beta = 0.3
            gamma_initial = [((i+1) / circ_depth) * Delta_gamma for i in range(circ_depth)]
            beta_initial = [(1 - i/circ_depth) * Delta_beta for i in range(circ_depth)]
        else:
            raise Exception(f"unknown parameter_initialization method: '{parameter_initialization}'")
        
        params_start = np.array(gamma_initial + beta_initial)
        tracker["params_start"] = params_start
    
        # set bounds for search space
        bnds_gamma = [(0, 2 * math.pi) for _ in range(int(len(params_start) / 2))]
        bnds_beta = [(0, 1/2 * math.pi) for _ in range(int(len(params_start) / 2))]
        bnds = bnds_gamma + bnds_beta
        
        # run classical optimization (example: method='Nelder-Mead')
        if self.max_calculation_time_exists == True:
            try:
                result = minimize(
                    self.objective_function_energyexpect,
                    params_start,
                    args=(device, ising, n_shots, tracker),
                    method=opt_method,
                    bounds=bnds,
                    options={"disp": True},
                    callback=Callback_StopScipyMinimizer(self.max_training_time_in_sec)
                    )
                # store result parameters of classical optimization
                result_params = result.x.tolist()
                tracker["params_optimal"] = result_params
            except OptimizationTimeout:
                result_params = tracker['per_cycle_params'][-1].tolist()
                self.config.logger.info(f"finished scipy optimizer after exceeding time-limit of {self.max_training_time_in_sec} ")
        else:
            result = minimize(
                self.objective_function_energyexpect,
                params_start,
                args=(device, ising, n_shots, tracker),
                method=opt_method,
                bounds=bnds,
                options={"disp": True}
                )
            # store result parameters of classical optimization
            result_params = result.x.tolist()
            tracker["params_optimal"] = result_params
        
        self.config.logger.info("Training of QAOA parameters finished.")
        self.config.logger.info(f"Classical optimizer {opt_method} terminated with parameters {result_params}\n")
        
        self.time_taken_during_training = tracker["time_passed_after_cycle"][-1]
        return result_params, tracker
    
    
    def objective_function_energyexpect(self, 
        params: list, 
        device: any, 
        ising: tuple, 
        n_shots: int, 
        tracker: dict
        ) -> float:
        """
        objective function takes a list of variational parameters as input,
        builds a QAOA circuit, runs this circuit n_shots times and returns
        the mean energy value of a shot.
        """
        qaoa_circuit = self.create_qaoa_circuit(params, ising)
        
        if self.training_with_local_simulator == False:
            exec_on_quantum_device = True
            device_type = "QC"
        else:
            exec_on_quantum_device = False
            device_type = "QC_Simulator"
        
        #execute the circuit
        results = self.run_circuit(device, qaoa_circuit, n_shots, exec_on_quantum_device)
        result = results[0] #take first result sample
        self.qaoa_results.append((f"paramtraining_cycle_{tracker['count']}", result, device_type))
        
        if exec_on_quantum_device == True:
            counts = result[0].data.meas.get_counts()
        else:
            counts = dict(result.get_counts())
        
        # calculate the energy expectation
        energy_sum = 0
        for bitstring, count in counts.items():
            # convert results (0 and 1) to ising (1 and -1)
            ising_bitstring = np.array([-1 if bit == '1' else 1 for bit in bitstring])
            # invert the bitstring because qiskit outputs inverted bitstrings
            ising_bitstring_inverted = np.flip(ising_bitstring)
            # add the ising energy of the bitstring to the energy_sum
            energy_sum += count * calc_ising_energy_of_bitstring_array(ising[0], ising[1], ising[2], ising_bitstring_inverted)
        energy_expectation = energy_sum / n_shots
        
        #update the tracker
        tracker["per_cycle_energy_expectation"].append(energy_expectation)
        cycle_number = tracker["count"] + 1
        tracker.update({"count": cycle_number, "res": result})
        tracker["per_cycle_params"].append(params)
        tracker["time_passed_after_cycle"].append(time.time() - tracker["training_start_time"])
        
        #log the calculation state
        self.config.logger.info(f"Energy expectation value during cycle {cycle_number-1}: {energy_expectation} \n")
    
        return energy_expectation
    

    def create_qaoa_circuit(self, params: list, ising: tuple) -> QuantumCircuit:
        """
        function that returns a full QAOA qiskit-circuit for an ising and certain 
        beta- and gamma-parameters 
        """        
        # initialize empty circuit
        n_qubits = ising[0].shape[0]
        qaoa_circ = QuantumCircuit(n_qubits)
        
        # add Hadamard to all qubits to initialize the |+> state
        for i in range(n_qubits):
            qaoa_circ.h(i)
        
        # setup two parameter families
        circ_depth = int(len(params) / 2)
        gammas = params[:circ_depth]
        betas = params[circ_depth:]
    
        # add QAOA circuit layer blocks
        for layer in range(circ_depth):
            #add a problem_unitary-circuit
            qaoa_circ.append(self.problem_unitary(gammas[layer], ising), range(n_qubits))
            #add a mixing_unitary-circuit
            qaoa_circ.append(self.mixing_unitary(betas[layer], n_qubits), range(n_qubits))
        
        # add measurements
        qaoa_circ.measure_all()
        
        #decompose the internal circuits
        qaoa_circ = qaoa_circ.decompose()
        
        return qaoa_circ
    
        
    def run_circuit(self, 
                    device: any, 
                    qaoa_circuit: QuantumCircuit, 
                    n_shots: int,
                    exec_on_quantum_device=False,
                    time_limit=None,
                    abort_code_after_job_submission=False
                    ) -> any:
        """
        function to run a QAOA circuit on a device with n shots, possibly on a 
        real gate-based quantum computer
        """ 
        results = [] #initialize
        
        if exec_on_quantum_device == True:
            circuit_depth_before = qaoa_circuit.depth()
            circuit_numqubits_before = qaoa_circuit.num_qubits
            
            #transpile the circuit so that it fits the hardware
            self.config.logger.info("Transpiling the circuit for the hardware ...")
            pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)  #0: no opt, 1: light opt, 2: heavy opt, 3: really heavy opt
            qaoa_circuit = pm.run(qaoa_circuit)
            
            circuit_depth_after = qaoa_circuit.depth()
            circuit_numqubits_after = qaoa_circuit.num_qubits            
            
            self.config.logger.info("Finished transpiling.")
            self.config.logger.info(f"Depth went from {circuit_depth_before} to {circuit_depth_after}")
            self.config.logger.info(f"Number of qubits went from {circuit_numqubits_before} to {circuit_numqubits_after}\n")
            
        # possibly adjust number of shots
        if time_limit != None:
            if exec_on_quantum_device:
                n_shots = self.get_optimal_number_of_shots_for_max_execution_time(time_limit, qaoa_circuit)
                
                ####################################
                # TODO should be deleted later, adjustments are based on real experiments for portfolioopt,
                n_shots_adjustment_dict = {3: 55/50, 5: 55/36, 7: 55/30, 10: 55/26, 15: 55/24, 20: 55/23, 25: 55/22, 30: 55/21} 
                if circuit_numqubits_before / 4 in n_shots_adjustment_dict.keys():
                    n_shots = int(n_shots * n_shots_adjustment_dict[circuit_numqubits_before / 4])
                ####################################
                
                # execute the circuit on the device
                start_time = time.time()
                self.config.logger.info(f"Start running the circuit {n_shots} times...")
                job = device.run([qaoa_circuit], shots=n_shots)
                job_id = job.job_id()
                self.ibm_job_ids.append(job_id)
                self.config.logger.info(f"Job-ID: {job_id}")
                self.config.logger.info(f"Job-Status: {job.status()}")
                
                if abort_code_after_job_submission == True:
                    self.config.logger.info("Further analysis will be aborted, because job was already submitted to IBM QC")
                    return [] #empty results list
                else:
                    results.append(job.result())
                    self.config.logger.info("Finished sampling.")
                    self.config.logger.info(f"Job-Usage in seconds: {job.usage()}")
                    return results
            else:
                # determine the batch size of sampling to not exceed the time limit
                time_elapsed_during_test = 0
                num_shots_for_batches = 1
                counter = 0
                while time_elapsed_during_test < 4:
                    time_meas = time.time()
                    num_shots_for_batches *= 2
                    testjob = device.run([qaoa_circuit], shots=num_shots_for_batches)
                    test_result = testjob.result()
                    time_elapsed_during_test = time.time() - time_meas
                    counter += 1                
                self.config.logger.info(f"adjusted the number of reads per batch to {num_shots_for_batches} by doing {counter} method executions")
                
                self.config.logger.info(f"Start running the circuit in batches of {num_shots_for_batches} shots for {time_limit}s ...")
                # run the batches until the timelimit has expired
                start_time = time.time()
                while time.time() - start_time <= time_limit:
                    job = device.run([qaoa_circuit], shots=num_shots_for_batches)
                    results.append(job.result())
                    self.config.logger.info("finished a batch of QAOA")
                return results
            
        else:
            # execute the circuit on the device
            self.config.logger.info(f"Start running the circuit {n_shots} times...")
            job = device.run([qaoa_circuit], shots=n_shots)
            results.append(job.result())
            return results

    
    def get_optimal_number_of_shots_for_max_execution_time(self, timelimit, qaoa_circuit):
        ''' 
        returns the optimal number of shots on a real IBM QC for a given timelimit
        '''
        from qiskit.converters import circuit_to_dag

        dag = circuit_to_dag(qaoa_circuit)
        layers = list(dag.layers())
        
        total_duration_circuit = 0
        for layer in layers:
            max_gate_duration_in_layer = 0
            for node in layer["graph"].op_nodes():
                try:
                    instr = node.name
                    qubits = tuple(q._index for q in node.qargs)
                    duration = self.backend.target[instr][qubits].duration
                    if duration > max_gate_duration_in_layer:
                        max_gate_duration_in_layer = duration
                except Exception:
                    self.config.logger.info(f"error with {instr}")
            total_duration_circuit += max_gate_duration_in_layer
            
        # formula by https://quantum.cloud.ibm.com/docs/de/guides/estimate-job-run-time
        optimal_number_of_shots = (timelimit - 2) / (0.00025 + total_duration_circuit) 
        
        return int(optimal_number_of_shots)
        
    
    def problem_unitary(self, gamma: float, ising: tuple) -> QuantumCircuit:
        """
        function that returns a qiskit-circuit for evolution with problem Hamiltonian
        for a certain ising formulation 
        -->applies rzz-gates for ising matrix and rz-gates for ising vector
        """
        # initialize circuit object
        n_qubits = ising[0].shape[0]
        circ_problem = QuantumCircuit(n_qubits)
        
        # apply rzz-gate for every non-zero element in the ising matrix (with corresponding interaction strength)
        ising_matrix = ising[0]
        for (qubit1, qubit2), interaction_strength in np.ndenumerate(ising_matrix):
            if interaction_strength != 0:
                circ_problem.rzz(2 * interaction_strength * gamma, qubit1, qubit2) # R_zz-gate
        
        # apply rz-gate for every non-zero element in the ising vector (with corresponding interaction strength)
        ising_vector = ising[1]
        for qubit1, interaction_strength in enumerate(ising_vector):
            if interaction_strength != 0:
                circ_problem.rz(2 * interaction_strength * gamma, qubit1) # R_z-rotation
         
        return circ_problem
    
    
    def mixing_unitary(self, beta: float, n_qubits: int) -> QuantumCircuit:
        """
        Function that returns a qiskit-circuit for mixing Hamiltonian 
        -->applies rx-gates on all qubits       
        """
        # initialize circuit object
        circ_mixing = QuantumCircuit(n_qubits)
        
        # apply rx-gates to every qubit
        for qubit in range(n_qubits):
            circ_mixing.rx(2 * beta, qubit)
    
        return circ_mixing
    
        
    def sample_qaoa_circuit(self, params: list, device: any, n_shots: int, ising: tuple, abort_code_after_job_submission=False) -> any:
        """
        function that samples a qaoa circuit with the goal to get good solutions for a given ising
        """
        self.config.logger.info(f"Start to sample QAOA circuit with parameters {params}")
        # create the circuit with a beta and gamma
        qaoa_circuit = self.create_qaoa_circuit(params, ising)
        
        # run the circuit   
        if self.device_name_sampling in ['IBM_Quantum_Computer']:
            exec_on_quantum_device = True
            device_type = "QC"
        else:
            exec_on_quantum_device = False
            device_type = "QC_Simulator"
        
        if self.max_calculation_time_exists:
            if hasattr(self, 'time_taken_during_training'):
                time_limit = self.max_sampling_time_in_sec + max(0, (self.max_training_time_in_sec - self.time_taken_during_training))
            else:
                time_limit = self.max_sampling_time_in_sec
            results = self.run_circuit(device, qaoa_circuit, n_shots, exec_on_quantum_device, time_limit, abort_code_after_job_submission)    
        else:
            results = self.run_circuit(device, qaoa_circuit, n_shots, exec_on_quantum_device, None, abort_code_after_job_submission)
        self.config.logger.info("Finished sampling the QAOA circuit / submitting the IBM-Job \n")
        return results, device_type
    
    
    def grid_search_qaoa_one_layer(self, grid_size: int, ising: any, store_dir: str, config: Config, max_calc_time=None) -> (float, float):
        '''
        function that tries to find optimal qaoa parameters for circuit depth p=1 
        corresponding to an ising-problem-formulation via a grid search
        '''
        config.logger.info("Start of grid search for QAOA parameters of circuit depth 1")
        
        if max_calc_time != None:
            start_time = time.time()
            testset_size = 10
            for _ in range(testset_size):
                energy_expectation = self.get_energy_expectation_of_qaoa_circuit_depth_1(ising, 1, 1)
                time_took = (time.time() - start_time + 0.0001) / testset_size
            grid_size = max(1, math.floor(math.sqrt(max_calc_time / time_took)))
            config.logger.info(f"grid search size adjusted to {grid_size}x{grid_size}")
        
        #initialize a numpy-array from which the grid graphic will be filled
        grid_results = np.zeros((grid_size, grid_size)) 
        range_beta = math.pi / 2
        range_gamma = 2 * math.pi
        
        calc_state_old = 0
        for beta_index in range(grid_size):
            beta = beta_index * ( range_beta / grid_size )
            
            for gamma_index in range(grid_size):
                gamma = gamma_index * ( range_gamma / grid_size )
                
                energy_expectation = self.get_energy_expectation_of_qaoa_circuit_depth_1(ising, beta, gamma)
                grid_results[beta_index, gamma_index] = energy_expectation
                            
                #log the state of the grid search
                calc_state_new = round(100 * (beta_index * grid_size + gamma_index + 1) / (grid_size**2), 0)
                if calc_state_new != calc_state_old:
                    calc_state_old = calc_state_new
                    config.logger.info("State of the grid search for beta and gamma: " + str(calc_state_old) + "%")
                                
        #create a heatmap graphic and save this graphic to the export-path
        plt.figure(figsize=(20,5))
        plt.imshow(grid_results, cmap='viridis', extent=[0, range_gamma, 0, range_beta])
        plt.colorbar(label = 'Expectation value of objective')
        plt.ylabel('Beta')
        plt.xlabel('Gamma')
        plt.title('Result of Gridsearch')
        plt.savefig(store_dir + "\gridsearch_heatmap_for_circuit_depth_one.png")
        plt.clf()
        
        # find the index with minimum value in grid_results
        pickle.dump(grid_results, open(store_dir + "\grid_results.pkl", 'wb'))
        (beta_optimal_index, gamma_optimal_index) = np.unravel_index(np.argmin(grid_results), grid_results.shape)
        grid_search_beta = beta_optimal_index * ( range_beta / grid_size )
        grid_search_gamma = gamma_optimal_index * ( range_gamma / grid_size )
        
        config.logger.info("Grid search for QAOA parameters has finished.")
        config.logger.info(f"Grid search terminated with parameters {grid_search_gamma}, {grid_search_beta}\n")
    
        return grid_search_beta, grid_search_gamma
    
    
    def get_energy_expectation_of_qaoa_circuit_depth_1(self, ising: tuple, beta: float, gamma: float) -> float:
        '''
        function that calculates the energy expectation value of a qaoa circuit
        with depth 1 only by its ising and beta/gamma-parameters.
        It was derived from formula 13 from the following paper:
        https://iopscience.iop.org/article/10.1088/2058-9565/ac9013/pdf
        ''' 
        #initialize the expectation value, things will be added
        expectation = 0 
        
        ising_mat = ising[0] #ising-matrix
        ising_vec = ising[1] #ising-vector
        ising_offset = ising[2] #ising-offset
        
        n_qubits = ising_mat.shape[0]
        assert (ising_mat == np.triu(ising_mat, k=1)).all() #check if the ising matrix is an upper diagonal matrix
        ising_mat = np.transpose(ising_mat) + ising_mat # the formula was defined for edges ij on a graph, so we put all the values above the diagonal also under the diagonal
        
        #iterate through ising-matrix
        for coords_mat, coeff_mat in np.ndenumerate(ising_mat):
            if coeff_mat != 0 and coords_mat[0]<coords_mat[1]:
                term_to_add_part1 = (coeff_mat * math.sin(4*beta)) / 2 * math.sin(2*gamma*coeff_mat)
                
                term_to_add_part2 = math.cos(2*gamma*ising_vec[coords_mat[0]])
                for idx in range(n_qubits):
                    if idx not in coords_mat: # k != i,j
                        term_to_add_part2 *= math.cos(2*gamma*ising_mat[coords_mat[0], idx])
                
                term_to_add_part3 = math.cos(2*gamma*ising_vec[coords_mat[1]])
                for idx in range(n_qubits):
                    if idx not in coords_mat: # k != i,j
                        term_to_add_part3 *= math.cos(2*gamma*ising_mat[coords_mat[1], idx])
                
                term_to_add_part4 = coeff_mat / 2 * (math.sin(2*beta))**2
                
                term_to_add_part5 = math.cos(2*gamma*(ising_vec[coords_mat[0]] + ising_vec[coords_mat[1]]))
                
                term_to_add_part6 = 1 #initial product value
                for idx in range(n_qubits):
                    if idx not in coords_mat:  # k != i,j
                        term_to_add_part6 *= math.cos(2*gamma*(ising_mat[coords_mat[0], idx] + ising_mat[coords_mat[1], idx]))
                
                term_to_add_part7 = math.cos(2*gamma*(ising_vec[coords_mat[0]] - ising_vec[coords_mat[1]]))
                
                term_to_add_part8 = 1 #initial product value
                for idx in range(n_qubits):
                    if idx not in coords_mat:  # k != i,j
                        term_to_add_part8 *= math.cos(2*gamma*(ising_mat[coords_mat[0], idx] - ising_mat[coords_mat[1], idx]))
                                
                expectation += term_to_add_part1 * (term_to_add_part2 + term_to_add_part3) - \
                               term_to_add_part4 * (term_to_add_part5 * term_to_add_part6 - term_to_add_part7 * term_to_add_part8)
            else:
                continue
            
        #iterate through ising-vector
        for coords_vec, coeff_vec in np.ndenumerate(ising_vec):
            if coeff_vec != 0:
                
                term_to_add_part9 = coeff_vec * math.sin(2*beta) * math.sin(2*gamma*coeff_vec)
                
                term_to_add_part10 = 1 #initial product value
                for idx in range(n_qubits):
                    if idx not in coords_vec: # k != i
                        term_to_add_part10 *= math.cos(2*gamma*ising_mat[coords_vec[0], idx])                    
                        
                expectation += term_to_add_part9 * term_to_add_part10
            else:
                continue
        
        #add the ising-constant
        expectation += ising_offset
        
        return expectation

        
    def analyse_solver_result(self, solver_result, config: Config, report: Report) -> (dict, dict):
        ''' 
        function to analyse the result of the QAOA solver
        '''
        start_time = time.time()
        # first let's get the original variable names that were in the MIP
        variable_names_list = [] 
        for variable in self.mip_for_feasibility_analysis.variables:
            variable_names_list.append(variable.name)
        
        #merge all the sampling results into 1 dict
        sampling_counts_overall = {}
        for (result_name, sampling_result, qc_or_not) in solver_result:
            config.logger.info(f"start to get the counts of the qaoa sampling results of {result_name}")
            sampling_counts = self.get_qiskit_qaoa_counts(sampling_result, qc_or_not)
            sampling_counts_overall = self.merge_qaoa_count_dicts(
                sampling_counts_overall, sampling_counts, result_name)
        
        #feasibility analysis
        config.logger.info("start to process the qaoa sampling results by analysing the energies of the samples")
        result_dict, all_energies = self.analyse_energies_of_samples(
            sampling_counts_overall, qc_or_not, result_name, config, report)
        config.logger.info("finished analysing the energies of the samples")
    
        config.logger.info("start to analyse the feasibility of the qaoa sampling results")
        result_dict_feasibility_analysed = analyse_feasibility_of_solver_solutions(
            result_dict, self.qubo, self.mip_for_feasibility_analysis, config, report)
        config.logger.info("finished analysing the feasibility of the qaoa sampling results") 
            
        config.logger.info("start to get all the feasible solutions of the qaoa sampling results")
        result_feas_sol_dict, result_feas_sol_df = get_all_feasible_solutions(
            result_dict_feasibility_analysed, variable_names_list, report)
        if config.report_config["save_feasible_sols_to_excel"] == True:
            result_feas_sol_df.to_excel(os.path.join(config.EXPORT_PATH, "feasible_solutions.xlsx"), index=False)
            config.logger.info("They were saved to excel to the export-path")
        config.logger.info("finished to get all the feasible solutions of the qaoa sampling results.")
        
        #postprocess with steepest descent
        if config.solve_method_config['postprocess_with_steepest_descent'] == True:
            config.logger.info("start to execute steepest_descent-postprocess on the feasible solutions")
            result_feas_sol_dict, _ = execute_steepest_descent_on_bitstrings(
                self.opt_problem, result_feas_sol_dict, config, report)                
            config.logger.info("finished steepest_descent-postprocess on the feasible solutions")
        
        #visualisaiton
        if config.report_config["visualize_distribution_of_obj_values"] == True:
            config.logger.info("visualize the distribution of objective values and their number of violated constraints of sampling results")
            create_barplot_obj_values_with_number_of_violated_constraints(
                result_dict_feasibility_analysed, config.EXPORT_PATH, "barplot_obj_values")
            config.logger.info("visualize probability densitys of energies of multiple sampling results")
            visualize_qaoa_distribution_of_energies(
                all_energies, config.EXPORT_PATH, "distribution_obj_values_of_qaoa_sampling")
            
        config.logger.info("finished analysing the QAOA sampling results\n")
        
        write_kpis_to_report(feas_sol_dict = result_feas_sol_dict, 
                             number_of_reads = self.number_of_reads, 
                             config = config,
                             report = report)        
        
        
        report.time_measurements['opt_method_result_analysis_total_time_consumption'] = time.time() - start_time
        return result_feas_sol_dict, result_dict_feasibility_analysed
    
    
    def get_qiskit_qaoa_counts(self, result: any, qc_or_not: str):
        ''' 
        function to get the qiskit counts of QAOA sampling results
        '''
        if qc_or_not in ["QC"]: # if quantum sampled
            bitstring_counts = result[0].data.meas.get_counts() 
        elif qc_or_not in ["QC_Simulator"]: # if classically simulated  
            bitstring_counts = dict(result.get_counts())
        else:
            raise CustomizedError("there is no method specified for the processing of the solver result of the used device")
        return bitstring_counts
        
    
    def merge_qaoa_count_dicts(self, old_dict: dict, new_dict: dict, new_results_name: str) -> dict:
        ''' 
        function that merges 2 result dictionarys of qaoa.
        for bitstrings that exist as keys in both, their counts are added.
        '''
        merged_dict = copy.deepcopy(old_dict) # initialize

        for bitstr, count in new_dict.items():
            if bitstr not in merged_dict.keys():
                merged_dict[bitstr] = {'count': count,
                                       'specific_counts': 
                                           {f"{new_results_name}": count}
                                       }
            else:
                merged_dict[bitstr]['count'] += count    
                merged_dict[bitstr]['specific_counts'][f"{new_results_name}"] = count    
            self.number_of_reads += count
        return merged_dict
        

    def analyse_energies_of_samples(self, 
            bitstring_counts: dict, 
            qc_or_not: str,
            sampling_method: str, 
            config: Config,
            report: Report) -> (dict, list):
        ''' 
        function to process a qaoa solver result and adds it to a result_dict 
            { (bitstr): {'count': x, 'energy': y}, ... } 
        from the sampling result of a qiskit-QAOA-circuit. 
        it also outputs a list of all measured energies
        '''
        start_time = time.time()
        result_dict = {} # initialize
        all_energies_of_methods = {} # initialize
        for bitstring, count_dict in bitstring_counts.items():
            count = count_dict['count']
            ising_bitstring = np.array([-1 if bit == '1' else 1 for bit in bitstring]) # convert results (0 and 1) to ising (1 and -1)
            ising_bitstring = np.flip(ising_bitstring) # invert the bitstring because qiskit outputs inverted bitstrings
            energy = calc_ising_energy_of_bitstring_array(self.ising_matrix, self.ising_vector, self.ising_offset, ising_bitstring)
            
            if config.report_config["visualize_distribution_of_obj_values"] == True:
                for qaoa_method in count_dict['specific_counts'].items():
                    if qaoa_method not in all_energies_of_methods.keys():
                        all_energies_of_methods[qaoa_method] = [energy for _ in range(count)]
                    else:
                        all_energies_of_methods[qaoa_method] += [energy for _ in range(count)] 
            
            qubo_bit_str = ', '.join([str(x) for x in bitstring[::-1]])
            result_dict[qubo_bit_str] = {}
            result_dict[qubo_bit_str]['count'] = count
            result_dict[qubo_bit_str]['specific_counts'] = count_dict['specific_counts']
            result_dict[qubo_bit_str]['energy'] = energy   
        
        if config.report_config["analyse_only_best_sols"] == True:
            sorted_items = sorted(result_dict.items(), key=lambda item: item[1]['energy'])
            number_of_sols_to_keep = config.report_config["number_of_sols_to_be_analysed"]
            result_dict = dict(sorted_items[:number_of_sols_to_keep])
            config.logger.info(f"We only keep the {number_of_sols_to_keep} best solutions")
            
        report.time_measurements['process_qaoa_response'] = time.time() - start_time
        return result_dict, all_energies_of_methods
