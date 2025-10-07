# -*- coding: utf-8 -*-
"""
Created on 07.03.2024

@author: Eric Stopfer
"""
import logging
import pdb
import os
import time
import shutil
import ctypes
import re
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from qiskit_ibm_runtime import QiskitRuntimeService

from config.config import Config
from src.run_optimization import Opt_Run, execute_opt_run
from src.utils.error_utils import CustomizedError
from src.utils.visualisation_utils import visualize_metric_forproblemsize_foroptmethod, visualize_dict
from src.utils.config_file_utils import find_config_files, add_paths_to_config_files
from src.utils.import_utils import create_class_instances
from src.utils.qc_utils import create_ibm_session
from src.utils.grbcluster_utils import abort_existing_grbcluster_jobs
from src.problems.MarkowitzPortfolio.visualize_markowitz_benchmark_results import visualize_markowitz_benchmark_results
 


class Benchmark:
    def __init__(self, config_file_path: str, visualisation_dict = None):
        ''' 
        function that initializes a benchmark run by reading and checking the
        validity of a benchmark config file
        '''
        self.benchmark_config_data = self.read_benchmark_config_file(config_file_path)
        self.check_validity_benchmark_config_file()
        
        self.benchmark_config_data_normal, self.benchmark_config_data_ibm = self.separate_config_data_according_to_hardware()
        self.benchmark_config_file_paths_normal = add_paths_to_config_files(self.benchmark_config_data_normal)
        self.benchmark_config_file_paths_ibm = add_paths_to_config_files(self.benchmark_config_data_ibm)
        
        self.ibm_job_ids = [] # if there are some problems solved on the IBM-QCs, this list will contain their job_ids 
        self.max_problem_size_for_graphics = 25 # 7 # 800 # 30 # 1000 # 
        self.min_problem_size_for_graphics = 7 # 5 # 
        if visualisation_dict != None: 
            self.max_problem_size_for_graphics = visualisation_dict['max_problem_size_for_graphics']
            self.min_problem_size_for_graphics = visualisation_dict['min_problem_size_for_graphics']
            self.solving_methods = visualisation_dict['solving_methods']
        else:
            self.max_problem_size_for_graphics = 0
            self.min_problem_size_for_graphics = 1000000
            self.solving_methods = {# 'config_name': 'explanatory_name_for_graphic',
                'mipsolver_scip_timelimit1hour.json':        "SCIP with gap 0%", 
                'mipsolver_scip_timelimit1hour_gap5%.json':  "SCIP with gap 5%",
                'mipsolver_gurobi_timelimit1hour.json':      "Gurobi with gap 0%",
                'mipsolver_gurobi_timelimit1hour_gap5%.json':"Gurobi with gap 5%",
                
                'optheuristic_default_timelimit1min.json':   "Problem-specific heuristic",
                
                'randomsamplingqubo_readsfor1min.json': "Random sampling 60s",
                'randomsamplingqubo_100000reads.json': "Random sampling 100,000 samples",
                'randomsamplingqubo_330000reads.json': "Random sampling 330,000 samples",
                            
                'greedy_dwavegreedyalgorithm_readsfor1min':   "Steepest descent", 
                
                'simulatedanneal_dwave_readsfor1min':        "Simulated annealing", 
                'simulatedanneal_dwave_50optheurreads':     "Simulated annealing heuristically boosted",
                
                'tabusearch_dwavetabusampler_readsfor1min_20timelimit.json':      "Tabu search",
                'tabusearch_dwavetabusampler_50optheurreads_20timelimit.json':  "Tabu search heuristically boosted",
                
                'qaoa_localsimulator_cobyla_1layer':       "QAOA LocalSim, p=1 cobyla", 
                'qaoa_localsimulator_cobyla_2layer':     "QAOA LocalSim, p=2 cobyla", 
                'qaoa_localsimulator_cobyla_3layer':       "QAOA LocalSim, p=3 cobyla", 
                'qaoa_localsimulator_gridsearch':          "QAOA LocalSim, p=1, grid search",
                'qaoa_localsimulator_linearramp_1layer_1minsampl.json':   "QAOA LocalSim, p=1, linear ramp",
                'qaoa_localsimulator_linearramp_2layer_1minsampl.json':   "QAOA LocalSim, p=2, linear ramp",
                'qaoa_localsimulator_linearramp_3layer_1minsampl.json':   "QAOA LocalSim, p=3, linear ramp",
                
                'qaoa_ibm_qc_gridsearch_30sectrain_30secsampl.json':    "QAOA IBM, p=1, grid search",
                'qaoa_ibm_qc_linearramp_1layer_1minsampl.json':         "QAOA IBM, p=1, linear ramp",
                'qaoa_ibm_qc_linearramp_2layer_1minsampl.json':       "QAOA IBM, p=2, linear ramp",
                'qaoa_ibm_qc_linearramp_3layer_1minsampl.json':         "QAOA IBM, p=3, linear ramp",
                'qaoa_ibm_qc_linearramp_4layer_1minsampl.json':       "QAOA IBM, p=4, linear ramp",
                'qaoa_ibm_qc_linearramp_5layer_1minsampl.json':         "QAOA IBM, p=5, linear ramp",
     
                ('quantumanneal_dwave_readsfor1min_1annealingtime.json',
                'quantumanneal_dwave_readsfor1min_1annealingtime_defaultcs.json'):    r"QA D-Wave, $\tau$=1µs, default cs",
                ('quantumanneal_dwave_readsfor1min_5annealingtime.json',
                'quantumanneal_dwave_readsfor1min_5annealingtime_defaultcs.json'):    r"QA D-Wave, $\tau$=5µs, default cs", 
                ('quantumanneal_dwave_readsfor1min_20annealingtime.json',
                'quantumanneal_dwave_readsfor1min_20annealingtime_defaultcs.json'):   r"QA D-Wave, $\tau$=20µs, default cs", 
                ('quantumanneal_dwave_readsfor1min_50annealingtime.json',
                'quantumanneal_dwave_readsfor1min_50annealingtime_defaultcs.json'):   r"QA D-Wave, $\tau$=50µs, default cs",
                ('quantumanneal_dwave_readsfor1min_20annealingtime_3cs.json', 
                 'quantumanneal_dwave_readsfor1min_20annealingtime_4cs.json', 
                 'quantumanneal_dwave_readsfor1min_20annealingtime_5cs.json'):        r"QA D-Wave, $\tau$=20µs, adjusted cs",  
                ('quantumanneal_dwave_readsfor1min_50annealingtime_3cs.json',
                 'quantumanneal_dwave_readsfor1min_50annealingtime_4cs.json',
                 'quantumanneal_dwave_readsfor1min_50annealingtime_5cs.json'):        r"QA D-Wave, $\tau$=50µs, adjusted cs", 
                }
    
    @staticmethod
    def read_benchmark_config_file(file_path: str) -> any:
        ''' 
        function that reads the data inside a config-json-file
        '''
        if os.path.exists(file_path):
            with open(file_path) as json_file:
                try:
                    data = json.load(json_file)
                except:
                    raise CustomizedError("BENCHMARK_CONFIG_FILE_ERROR: check that the json is formatted like a list of lists of length 2. \n \
                                          Also check for wrong comma placements, possibly at the end of the list")
                
            logging.info("The configuration from the benchmark-config-file was taken over")
        else:
            raise CustomizedError("There was no config-file given. The default configuration from config.py is used")
        return data
    

    def check_validity_benchmark_config_file(self):
        ''' 
        function that checks the validity of the data from a benchmark-config-file and 
        throws an error if it is not valid
        '''
        all_problem_config_files = find_config_files(os.path.join(Config.IMPORT_PATH, "problem_configs"))
        all_solver_config_files = find_config_files(os.path.join(Config.IMPORT_PATH, "solver_configs"))
        all_report_config_files = find_config_files(os.path.join(Config.IMPORT_PATH, "report_configs"))
        
        assert isinstance(self.benchmark_config_data, list), "BENCHMARK_CONFIG_FILE_ERROR: the benchmark-config file has to be a list when it is read in"
        for obj in self.benchmark_config_data:
            #check if list of length 2
            assert isinstance(obj, list), f"BENCHMARK_CONFIG_FILE_ERROR: benchmark_config objects have to be lists of length 3. But '{obj}' is not"
            assert len(obj) == 3, f"BENCHMARK_CONFIG_FILE_ERROR: benchmark_config objects have to have length 3 (->problem, solver, report). But '{obj}' has length {len(obj)}"
            problem_config_name, solver_config_name, report_config_name = obj[0], obj[1], obj[2]
            assert problem_config_name in all_problem_config_files, f"BENCHMARK_CONFIG_FILE_ERROR: the file {problem_config_name} doesn't exist"
            assert solver_config_name in all_solver_config_files, f"BENCHMARK_CONFIG_FILE_ERROR: the file {solver_config_name} doesn't exist"
            assert report_config_name in all_report_config_files, f"BENCHMARK_CONFIG_FILE_ERROR: the file {report_config_name} doesn't exist"
            
        logging.info("validity check of the benchmark-config-file was successful") 
    
    
    def separate_config_data_according_to_hardware(self):
        ''' 
        function to separate calculation configs into 3 groups: normal, ibm, dwave
        '''
        ibm_solver_configs = ['qaoa_ibm_qc_cobyla_1layer_52sectrain_8secsampl.json',
                              'qaoa_ibm_qc_cobyla_2layer_52sectrain_8secsampl.json',
                              'qaoa_ibm_qc_cobyla_3layer_52sectrain_8secsampl.json',
                              'qaoa_ibm_qc_linearramp_1layer_1minsampl.json',
                              'qaoa_ibm_qc_linearramp_2layer_1minsampl.json',
                              'qaoa_ibm_qc_linearramp_3layer_1minsampl.json',
                              'qaoa_ibm_qc_linearramp_4layer_1minsampl.json',
                              'qaoa_ibm_qc_linearramp_5layer_1minsampl.json',
                              'qaoa_ibm_qc_gridsearch_30sectrain_30secsampl.json']
        config_data_ibm = []
        config_data_normal = []
        for [problem_config, solver_config, report_config] in self.benchmark_config_data:
            if solver_config in ibm_solver_configs:
                config_data_ibm.append([problem_config, solver_config, report_config])
            else:
                config_data_normal.append([problem_config, solver_config, report_config])
        return config_data_normal, config_data_ibm
                
    
    def create_benchmark_export_folder(self, folder_name: str = ""):
        '''
        function to create a folder for export results with a unique name
        '''
        if folder_name != "":
            self.BENCHMARK_EXPORT_PATH = os.path.join(Config.EXPORT_PATH_BASE, folder_name)
        else:
            self.BENCHMARK_EXPORT_PATH = os.path.join(Config.EXPORT_PATH_BASE, f"benchmarkrun_{time.strftime('%Y_%m_%d_%H_%M_%S')}")
        
        if not os.path.exists(self.BENCHMARK_EXPORT_PATH):
            os.makedirs(self.BENCHMARK_EXPORT_PATH, exist_ok=True)

    
    def write_benchmark_json_to_export_folder(self, config_file_path: str):
        """
        function that saves the benchmark-config-object to json
        """        
        origin_path = config_file_path        
        dest_path = os.path.join(self.BENCHMARK_EXPORT_PATH, "benchmark_config.json")
        shutil.copy2(origin_path, dest_path)


    def run_benchmark_opt_runs(self, 
                               parallel_calculation: bool, 
                               number_of_processors: int = os.cpu_count()-2
                               ):
        ''' 
        function that runs all the optimizations specified in the benchmark config file
        '''        
        solver_api_tokens_path = os.path.dirname(os.path.dirname(__file__)) + "\\config" + "\\config_files\\" + "solver_api_tokens.json"        
        if Config.GUROBI_LICENSE_AVAILABLE == False:
            abort_existing_grbcluster_jobs(solver_api_tokens_path)
        
        opt_results = []
        #run normal benchmarks
        if parallel_calculation == True:
            with ProcessPoolExecutor(max_workers=number_of_processors) as executor: # for parallel processing
                futures = [executor.submit(execute_opt_run, problem_config, solver_config, report_config, self.BENCHMARK_EXPORT_PATH) 
                           for [problem_config, solver_config, report_config] in self.benchmark_config_file_paths_normal]
                for future, config in zip(futures, self.benchmark_config_data_normal):
                    report = future.result()
                    opt_results.append((config, report))
        else:
            for [problem_config, solver_config, report_config] in self.benchmark_config_file_paths_normal:
                report = execute_opt_run(problem_config, solver_config, report_config, self.BENCHMARK_EXPORT_PATH)
                config = [problem_config, solver_config, report_config]
                opt_results.append((config, report))
        
        #run IBM QAOA benchmarks
        if len(self.benchmark_config_file_paths_ibm) > 0:
            with_ibm_session = False
            if with_ibm_session:
                self.ibm_session = create_ibm_session(solver_api_tokens_path)
                for [problem_config, solver_config, report_config] in self.benchmark_config_file_paths_ibm:
                    opt_run = Opt_Run(problem_config, solver_config, report_config, self.BENCHMARK_EXPORT_PATH)
                    opt_run = Opt_Run(problem_config, solver_config, report_config, self.BENCHMARK_EXPORT_PATH, self.ibm_session)
                    report = opt_run.run_optimization()
                    config = [problem_config, solver_config, report_config]
                    opt_results.append((config, report))
                self.ibm_session.close()
            else:
                for [problem_config, solver_config, report_config] in self.benchmark_config_file_paths_ibm:
                    opt_run = Opt_Run(problem_config, solver_config, report_config, self.BENCHMARK_EXPORT_PATH, abort_code_after_ibm_job_submission=True)
                    report = opt_run.run_optimization()
                    ibm_job_id_list = opt_run.solver.ibm_job_ids
                    config = [problem_config, solver_config, report_config]
                    self.ibm_job_ids.append((opt_run.config.EXPORT_PATH, config, ibm_job_id_list))
                    opt_results.append((config, report))
                
                json_file_path = os.path.join(self.BENCHMARK_EXPORT_PATH, 'ibm_job_ids.json')
                with open(json_file_path, "w") as file:
                    json.dump(self.ibm_job_ids, file, indent=4) # indent=4 improves readability of the json
                    
        return opt_results
    
    
    def calc_optimality_kpis(self):
        ''' 
        function that calculates the approximation ratio of the opt-runs by scanning
        the benchmark export folder for SCIP optimizations to optimality
        '''
        opt_sol_info = self.get_all_optimal_solutions_in_export_path()
        
        if opt_sol_info != {}:
            for item in os.listdir(self.BENCHMARK_EXPORT_PATH):
                root_path = os.path.join(self.BENCHMARK_EXPORT_PATH, item)
                if "MIP_Solver" not in item and os.path.isdir(root_path):
                    if "problemsolverreport_combination.json" in os.listdir(root_path) and "report.json" in os.listdir(root_path):
                        file_path = os.path.join(root_path, "problemsolverreport_combination.json")
                        with open(file_path, 'r') as file:
                            config_data = json.load(file)
                        problem_config = config_data[0]
                        problem_config_cleaned = problem_config[problem_config.find('quopt'):]
                        report_path = os.path.join(root_path, "report.json")
                        try:
                            with open(report_path, 'r') as file:
                                report_data = json.load(file)
                        except:
                            print(f"error with report.json in {root_path}")
                            pdb.set_trace()
                        #delete prior optimality kpi results from report
                        optimality_kpis = ['approximation_ratio', 'absolute_optimality_gap', 'relative_optimality_gap']
                        for kpi in optimality_kpis:
                            if kpi in report_data['solution_quality'].keys():
                                del report_data['solution_quality'][kpi]
                        
                        #calc optimality kpis
                        if not all(metric in report_data['solution_quality'].keys() for metric in optimality_kpis):
                            if 'best_feasible_solution_obj_value' in report_data['solution_quality'].keys():
                                if os.path.normpath(problem_config_cleaned) in [os.path.normpath(path) for path in opt_sol_info.keys()]:
                                    best_sol = report_data['solution_quality']['best_feasible_solution_obj_value']    
                                    opt_sol = opt_sol_info[os.path.normpath(problem_config_cleaned)] 
                                    approximation_ratio = best_sol / opt_sol
                                    absolute_optimality_gap = abs(best_sol - opt_sol)
                                    relative_optimality_gap = abs(best_sol - opt_sol) / opt_sol
                                else:
                                    approximation_ratio = 'no known optimal solution thus no info'
                                    absolute_optimality_gap = 'no known optimal solution thus no info'
                                    relative_optimality_gap = 'no known optimal solution thus no info'
                            else: #no feasible solutions found in this on
                                approximation_ratio = 'no feasible solution found'
                                absolute_optimality_gap = 'no feasible solution found'
                                relative_optimality_gap = 'no feasible solution found'
                                
                            report_data['solution_quality']['approximation_ratio'] = approximation_ratio
                            report_data['solution_quality']['absolute_optimality_gap'] = absolute_optimality_gap
                            report_data['solution_quality']['relative_optimality_gap'] = relative_optimality_gap
                            with open(report_path, "w") as file:
                                json.dump(report_data, file, indent=4) 
                                
    
    def get_all_optimal_solutions_in_export_path(self):
        ''' 
        function that scans all the opt-runs in a path for MIPSolver-solutions and 
        saves the optimal objective values into a dictionary that looks like this:
            { problem_config1: optimal_obj_value, 
              problem_config2: optimal_obj_value, ...}
        '''
        opt_sol_info = {}
        for item in os.listdir(self.BENCHMARK_EXPORT_PATH):
            root_path = os.path.join(self.BENCHMARK_EXPORT_PATH, item)
            if os.path.isdir(root_path):
                if "problemsolverreport_combination.json" in os.listdir(root_path):
                    file_path = os.path.join(root_path, "problemsolverreport_combination.json")
                    with open(file_path, 'r') as file:
                        config_data = json.load(file)
                    this_problem_config_path = os.path.normpath(config_data[0])
                    this_problem_config_path_cleaned = this_problem_config_path[this_problem_config_path.find('quopt'):]
                    if this_problem_config_path_cleaned not in opt_sol_info.keys(): #if not calculated yet
                        this_solver_config_path = config_data[1]
                        mipsolver_strings = ["mipsolver_scip_timelimit1hour.json", "mipsolver_gurobi_timelimit1hour.json"]
                        if any(mipsolver in this_solver_config_path for mipsolver in mipsolver_strings):
                            mipsolve_report_path = os.path.join(root_path, "report.json")
                            with open(mipsolve_report_path, 'r') as file:
                                mipsolve_report_data = json.load(file)
                            if 'best_feasible_solution_obj_value' in mipsolve_report_data['solution_quality'].keys():
                                opt_sol = mipsolve_report_data['solution_quality']['best_feasible_solution_obj_value']
                                opt_sol_info[this_problem_config_path_cleaned] = opt_sol
        return opt_sol_info
        
            
    def calc_qscore(self, samplesize_random: int):
        ''' 
        function to calculate the qscore for problem-solver-combination
        -->reason for existence is to not have to calculate the same 
        optimal solution and random solution for the qscore over and over again
        '''
        #calculate the qscore metrics
        qscore_folder_name = "calculations_for_qscore"
        calcs_for_qscore_path = os.path.join(self.BENCHMARK_EXPORT_PATH, qscore_folder_name)
        os.makedirs(calcs_for_qscore_path)
        opt_and_rand_sol_info = {}
        for [problem_config, solver_config, report_config] in self.benchmark_config_file_paths_normal:
            if problem_config not in opt_and_rand_sol_info.keys(): #if not calculated yet
                opt_and_rand_sol_info[problem_config] = {}
                config = Config(problem_config, 
                                solver_config, 
                                report_config, 
                                benchmark_export_folder_path=calcs_for_qscore_path)
                problem_class, _, report_class = create_class_instances(config)
                report = report_class(config)
                opt_problem = problem_class(config, report)
                mapped_problem = opt_problem.map_problem(config, report)
                C_optsol = opt_problem.calc_C_optsol(config, report)        
                start_time_rand = time.time()
                C_rand, realized_samplesize, counter_random_gen = opt_problem.calc_C_rand(samplesize_random, config, report)
                opt_and_rand_sol_info[problem_config]['opt_sol'] = C_optsol
                opt_and_rand_sol_info[problem_config]['rand_sol'] = C_rand
                opt_and_rand_sol_info[problem_config]['counter_random_sampling_for_qscore'] = counter_random_gen
                opt_and_rand_sol_info[problem_config]['samplesize_randomsample_for_qscore'] = realized_samplesize
        
        #write the found qscoremetrics in the report.jsons
        for item in os.listdir(self.BENCHMARK_EXPORT_PATH):
            root_path = os.path.join(self.BENCHMARK_EXPORT_PATH, item)
            if os.path.isdir(root_path):
                if item != qscore_folder_name and "problemsolverreport_combination.json" in os.listdir(root_path):
                    file_path = os.path.join(root_path, "problemsolverreport_combination.json")
                    with open(file_path, 'r') as file:
                        config_data = json.load(file)
                    problem_config = config_data[0]
                    opt_sol = opt_and_rand_sol_info[problem_config]['opt_sol']
                    rand_sol = opt_and_rand_sol_info[problem_config]['rand_sol']
                    report_path = os.path.join(root_path, "report.json")
                    with open(report_path, 'r') as file:
                        report_data = json.load(file)
                    if 'best_feasible_solution_obj_value' in report_data['solution_quality'].keys():
                        best_sol = report_data['solution_quality']['best_feasible_solution_obj_value']
                        report_data['solution_quality']['qscore_metric'] = (best_sol - rand_sol) / (opt_sol - rand_sol)
                        report_data['solution_quality']['counter_random_sampling_for_qscore'] = opt_and_rand_sol_info[problem_config]['counter_random_sampling_for_qscore']
                        report_data['solution_quality']['samplesize_randomsample_for_qscore'] = opt_and_rand_sol_info[problem_config]['samplesize_randomsample_for_qscore']
                    else:
                        report_data['solution_quality']['qscore_metric'] = 0
                    with open(report_path, "w") as file:
                        json.dump(report_data, file, indent=4) 
    
    
    def analyse_benchmark_opt_run_results(self):
        ''' 
        function to analyse the results of a benchmark run.
        the results are inside of a list as tuples of config-name and opt-run-report-object
        '''
        opt_results_dict = self.get_existing_benchmark_results()
        self.executable_status = self.analyse_code_termination(opt_results_dict)
        
        kpi_dicts = []
        metrics_to_analyse = {
            # "metric_name_in_report": "description of metric",
            "approximation_ratio":                      "approximation ratio",
            "number_of_shots":                          "number of samples",
            "absolute_optimality_gap":                  "absolute optimality gap",
            "relative_optimality_gap":                  "realtive optimality gap",
            "feasible_solutions_how_many":              "number of found feasible solutions",
            "opt_method_run_total_time_consumption":    "solving time in s",
            "qscore_metric":                            "Qscore metric",
            "counter_random_sampling_for_qscore":       "counter of random sampling for Qscore",
            "samplesize_randomsample_for_qscore":       "samplesize of randomsample for Qscore",
            "feasible_solution_probability":            "feasible solution probability",
            "overall_calculation_time":                 "total calculation time in s",
            "opt_method_result_analysis_total_time_consumption": "result analysis time in s",
            }        
        problem_size_identifiers = {"MarkowitzPortfolio":   'assets', 
                                    "TSP":                  'nodes', 
                                    "BinPacking":           'objects', 
                                    "CVRP":                 'nodes'}
        
        for problem_type, problem_type_dict in opt_results_dict.items():
            for metric, metric_descr in metrics_to_analyse.items():
                problem_formulation_metric_dict = {}
                for problem_formulation, opt_results in problem_type_dict.items():
                    if opt_results != []:
                        kpi_dict = self.analyse_kpis(opt_results, metric, problem_type, problem_formulation)
                        kpi_dicts.append((metric, kpi_dict))
                        metric_dict, average_metric_dict, std_dev_metric_dict = self.calc_average_of_metric_for_problem_size(
                            metric_result_file_path = os.path.join(self.BENCHMARK_EXPORT_PATH, f"results_{problem_type}_{problem_formulation}_{metric}.json"), 
                            problem_type = problem_type,
                            problem_size_identifier = problem_size_identifiers[problem_type])
                        problem_formulation_metric_dict[problem_formulation] = {}
                        problem_formulation_metric_dict[problem_formulation]['average'] = average_metric_dict
                        problem_formulation_metric_dict[problem_formulation]['standard deviation'] = std_dev_metric_dict
                        if problem_type == "MarkowitzPortfolio":
                            if len(opt_results) <= 10: # the following methods only make sense for benchmark runs of only one problem
                                visualize_markowitz_benchmark_results(opt_results, self.BENCHMARK_EXPORT_PATH, only_best_sols=True)
                                visualize_markowitz_benchmark_results(opt_results, self.BENCHMARK_EXPORT_PATH, only_best_sols=False)
                
                metrics_with_log_scale = ['number_of_shots', 
                                          'approximation_ratio',
                                          'overall_calculation_time', 
                                          'opt_method_run_total_time_consumption', 
                                          'opt_method_result_analysis_total_time_consumption',
                                          'feasible_solutions_how_many']
                metrics_with_percentage_scale = ['feasible_solution_probability']
                logarithmic_scale = True if metric in metrics_with_log_scale else False
                percentage_scale = True if metric in metrics_with_percentage_scale else False
                visualize_metric_forproblemsize_foroptmethod(
                    problem_formulation_metric_dict = problem_formulation_metric_dict,  
                    problem_type = problem_type,
                    max_problem_size = self.max_problem_size_for_graphics,
                    min_problem_size = self.min_problem_size_for_graphics,
                    metric_name = metric_descr,
                    export_path = self.BENCHMARK_EXPORT_PATH,
                    with_errorbars = True,
                    logarithmic_scale = logarithmic_scale,
                    percentage_scale = percentage_scale
                    )
        
            
    def get_existing_benchmark_results(self) -> dict:
        '''
        scans the export folder for the report- and solution_dict.jsons and saves the data into a dict of the following strucutre: 
        {problem_type: [(problemsolverreport_combo, report, solution_dict), ...], ... }
        '''
        if not os.path.exists(self.BENCHMARK_EXPORT_PATH):
            raise FileNotFoundError(f"Folder '{self.BENCHMARK_EXPORT_PATH}' does not exist.")
        
        opt_results_dict = {
                            "MarkowitzPortfolio": 
                                {'MaxRet': [], 'MinVola': [], 'MultiObj': []}, 
                            "TSP": 
                                {'var_for_each_edge': [], 'var_for_node_at_timestep': []}, 
                            "BinPacking": 
                                {'without_incompatibilities_and_precedences': [], 'withincompatibilities': [], 'withprecedences': [], 'with_incompatibilities_and_precedences': []}, 
                            "CVRP": 
                                {},
                            }
        for problem_type in opt_results_dict.keys():
            for problem_formulation in opt_results_dict[problem_type].keys():
                for subdir in os.listdir(self.BENCHMARK_EXPORT_PATH):
                    subdir_path = os.path.join(self.BENCHMARK_EXPORT_PATH, subdir)
                    if os.path.isdir(subdir_path):
                        probsolvrep_combo_json_path = os.path.join(subdir_path, "problemsolverreport_combination.json")
                        report_json_path = os.path.join(subdir_path, "report.json")
                        solution_dict_json_path = os.path.join(subdir_path, "solution_dict.json")
            
                        if os.path.exists(probsolvrep_combo_json_path):
                            with open(probsolvrep_combo_json_path, "r") as file:
                                probsolvrep_combo = json.load(file)
                            problem_config = probsolvrep_combo[0]
                            if problem_type in problem_config and problem_formulation.lower() in problem_config:
                                if os.path.exists(report_json_path) and os.path.exists(solution_dict_json_path):
                                    with open(solution_dict_json_path, "r") as file:
                                        solution_dict = json.load(file)
                                    with open(report_json_path, "r") as file:
                                        report_dict = json.load(file)
                                    class DictToClass:
                                        def __init__(self, dictionary):
                                            for key, value in dictionary.items():
                                                setattr(self, key, value)
                                    report_obj = DictToClass(report_dict)
                                    opt_results_dict[problem_type][problem_formulation].append((probsolvrep_combo, report_obj, solution_dict))
                                # else:
                                #     opt_results_dict[problem_type][problem_formulation].append((probsolvrep_combo, False, None))
        return opt_results_dict
        
                
    def analyse_code_termination(self, opt_results_dict: dict) -> bool:
        ''' 
        function to analyse if the code terminated for each benchmark run.
        It saves a result file with a execution_success-dict to the export folder. 
        It returns True if all calculations were executed successfully, otherwise False.
        '''
        execution_success_dict = {}
        executable_status = True
        
        for problem_type, problem_type_dict in opt_results_dict.items():
            for problem_formulation, opt_results in problem_type_dict.items():
                for [problem_config, solver_config, _], report_obj, _ in opt_results:
                    if report_obj != False:
                        opt_run_status = "successful"
                    else:
                        opt_run_status = "failed"
                        executable_status = False
                        print(f"solving '{problem_config}' with '{solver_config} has failed")
                    
                    if problem_config not in execution_success_dict.keys():
                        execution_success_dict[problem_config] = {}
                    execution_success_dict[problem_config][solver_config] = opt_run_status
            
        self.write_result_dict_to_export_folder_as_json(execution_success_dict, "results_executabletest.json")
        
        return executable_status
            
    
    def analyse_kpis(self, opt_results: list, kpi_to_analyse: str, problem_type: str, problem_formulation: str) -> dict:
        ''' 
        function to analyse the optimization results regarding a KPI (like Q-Score, feas_sol_probability, ...).
        It returns a kpi-analysis-dict which is structured the following way:
            = { problem_config1: {solver_config1: kpi_value,
                                  solver_config2: kpi_value}, ...}
        It writes the results to a result-dict-jsons. 
        '''
        kpi_dict = {}
        for [problem_config, solver_config, _], report_obj, _ in opt_results:
            problem_config_file_name = problem_config.split("\\")[-1]
            if problem_config_file_name not in kpi_dict.keys(): #initialize
                kpi_dict[problem_config_file_name] = {}
            
            if report_obj != False:
                if kpi_to_analyse in report_obj.solution_quality.keys():
                    kpi_value = report_obj.solution_quality[kpi_to_analyse]
                    kpi_dict[problem_config_file_name][solver_config] = kpi_value
                elif kpi_to_analyse in report_obj.time_measurements.keys():
                    kpi_value = report_obj.time_measurements[kpi_to_analyse]
                    kpi_dict[problem_config_file_name][solver_config] = kpi_value
                # elif kpi_to_analyse in report_obj.calculation_statistics.keys():
                #     kpi_value = report_obj.calculation_statistics[kpi_to_analyse]
                #     kpi_dict[problem_config_file_name][solver_config] = kpi_value
                else:
                    kpi_dict[problem_config_file_name][solver_config] = f"'{kpi_to_analyse}' was not calculated"
            else:
                kpi_dict[problem_config_file_name][solver_config] = "opt_run didn't terminate successfully"
        
        self.write_result_dict_to_export_folder_as_json(kpi_dict, f"results_{problem_type}_{problem_formulation}_{kpi_to_analyse}.json")
        
        return kpi_dict
    
    
    def calc_average_of_metric_for_problem_size(self, metric_result_file_path: str, problem_type: str, problem_size_identifier: str) -> (dict, dict):
        ''' 
        function that takes a path that leads to a metric_result_file_path which contains 
        information about a metric of each solving method for each problem in the 
        benchmark run
        returns two dicts:
        1) {solve_method: {problem_size: [list of metric values]}}
        2) {solve_method: {problem_size: average of list of metric values}}
        '''
        # read the given metric result json file
        with open(metric_result_file_path) as json_file:
            metric_data = json.load(json_file)
        
        assert all(isinstance(x, str) or isinstance(x, tuple) for x in self.solving_methods.keys())
        
        # fill the metric dict
        metric_dict = {method_description: {} for method_description in self.solving_methods.values()}
        for problem_config_path, result_dict in metric_data.items():
            for solver_config_path, metric in result_dict.items():
                for method_identifier_s, method_description in self.solving_methods.items():
                    if isinstance(method_identifier_s, str):
                        if method_identifier_s in solver_config_path:
                            problem_size = int(re.search(rf"_(\d+){problem_size_identifier}", problem_config_path).group(1))
                            if problem_size not in metric_dict[method_description]:
                                metric_dict[method_description][problem_size] = []
                            metric_dict[method_description][problem_size].append(metric)
                            break
                    elif isinstance(method_identifier_s, tuple):
                        for method_identifier in method_identifier_s:
                            if method_identifier in solver_config_path:
                                problem_size = int(re.search(rf"_(\d+){problem_size_identifier}", problem_config_path).group(1))
                                if problem_size not in metric_dict[method_description]:
                                    metric_dict[method_description][problem_size] = []
                                metric_dict[method_description][problem_size].append(metric)
                                break
                    
        #calc the average for each entry in the metric dict
        average_metric_dict = {method_description: {} for method_identifier, method_description in self.solving_methods.items()}
        std_dev_metric_dict = {method_description: {} for method_identifier, method_description in self.solving_methods.items()}
        for method_description, problem_size_dict in metric_dict.items():
            for problem_size, metric_list in problem_size_dict.items():
                floats_in_metric_list = [x for x in metric_list if isinstance(x, float) or isinstance(x, int)]
                if floats_in_metric_list != []:
                    mean = np.mean(floats_in_metric_list) 
                    std_dev = np.std(floats_in_metric_list)
                    average_metric_dict[method_description][problem_size] = mean
                    std_dev_metric_dict[method_description][problem_size] = std_dev
                else:
                    print(f"there is an error when treating the metric of {method_description} for problemsize {problem_size} of {metric_result_file_path}")
                    
        if "feasible_solutions_how_many" in metric_result_file_path:
            number_of_problems_solved_dict = {method_description: {} for method_identifier, method_description in self.solving_methods.items()}
            for method_description, problem_size_dict in metric_dict.items():
                for problem_size, metric_list in problem_size_dict.items():
                    number_of_problems = len(metric_list)
                    number_of_problems_with_feasibles_found = len([x for x in metric_list if x > 0])
                    number_of_problems_solved_dict[method_description][problem_size] = number_of_problems_with_feasibles_found / number_of_problems
            
            visualize_dict(
                data = number_of_problems_solved_dict,
                problem_type = problem_type, 
                max_problem_size = self.max_problem_size_for_graphics,
                min_problem_size = self.min_problem_size_for_graphics,
                metric_name = "feasibility percentage", 
                export_path = self.BENCHMARK_EXPORT_PATH,
                logarithmic_scale = False,
                percentage_scale = True)
            json_file_name = metric_result_file_path[:-32] + "feasibility percentage.json"
            with open(json_file_name, "w") as f:
                json.dump(number_of_problems_solved_dict, f, indent=4)
        
        return metric_dict, average_metric_dict, std_dev_metric_dict
            
    
    def write_result_dict_to_export_folder_as_json(self, result_dict: dict, filename: str):
        ''' 
        function that writes a result dict to the benchmark-export-path as a json
        '''
        if not filename.endswith(".json"):
            raise CustomizedError("filename of the result_dictionary to save has to end with '.json'")
            
        json_file_path = os.path.join(self.BENCHMARK_EXPORT_PATH, filename)
        with open(json_file_path, "w") as file:
            json.dump(result_dict, file, indent=4) # indent=4 adds for new keys of the dict 4 spaces -> improves readability of the json
           
    
    def analyse_ibm_job_results(self, folder_path_with_existing_results):
        ''' 
        function that analyses the results of the jobs that were submitted to the
        IBM Quantum computers
        '''
        assert "ibm_job_ids.json" in os.listdir(folder_path_with_existing_results), "missing ibm_job_ids file for IBM-Job-Result-Analysis"
        
        #get the ibm job ids and save a copy to the Benchmark export path
        ibm_job_ids_file_path = os.path.join(folder_path_with_existing_results, "ibm_job_ids.json")
        with open(ibm_job_ids_file_path, "r") as file:
            ibm_job_ids = json.load(file)
        shutil.copy(ibm_job_ids_file_path, os.path.join(self.BENCHMARK_EXPORT_PATH, "ibm_job_ids.json"))
        
        solver_api_tokens_path = os.path.dirname(os.path.dirname(__file__)) + "\\config" + "\\config_files\\" + "solver_api_tokens.json"
        with open(solver_api_tokens_path, "r") as json_file:
            solver_api_tokens_data = json.load(json_file)        
        ibm_token = solver_api_tokens_data['IBM_Token']
        if Config.GUROBI_LICENSE_AVAILABLE == False:
            abort_existing_grbcluster_jobs(solver_api_tokens_path)
        
        service = QiskitRuntimeService(
            channel='ibm_quantum',
            instance='fraunhofer/bayern/iis',
            token=ibm_token
            )
        for export_path, [problem_config, solver_config, report_config], job_ids in ibm_job_ids:
            job_results_list = []
            jobs_time_usage = 0
            for job_id in job_ids:
                job = service.job(job_id)
                if job.in_final_state():
                    job_result = job.result()
                    jobs_time_usage += job.usage()
                    job_results_list.append(("QAOA on IBM QC", job_result, "QC"))
                else:
                    print(job.error_message())
                    
            opt_run = Opt_Run(problem_config, solver_config, report_config, self.BENCHMARK_EXPORT_PATH, existing_ibm_job_result = job_results_list)
            opt_run.run_optimization()
            
            #add some report info
            report_file_path = os.path.join(export_path, "report.json")
            if os.path.exists(report_file_path):
                with open(report_file_path, "r") as og_report_file:
                    og_report_data = json.load(og_report_file)                            
                og_opt_method_duration = og_report_data['time_measurements']['opt_method_run_total_time_consumption']
            else:
                og_opt_method_duration = 0
            opt_run.report.time_measurements['QAOA_run_on_IBM_QC'] = jobs_time_usage
            opt_run.report.time_measurements['opt_method_run_total_time_consumption'] = og_opt_method_duration + jobs_time_usage
            opt_run.report.write_to_json(opt_run.config) 
                

def execute_benchmark_run(config_file_path: str, 
                          parallel_calculation: bool,
                          number_of_processors_for_parallel: int = os.cpu_count() - 2,
                          existing_folder_name: str = "",
                          calculate_qscore_afterwards: bool = False):
    ''' 
    function to execute a whole benchmark run for an input config-file
    '''
    benchmark = Benchmark(config_file_path)
    benchmark.create_benchmark_export_folder(existing_folder_name)
    benchmark.write_benchmark_json_to_export_folder(config_file_path)
    
    opt_results = benchmark.run_benchmark_opt_runs(parallel_calculation, number_of_processors_for_parallel)
    
    benchmark.calc_optimality_kpis()
    if calculate_qscore_afterwards == True:
        benchmark.calc_qscore(samplesize_random=1000)
    benchmark.analyse_benchmark_opt_run_results()
    


if __name__ == "__main__":
    config_file_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "config", "config_files", "benchmark_config.json"
        )

    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    execute_benchmark_run(
        config_file_path = config_file_path,
        # parallel_calculation = True,
        # number_of_processors_for_parallel = 4 #os.cpu_count() - 3  # cpu-count of my laptop: 12
        parallel_calculation = False,
        existing_folder_name = '',
        calculate_qscore_afterwards = False
        )
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)

    