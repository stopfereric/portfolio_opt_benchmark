# -*- coding: utf-8 -*-
"""
Created on 10.01.2025

@author: Eric Stopfer
"""
import json
import os 
import pdb

from config.config import Config


def find_config_files(directory: str):
    ''' 
    function that returns a list of all file paths in the input directory that end with ".json" 
    '''
    config_files_list = []
    for root, dirs, filenames in os.walk(directory):
        for file_name in filenames:
            if file_name.endswith('.json'):
                config_files_list.append(file_name)
            
    return config_files_list

    
def add_paths_to_config_files(benchmark_config_data: list) -> list:
    ''' 
    function that adds paths to the file-names in the benchmark_config_data
    '''
    benchmark_config_paths = []
    all_problems_config_path = os.path.join(Config.IMPORT_PATH, "problem_configs")
    all_solvers_config_path = os.path.join(Config.IMPORT_PATH, "solver_configs")
    all_reports_config_path = os.path.join(Config.IMPORT_PATH, "report_configs")
    for [problem_config_name, solver_config_name, report_config_name] in benchmark_config_data:
        problem_config_path = find_file_in_path(problem_config_name, all_problems_config_path)
        solver_config_path = find_file_in_path(solver_config_name, all_solvers_config_path)
        report_config_path = find_file_in_path(report_config_name, all_reports_config_path)
        benchmark_config_paths.append([problem_config_path, solver_config_path, report_config_path])
    return benchmark_config_paths


def find_file_in_path(filename: str, directory: str) -> str:
    '''
    function that finds a file that is called -filename- inside
    a certain input directory by walking through this directory and 
    terminating at the first find
    '''
    for root, dirs, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    


def create_markowitz_benchmark_config_files(problemconfigbasenames: list,
                                            number_of_assets: list,
                                            number_of_problemconfigs_per_assetnumber: int,
                                            solve_methods: list, 
                                            reportconfigname: str, 
                                            configfilepath: str, 
                                            configfilename: str):
    ''' 
    function that creates markowitz benchmark config files
    '''
    benchmark_list = []
    
    for asset_number in number_of_assets:
        for problemconfigbasename in problemconfigbasenames:
            for problem_idx in range(number_of_problemconfigs_per_assetnumber):
                for solve_method in solve_methods:
                    problem_config_name = problemconfigbasename + f"{asset_number}assets_{problem_idx}.json"
                    benchmark_list.append([problem_config_name, solve_method, reportconfigname])
                
    with open(os.path.join(configfilepath, configfilename), 'w') as file:
        json.dump(benchmark_list, file, indent=4)
                
    
if __name__ == "__main__":
    create_markowitz_benchmark_config_files(
        problemconfigbasenames = ["random_MarkowitzPortfolio_fromnasdaq_minvola_", 
                                 # "random_MarkowitzPortfolio_fromnasdaq_maxret_", 
                                 # "random_MarkowitzPortfolio_fromnasdaq_multiobj_",
                                 ],
        number_of_problemconfigs_per_assetnumber = 10, 
        
    # %% all solve methods
        # number_of_assets = [3, 5, 7] + 
        #                    list(range(10,100,10)) +
        #                    list(range(100,300,50)) +
        #                    list(range(300,1000,100)) +
        #                    [1000],
        number_of_assets = [7,10,15,20,25,30],
        solve_methods = [
            # 'mipsolver_scip_timelimit1hour.json',
            # 'mipsolver_scip_timelimit10min.json',
            # 'mipsolver_scip_timelimit1min.json',
            # 'mipsolver_scip_timelimit1hour_gap0,1%.json',
            # 'mipsolver_scip_timelimit1hour_gap1%.json',
            # 'mipsolver_scip_timelimit1hour_gap5%.json',
            # 'mipsolver_gurobi_timelimit1hour.json',
            # 'mipsolver_gurobi_timelimit1hour_gap5%.json',
            # 'mipsolver_gurobi_computeserver_timelimit1hour.json',
            # 'mipsolver_gurobi_computeserver_timelimit1hour_gap5%.json',
            
            # 'greedy_dwavegreedyalgorithm_readsfor1min_20timelimit.json', 
            # 'greedy_dwavegreedyalgorithm_1000reads_20timelimit.json',
            # 'greedy_dwavegreedyalgorithm_100000reads_20timelimit.json',
            
            # 'qaoa_ibm_qc_localtrainingcobyla_1layer_52sectrain_8secsampl.json',
            # 'qaoa_ibm_qc_localtrainingcobyla_2layer_52sectrain_8secsampl.json',
            # 'qaoa_ibm_qc_localtrainingcobyla_3layer_52sectrain_8secsampl.json',
            # 'qaoa_ibm_qc_gridsearch_30sectrain_30secsampl.json',   
            # 'qaoa_ibm_qc_linearramp_1layer_1minsampl.json',
            # 'qaoa_ibm_qc_linearramp_2layer_1minsampl.json',
            # 'qaoa_ibm_qc_linearramp_3layer_1minsampl.json',
            # 'qaoa_localsimulator_linearramp_1layer_1minsampl.json',
            # 'qaoa_localsimulator_linearramp_2layer_1minsampl.json',
            # 'qaoa_localsimulator_linearramp_3layer_1minsampl.json',
            # 'qaoa_localsimulator_cobyla_1layer_52sectrain_8secsampl.json',
            # 'qaoa_localsimulator_cobyla_2layer_52sectrain_8secsampl.json',
            # 'qaoa_localsimulator_cobyla_3layer_52sectrain_8secsampl.json',
            # 'qaoa_localsimulator_gridsearch_52sectrain_8secsampl.json',
            
            # 'quantumanneal_dwave_5000reads_20annealingtime_defaultcs.json',
            # 'quantumanneal_dwave_10000reads_5annealingtime_defaultcs.json',
            # 'quantumanneal_dwave_readsfor1min_1annealingtime_defaultcs.json', 
            # 'quantumanneal_dwave_readsfor1min_5annealingtime_defaultcs.json', 
            # 'quantumanneal_dwave_readsfor1min_20annealingtime_defaultcs.json',
            # 'quantumanneal_dwave_readsfor1min_50annealingtime_defaultcs.json', 
            # 'quantumanneal_dwave_readsfor1min_20annealingtime_2cs.json',
            # 'quantumanneal_dwave_readsfor1min_20annealingtime_3cs.json',
            # 'quantumanneal_dwave_readsfor1min_20annealingtime_4cs.json',
            # 'quantumanneal_dwave_readsfor1min_20annealingtime_5cs.json',
            # 'quantumanneal_dwave_readsfor1min_50annealingtime_2cs.json', 
            # 'quantumanneal_dwave_readsfor1min_50annealingtime_3cs.json', 
            # 'quantumanneal_dwave_readsfor1min_50annealingtime_4cs.json',
            # 'quantumanneal_dwave_readsfor1min_50annealingtime_5cs.json',
            
            # 'simulatedanneal_dwave_50optheurreads20annealingtime.json',
            # 'simulatedanneal_dwave_readsfor1min_20annealingtime.json',
            # 'simulatedanneal_dwave_1000reads_20annealingtime.json',
            # 'simulatedanneal_dwave_100000reads_20annealingtime.json',
            
            # 'tabusearch_dwavetabusampler_50optheurreads_20timelimit.json',
            # 'tabusearch_dwavetabusampler_readsfor1min_20timelimit.json',
            # 'tabusearch_dwavetabusampler_1000reads_20timelimit.json',
            # 'tabusearch_dwavetabusampler_100000reads_20timelimit.json',
        
            # 'optheuristic_default_timelimit1min.json',
        
            # 'randomsamplingqubo_readsfor1min.json',
            # 'randomsamplingqubo_330000reads.json',
            'randomsamplingqubo_100000reads.json',
            ], 
        
    # %% mip solver
        # number_of_assets = [3, 5, 7] + 
        #                    list(range(10,100,10)) +
        #                    list(range(100,300,50)) +
        #                    list(range(300,1000,100)) +
        #                    [1000, 2000], 
        # number_of_assets = [15, 25],
        # solve_methods = [
        #     'mipsolver_scip_timelimit1hour.json',
        #     'mipsolver_scip_timelimit1hour_gap5%.json',
        #     'mipsolver_gurobi_timelimit1hour.json',
        #     'mipsolver_gurobi_timelimit1hour_gap5%.json',
        #     ], 
        
    # %% heuristics
        # number_of_assets = [3, 5, 7] + 
        #                    list(range(10,100,10)) +
        #                    list(range(100,300,50)) +
        #                    list(range(300,1000,100)) +
        #                    [1000], 
        # # number_of_assets = [15, 25],
        # solve_methods = [            
        #     # 'mipsolver_gurobi_timelimit1hour.json',
        #     # 'greedy_dwavegreedyalgorithm_readsfor1min_20timelimit.json', 
        #     # 'simulatedanneal_dwave_readsfor1min_20annealingtime.json',
        #     # 'simulatedanneal_dwave_50optheurreads20annealingtime.json',
        #     # 'tabusearch_dwavetabusampler_readsfor1min_20timelimit.json',
        #     # 'tabusearch_dwavetabusampler_50optheurreads_20timelimit.json',
        #     'optheuristic_default_timelimit1min.json',
        #     'randomsamplingqubo_readsfor1min.json',
        #     ], 
        
    # %% Quantum Annealing
        # number_of_assets = [3,5,7,10,15,20,25],
        # number_of_assets = [5,7],
        # number_of_assets = [5,7,10,15,20,25],
        # number_of_assets = [7, 10, 15],
        # number_of_assets = [20],
        # number_of_assets = [25],
        # solve_methods = [
            # 'quantumanneal_dwave_readsfor1min_1annealingtime_defaultcs.json', 
            # 'quantumanneal_dwave_readsfor1min_5annealingtime_defaultcs.json', 
            # 'quantumanneal_dwave_readsfor1min_20annealingtime_defaultcs.json',
            # 'quantumanneal_dwave_readsfor1min_50annealingtime_defaultcs.json',
            # 'quantumanneal_dwave_readsfor1min_20annealingtime_2cs.json',
            # 'quantumanneal_dwave_readsfor1min_20annealingtime_3cs.json',
            # 'quantumanneal_dwave_readsfor1min_20annealingtime_4cs.json',
            # 'quantumanneal_dwave_readsfor1min_20annealingtime_5cs.json',
            # 'quantumanneal_dwave_readsfor1min_50annealingtime_2cs.json', 
            # 'quantumanneal_dwave_readsfor1min_50annealingtime_3cs.json', 
            # 'quantumanneal_dwave_readsfor1min_50annealingtime_4cs.json',
            # 'quantumanneal_dwave_readsfor1min_50annealingtime_5cs.json',
            # 'randomsamplingqubo_x_reads.json',
            # ],
    
    # %% QAOA on IBM
        # number_of_assets = [3,5,7,10,15,20,25,30],
        # solve_methods = [
        #     # 'qaoa_ibm_qc_localtrainingcobyla_3layer_52sectrain_8secsampl.json',
        #     # 'qaoa_ibm_qc_localtrainingcobyla_2layer_52sectrain_8secsampl.json',
        #     # 'qaoa_ibm_qc_localtrainingcobyla_1layer_52sectrain_8secsampl.json',
        #     'qaoa_ibm_qc_gridsearch_30sectrain_30secsampl.json',
        #     'qaoa_ibm_qc_linearramp_1layer_1minsampl.json',
        #     'qaoa_ibm_qc_linearramp_2layer_1minsampl.json',
        #     'qaoa_ibm_qc_linearramp_3layer_1minsampl.json',
        #     'qaoa_ibm_qc_linearramp_4layer_1minsampl.json',
        #     'qaoa_ibm_qc_linearramp_5layer_1minsampl.json',
        #     ],
        
    # %% QAOA on local
        # number_of_assets = [3,5,7],
        # solve_methods = [
        #     'qaoa_localsimulator_cobyla_3layer_52sectrain_8secsampl.json',
        #     'qaoa_localsimulator_cobyla_2layer_52sectrain_8secsampl.json',
        #     'qaoa_localsimulator_cobyla_1layer_52sectrain_8secsampl.json',
        #     'qaoa_localsimulator_gridsearch_30sectrain_30secsampl.json',    
        #     'qaoa_localsimulator_linearramp_1layer_1minsampl.json',
        #     'qaoa_localsimulator_linearramp_2layer_1minsampl.json',
        #     'qaoa_localsimulator_linearramp_3layer_1minsampl.json',
        #     ],
    # %% 
    
        # reportconfigname = 'report_everything.json', 
        # reportconfigname = 'report_only_the_best_sols_and_qscore.json', 
        reportconfigname = 'report_only_the_best_sols.json', 
        
        configfilepath = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + '/config/config_files/benchmark_configs/MarkowitzPortfolio',
        configfilename = "markowitz_newbenchmarkfile.json"
        )