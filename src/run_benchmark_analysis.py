# -*- coding: utf-8 -*-
"""
Created on 07.03.2024

@author: Eric Stopfer
"""
import os

from src.run_multiple_optimizations import Benchmark
from src.utils.resultfolder_utils import rename_files_for_computational_study

quantum_anneal_dict = {
    'identifier_for_graphics': 'annealing',
    'min_problem_size_for_graphics': 5,
    'max_problem_size_for_graphics': 25,
    'solving_methods': {
        'optheuristic_default_timelimit1min.json': "Problem-specific heuristic",
        'randomsamplingqubo_readsfor1min.json': "Random sampling 60s",
        'randomsamplingqubo_330000reads.json': "Random sampling 330,000 samples",
        ('quantumanneal_dwave_readsfor1min_20annealingtime.json',
         'quantumanneal_dwave_readsfor1min_20annealingtime_defaultcs.json'): r"QA $\tau$=20µs, default cs",
        ('quantumanneal_dwave_readsfor1min_50annealingtime.json',
         'quantumanneal_dwave_readsfor1min_50annealingtime_defaultcs.json'): r"QA $\tau$=50µs, default cs",
        ('quantumanneal_dwave_readsfor1min_20annealingtime_3cs.json',
         'quantumanneal_dwave_readsfor1min_20annealingtime_4cs.json',
         'quantumanneal_dwave_readsfor1min_20annealingtime_5cs.json'): r"QA $\tau$=20µs, adjusted cs",
        ('quantumanneal_dwave_readsfor1min_50annealingtime_3cs.json',
         'quantumanneal_dwave_readsfor1min_50annealingtime_4cs.json',
         'quantumanneal_dwave_readsfor1min_50annealingtime_5cs.json'): r"QA $\tau$=50µs, adjusted cs",
    }
}
qaoa_dict = {
    'identifier_for_graphics': 'qaoa',
    'min_problem_size_for_graphics': 7,
    'max_problem_size_for_graphics': 30,
    'solving_methods': {
        'optheuristic_default_timelimit1min.json': "Problem-specific heuristic",
        'randomsamplingqubo_readsfor1min.json': "Random sampling 60s",
        'randomsamplingqubo_100000reads.json': "Random sampling 100,000 samples",
        'qaoa_ibm_qc_gridsearch_30sectrain_30secsampl.json': "QAOA p=1, grid search",
        'qaoa_ibm_qc_linearramp_1layer_1minsampl.json': "QAOA p=1, linear ramp",
        'qaoa_ibm_qc_linearramp_3layer_1minsampl.json': "QAOA p=3, linear ramp",
        'qaoa_ibm_qc_linearramp_5layer_1minsampl.json': "QAOA p=5, linear ramp",
    }
}
heuristics_dict = {
    'identifier_for_graphics': 'heuristics',
    'min_problem_size_for_graphics': 5,
    'max_problem_size_for_graphics': 800,
    'solving_methods': {
        'optheuristic_default_timelimit1min.json': "Problem-specific heuristic",
        'randomsamplingqubo_readsfor1min.json': "Random sampling 60s",
        'greedy_dwavegreedyalgorithm_readsfor1min': "Steepest descent",
        'simulatedanneal_dwave_readsfor1min': "Simulated annealing",
        'tabusearch_dwavetabusampler_readsfor1min_20timelimit.json': "Tabu search",
    }
}
best_method_dict = {
    'identifier_for_graphics': 'bestmethod',
    'min_problem_size_for_graphics': 7,
    'max_problem_size_for_graphics': 25,
    'solving_methods': {
        'optheuristic_default_timelimit1min.json': "Problem-specific heuristic",
        'randomsamplingqubo_readsfor1min.json': "Random sampling 60s",
        ('quantumanneal_dwave_readsfor1min_50annealingtime_3cs.json',
         'quantumanneal_dwave_readsfor1min_50annealingtime_4cs.json',
         'quantumanneal_dwave_readsfor1min_50annealingtime_5cs.json'): r"QA $\tau$=50µs, adjusted cs",
        'qaoa_ibm_qc_linearramp_1layer_1minsampl.json': "QAOA p=1, linear ramp",
    }
}
visualisation_config_dicts = [
    quantum_anneal_dict,
    qaoa_dict,
    heuristics_dict,
    best_method_dict]



if __name__ == "__main__":
    analyse_existing_results = True
    analyse_ibm_job_results = False
    assert not analyse_existing_results or not analyse_ibm_job_results, "either analyse_existing_results OR analyse_ibm_job_results, not both"

    config_file_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "config", "config_files", "benchmark_config.json"
        )

    if analyse_existing_results:
        existing_results_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "results",
            "benchmarkrun_2025_10_07_11_44_43"
            )
        assert os.path.exists(existing_results_path), f"results path '{existing_results_path}' does not exist. You must specify in the code."

        for method_config in visualisation_config_dicts:
            benchmark = Benchmark(config_file_path, method_config)
            benchmark.create_benchmark_export_folder(existing_results_path)
            benchmark.calc_optimality_kpis()
            benchmark.analyse_benchmark_opt_run_results()
            rename_files_for_computational_study(path=benchmark.BENCHMARK_EXPORT_PATH,
                                                 method_config=method_config)

    elif analyse_ibm_job_results:
        existing_results_path_ibm_results = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "results",
            "benchmarkrun_2025_07_03_18_12_10"
            )
        assert os.path.exists(existing_results_path_ibm_results), f"IBM results path '{existing_results_path_ibm_results}' does not exist. You must specify in the code."

        benchmark = Benchmark(config_file_path)
        benchmark.create_benchmark_export_folder(folder_name = os.path.basename(existing_results_path_ibm_results) + "_ibm_analysis")
        benchmark.analyse_ibm_job_results(existing_results_path_ibm_results)
        benchmark.calc_optimality_kpis()
        benchmark.analyse_benchmark_opt_run_results()

    
    
    