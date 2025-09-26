# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 12:06:09 2025

@author: stopfer
"""
import os
import pdb
import json
import shutil

def clean_subdirs_by_search_str(root_dir, search_str, search_str_in):
    ''' 
    in the root directory in the problemsolverreport_combinations of the calculation
    result folder we want to delete all folders where a search_str has been found in 
    a certain element of the problemsolverreport_combination
    '''
    assert os.path.exists(root_dir), f"{root_dir} does not exist"
    assert search_str_in in ['problem', 'solver', 'report'], f"invalid search_str_in: {search_str_in}"
    
    idx = {"problem": 0,
           "solver": 1,
           "report": 2}[search_str_in]    
    dirs_to_rm = []
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if os.path.isdir(subdir_path):
            json_path = os.path.join(subdir_path, 'problemsolverreport_combination.json')
            if os.path.isfile(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        if search_str in str(data[idx]):
                            dirs_to_rm.append((subdir_path, str(data[idx])))
    
    pdb.set_trace()
    for (subdir, problem_config) in dirs_to_rm:
        shutil.rmtree(subdir)
                                

def copy_dirs_into_folder_by_search_str(origin_dir, dest_dir, search_str, search_str_in):
    ''' 
    in the origin directory there are some result folders that should be copied 
    to the destination directory. Conditionally on a search-string that is in an element
    of the problemsolverreport_combinations those folders will be copied
    '''
    assert os.path.exists(origin_dir), f"{origin_dir} does not exist"
    assert os.path.exists(dest_dir), f"{dest_dir} does not exist"
    assert search_str_in in ['problem', 'solver', 'report'], f"invalid search_str_in: {search_str_in}"
    
    idx = {"problem": 0,
           "solver": 1,
           "report": 2}[search_str_in]
    dirs_to_copy = []
    for subdir in os.listdir(origin_dir):
        subdir_path = os.path.join(origin_dir, subdir)
        if os.path.isdir(subdir_path):
            json_path = os.path.join(subdir_path, 'problemsolverreport_combination.json')
            if os.path.isfile(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        if search_str in str(data[idx]):
                            dirs_to_copy.append((subdir_path, subdir, str(data[idx])))
    
    # pdb.set_trace()
    for (subdir_path, subdir, problem_config) in dirs_to_copy:
        shutil.copytree(subdir_path, os.path.join(dest_dir, subdir))
                              

def relocate_subdirs_by_search_str(origin_dir, dest_dir, search_str, search_str_in):
    ''' 
    in the origin directory there are some result folders that should be relocated 
    to the destination directory. Conditionally on a search-string that is in an element
    of the problemsolverreport_combinations those folders will be relocated
    '''
    assert os.path.exists(origin_dir), f"{origin_dir} does not exist"
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    assert search_str_in in ['problem', 'solver', 'report'], f"invalid search_str_in: {search_str_in}"
    
    idx = {"problem": 0,
           "solver": 1,
           "report": 2}[search_str_in]
    dirs_to_relocate = []
    for subdir in os.listdir(origin_dir):
        subdir_path = os.path.join(origin_dir, subdir)
        if os.path.isdir(subdir_path):
            json_path = os.path.join(subdir_path, 'problemsolverreport_combination.json')
            if os.path.isfile(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        if search_str in str(data[idx]):
                            dirs_to_relocate.append((subdir_path, subdir, str(data[idx])))
    
    # pdb.set_trace()
    for (subdir_path, subdir, problem_config) in dirs_to_relocate:
        shutil.move(subdir_path, os.path.join(dest_dir, subdir))
                       

def kick_out_kpi_out_of_results_from_dir(directory, kpi_name):
    ''' 
    inside of a directory we want to kick out a certain solution-quality kpi 
    of the reports of those result folders
    '''
    assert os.path.exists(directory), f"{directory} does not exist"
     
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            report_path = os.path.join(subdir_path, 'report.json')
            if os.path.isfile(report_path):
                with open(report_path, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                    assert isinstance(report_data, dict)
                    if kpi_name in report_data['solution_quality'].keys():
                        del report_data['solution_quality'][kpi_name]
                        with open(report_path, 'w') as f:
                            json.dump(report_data, f)
             

def create_ibm_jobids_file_from_dir(directory, search_str="Job-ID: "):
    ''' 
    function to read the log-documents that were created during a benchmark run
    to generate an IBM_job_ids json file that is necessary for Job analysis.
    function is necessary if benchmark did not terminate successfully
    '''
    assert os.path.exists(directory), f"{directory} does not exist"
    
    job_ids = []
    for subdir in os.listdir(directory):
        #get result folder path
        subdir_path = os.path.join(directory, subdir)
        
        if os.path.isdir(subdir_path):
            #get problem-solver-report-combo
            probsolvrep_combo_json_path = os.path.join(subdir_path, "problemsolverreport_combination.json")
            with open(probsolvrep_combo_json_path, "r") as file:
                probsolvrep_combo = json.load(file)
            
            # get job-id
            log_path = os.path.join(subdir_path, "QuOpt.log")
            if os.path.exists(log_path):
                with open(log_path, 'r') as file:
                    for line in file:
                        if search_str in line:
                            job_id = line.split(" ")[-1][:-1]
                            job_ids.append([subdir_path, probsolvrep_combo, [job_id]])
    
    job_id_file_path = os.path.join(directory, "ibm_job_ids.json")
    with open(job_id_file_path, "w") as f:
        json.dump(job_ids, f, indent=4)
        

def rename_files_for_computational_study(path, method_config):
    #get the origin and destination paths
    for f in os.listdir(path):
        if "approximation ratio.png" in f:
            approximation_ratio_file_org_path = os.path.join(path, f)
            approximation_ratio_file_dst_path = os.path.join(path, f"{method_config['identifier_for_graphics']}_approximation_ratio.png")
        elif "number of samples.png" in f:
            number_of_samples_file_org_path = os.path.join(path, f)
            number_of_samples_file_dst_path = os.path.join(path, f"{method_config['identifier_for_graphics']}_numberofshots.png")
        elif "feasibility percentage.png" in f:
            feasibility_perc_file_org_path = os.path.join(path, f)
            feasibility_perc_file_dst_path = os.path.join(path, f"{method_config['identifier_for_graphics']}_problemswithfeasiblesfound.png")
            
    #delete old files
    if os.path.exists(approximation_ratio_file_dst_path):
        os.remove(approximation_ratio_file_dst_path)
    if os.path.exists(number_of_samples_file_dst_path):
        os.remove(number_of_samples_file_dst_path)
    if os.path.exists(feasibility_perc_file_dst_path):
        os.remove(feasibility_perc_file_dst_path)
        
    #rename the result files
    os.rename(src = approximation_ratio_file_org_path, dst = approximation_ratio_file_dst_path)
    os.rename(src = number_of_samples_file_org_path, dst = number_of_samples_file_dst_path)
    os.rename(src = feasibility_perc_file_org_path, dst = feasibility_perc_file_dst_path)  


if __name__ == "__main__":
    a=0
    
    # %% maxret new calculation
    
    # clean_subdirs_by_search_str(
    #     root_dir=r"C:\Users\stopfer\Documents\git\quopt\results\benchmarkrun_portfolioopt_scip", 
    #     search_str='maxret',
    #     search_str_in='problem'
    #     )    
    # copy_dirs_into_folder_by_search_str(
    #     origin_dir=r"C:\Users\stopfer\Documents\git\quopt\results\benchmark_mipsolver_maxret", 
    #     dest_dir=r"C:\Users\stopfer\Documents\git\quopt\results\benchmarkrun_portfolioopt_scip", 
    #     search_str="scip", 
    #     search_str_in="solver"
    #     )
    
    # clean_subdirs_by_search_str(
    #     root_dir=r"C:\Users\stopfer\Documents\git\quopt\results\benchmarkrun_portfolioopt_gurobi", 
    #     search_str='maxret',
    #     search_str_in='problem'
    #     )    
    # copy_dirs_into_folder_by_search_str(
    #     origin_dir=r"C:\Users\stopfer\Documents\git\quopt\results\benchmark_mipsolver_maxret", 
    #     dest_dir=r"C:\Users\stopfer\Documents\git\quopt\results\benchmarkrun_portfolioopt_gurobi", 
    #     search_str="gurobi", 
    #     search_str_in="solver"
    #     )
    
    # %% copy gurobi optimal sols into heuristics
    
    # copy_dirs_into_folder_by_search_str(
    #     origin_dir=r"C:\Users\stopfer\Documents\git\quopt\results\benchmarkrun_portfolioopt_gurobi", 
    #     dest_dir=r"C:\Users\stopfer\Documents\git\quopt\results\benchmarkrun_portfolioopt_heuristics", 
    #     search_str="mipsolver_gurobi_timelimit1hour.json", 
    #     search_str_in="solver"
    #     )
    
    # %% multiobj new calculation
    
    # clean_subdirs_by_search_str(
    #     root_dir=r"C:\Users\stopfer\Documents\git\quopt\results\benchmarkrun_portfolioopt_scip_fromotherfolder", 
    #     search_str='multiobj',
    #     search_str_in='problem'
    #     )    
    # copy_dirs_into_folder_by_search_str(
    #     origin_dir=r"C:\Users\stopfer\Documents\git\quopt\results\benchmarkrun_portfolioopt_mipsolver_multiobj", 
    #     dest_dir=r"C:\Users\stopfer\Documents\git\quopt\results\benchmarkrun_portfolioopt_scip_fromotherfolder", 
    #     search_str="scip", 
    #     search_str_in="solver"
    #     )
    
    # clean_subdirs_by_search_str(
    #     root_dir=r"C:\Users\stopfer\Documents\git\quopt\results\benchmarkrun_portfolioopt_gurobi_fromotherfolder", 
    #     search_str='multiobj',
    #     search_str_in='problem'
    #     )    
    # copy_dirs_into_folder_by_search_str(
    #     origin_dir=r"C:\Users\stopfer\Documents\git\quopt\results\benchmarkrun_portfolioopt_mipsolver_multiobj", 
    #     dest_dir=r"C:\Users\stopfer\Documents\git\quopt\results\benchmarkrun_portfolioopt_gurobi_fromotherfolder", 
    #     search_str="gurobi", 
    #     search_str_in="solver"
    #     )
    
    # %% copy gurobi optimal sols into quantumanneal
    
    # copy_dirs_into_folder_by_search_str(
    #     origin_dir=r"C:\Users\stopfer\Documents\git\quopt\results\benchmarkrun_portfolioopt_heuristics_new", 
    #     dest_dir=r"C:\Users\stopfer\Documents\git\quopt\results\benchmarkrun_portfolioopt_quantumanneal", 
    #     search_str="mipsolver_gurobi_timelimit1hour.json", 
    #     search_str_in="solver"
    #     )
    
    # %% create IBM Job-Ids file
    
    # create_ibm_jobids_file_from_dir(
    #     directory = r"C:\Users\stopfer\Documents\quopt\results\benchmarkrun_2025_07_14_12_12_06"
    #     )
    
    
    # %% relocate heuristics from heuristics result folder --> will get replaced
    
    # relocate_subdirs_by_search_str(
    #     origin_dir=r"C:\Users\stopfer\Documents\git\quopt\results\benchmarkrun_portfolioopt_heuristics_new", 
    #     dest_dir=r"C:\Users\stopfer\Documents\git\quopt\results\benchmarkrun_portfolioopt_heuristics_old", 
    #     search_str='greedy',
    #     search_str_in='solver'
    #     )    
    # relocate_subdirs_by_search_str(
    #     origin_dir=r"C:\Users\stopfer\Documents\git\quopt\results\benchmarkrun_portfolioopt_heuristics_new", 
    #     dest_dir=r"C:\Users\stopfer\Documents\git\quopt\results\benchmarkrun_portfolioopt_heuristics_old", 
    #     search_str='simulated',
    #     search_str_in='solver'
    #     )  
    # relocate_subdirs_by_search_str(
    #     origin_dir=r"C:\Users\stopfer\Documents\git\quopt\results\benchmarkrun_portfolioopt_heuristics_new", 
    #     dest_dir=r"C:\Users\stopfer\Documents\git\quopt\results\benchmarkrun_portfolioopt_heuristics_old", 
    #     search_str='tabu',
    #     search_str_in='solver'
    #     )  
    
    # %% delete old randomsampler experiments --> I improved the implementation and generated new results
    # clean_subdirs_by_search_str(
    #     root_dir = r"C:\Users\stopfer\Documents\git\quopt\results\benchmarkrun_portfolioopt_heuristics_new",
    #     search_str = 'randomsamplingqubo_readsfor1min.json',
    #     search_str_in = 'solver')
    
    # clean_subdirs_by_search_str(
    #     root_dir = r"C:\Users\stopfer\Documents\git\quopt\results\benchmarkrun_portfolioopt_heuristics_new",
    #     search_str = 'randomsamplingqubo_10000reads.json',
    #     search_str_in = 'solver')
    
    
    