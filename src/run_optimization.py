# -*- coding: utf-8 -*-
"""
Created on 07.03.2024

@author: Eric Stopfer
"""
import pdb
import time
import os

from config.config import Config
from src.utils import import_utils
from src.utils import grbcluster_utils


class Opt_Run:
    
    def __init__(self, 
                 problem_config_file_path: str,
                 solver_config_file_path: str,
                 report_config_file_path: str,
                 benchmark_export_folder_path: str = "",
                 ibm_session = None,
                 abort_code_after_ibm_job_submission = False,
                 existing_ibm_job_result = None
                 ):
        #initialize the config
        self.config = Config(problem_config_file_path,
                            solver_config_file_path, 
                            report_config_file_path,
                            benchmark_export_folder_path)
        
        #initialize the ibm treatments
        self.ibm_session = ibm_session
        self.abort_code_after_ibm_job_submission = abort_code_after_ibm_job_submission
        self.existing_ibm_job_result = existing_ibm_job_result
        assert not (ibm_session != None and abort_code_after_ibm_job_submission == True), 'either ibm_session has to be None or abort_code_after_ibm_job_submission has to be False'
        assert not (ibm_session != None and existing_ibm_job_result != True), 'either ibm_session has to be None or existing_ibm_job_result has to be None'
        assert not (existing_ibm_job_result != None and abort_code_after_ibm_job_submission == True), 'either existing_ibm_job_result has to be None or abort_code_after_ibm_job_submission has to be False'
        
        #initialize classes for problem, solver and report
        self.problem_class, self.solver_class, self.report_class = import_utils.create_class_instances(self.config)
        
        #initialize a report instance
        self.report = self.report_class(self.config)
        
        
    def run_optimization(self):
        """
        function that executes the whole quantum optimization process.
        1) create the optimization model
        2) map the optimization model to a formulation fitting for the chose solving method
        3) run the optimization on the device with the desired solving method
        4) analyse the quality of the optimization result
        5) map backwards the optimization result if it's not interpretable
        6) return the report that contains the optimization results
        """                
        #create an optimization problem instance
        self.opt_problem = self.problem_class(self.config, self.report)
        
        #map the problem to a formulation that fits the configured solver
        self.opt_problem.map_problem(self.config, self.report)
        
        #solve the problem
        if self.config.solve_method == "QAOA" and self.config.solve_method_device == "IBM_Quantum_Computer" and self.ibm_session != None:
            self.solver = self.solver_class(self.config, self.report, ibm_session=self.ibm_session)
        elif self.config.solve_method == "QAOA" and self.abort_code_after_ibm_job_submission:
            self.solver = self.solver_class(self.config, self.report, abort_code_after_job_submission=self.abort_code_after_ibm_job_submission)
        else:
            self.solver = self.solver_class(self.config, self.report)
            
        if self.config.solve_method == "QAOA" and self.existing_ibm_job_result != None:
            # pseudo solver run with early exit
            self.solver.run(self.opt_problem, self.config, self.report, early_run_exit=True)
            # analyse the existing ibm result
            solver_result_feas_sol_dict, solver_result_feasibility_analysed = self.solver.analyse_solver_result(
                self.existing_ibm_job_result, self.config, self.report)
        else:
            # run the opt method
            solver_result = self.solver.run(self.opt_problem, self.config, self.report)
            if self.config.solve_method == "QAOA" and self.abort_code_after_ibm_job_submission:
                # before returning nothing, we abort all grbcluster jobs
                solver_api_tokens_path = os.path.dirname(os.path.dirname(__file__)) + "\\config" + "\\config_files\\" + "solver_api_tokens.json"
                if Config.GUROBI_LICENSE_AVAILABLE == False:
                    grbcluster_utils.abort_existing_grbcluster_jobs(solver_api_tokens_path)
                self.report.write_to_json(self.config) 
                return self.report
        
            # analyse the solver result
            solver_result_feas_sol_dict, solver_result_feasibility_analysed = self.solver.analyse_solver_result(
                solver_result, self.config, self.report)
        
        #post-process the solution
        solver_result_postprocessed = self.opt_problem.postprocess_solver_result(
            solver_result_feas_sol_dict, solver_result_feasibility_analysed, self.config, self.report)
        
        #visualize the solver result
        if self.config.report_config["visualize_feasible_solutions"] == True:
            self.opt_problem.visualize_result(solver_result_postprocessed, self.config)
        
        #save the report file 
        self.report.time_measurements["overall_calculation_time"] = time.time() - self.config.total_start_time
        self.report.write_to_json(self.config)      
        
        self.config.logger.info("The optimization run has finished successfully\n\n\n")
        
        # pdb.set_trace()
        
        return self.report
    

def execute_opt_run(problem_config_file_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config_files", "default_problem_config.json"),
                    solver_config_file_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config_files", "default_solver_config.json"),
                    report_config_file_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config_files", "default_report_config.json"),
                    benchmark_export_folder_path: str = "",
                    ibm_session = None
                    ):
    ''' wrapper function that executes an entire optimization run'''
    opt_run = Opt_Run(
        problem_config_file_path,
        solver_config_file_path,
        report_config_file_path,
        benchmark_export_folder_path,
        ibm_session
        )
    opt_report = opt_run.run_optimization()
    return opt_report

    
if __name__ == "__main__":
    execute_opt_run()
    