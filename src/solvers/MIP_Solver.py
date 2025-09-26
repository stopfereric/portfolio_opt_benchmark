# -*- coding: utf-8 -*-
"""
Created on 07.03.2024

@author: Eric Stopfer
"""
import time
import pdb
import os

import pyscipopt as scip_opt
import gurobipy as gp

from config.config import Config
from src.report import Report
from src.utils.error_utils import CustomizedError
from src.utils.opt_utils import solve_opt_model_with_SCIP, solve_opt_model_with_gurobi
from src.utils.solver_result_utils import analyse_SCIP_result, analyse_gurobi_result, write_kpis_to_report



class MIP_Solver:
    '''
    is able to solve Mixed-Integer-Programs
    '''
    def __init__(self, config: Config, report: Report):
        """
        function that creates a MIP solver instance (-> Quantum Approximate Optimization Algorithm)
        """
        config.logger.info("we are using the following solving method: %s" %config.solve_method)
        config.logger.info("with the following device: %s" %config.solve_method_device)
        
        
    def run(self, opt_problem, config: Config, report: Report) -> any:
        '''
        Function that runs an optimization process for a Mixed-Integer-Program
        Current options are SCIP-Solver and Gurobi-Solver.
        '''
        #read the necessary attributes from the configuration
        try:
            time_limit = config.solve_method_config['time_limit']
            absolute_gap = config.solve_method_config['abs_gap']
            relative_gap = config.solve_method_config['rel_gap']
            presolve_method = config.solve_method_config['presolve_method']
            initial_lp_algorithm = config.solve_method_config['initial_lp_algorithm']
            
        except:
            raise CustomizedError(error_message="wrong or missing configuration of the MIP-Solver. check in config.py and config.json")
        
        self.mapped_opt_problem = opt_problem.problem_mapping
        
        self.mapped_opt_problem.solve_model( 
            time_limit = time_limit, 
            absolute_gap = absolute_gap,
            relative_gap = relative_gap,
            presolve_method = presolve_method
            )
        
        report.time_measurements['opt_method_run_total_time_consumption'] = self.mapped_opt_problem.get_solving_time()
        
        return self.mapped_opt_problem
            
    
    def analyse_solver_result(self, solver_result, config: Config, report: Report):
        '''
        function that analyses the result of the MIP-solver
        '''
        start_time = time.time()
        
        if config.solve_method_device == "SCIP":
            solution_dict = analyse_SCIP_result(solver_result, config.problem_name, config, report)
        elif config.solve_method_device in ["Gurobi_LocalLicense", "Gurobi_ComputeServer"]:
            solution_dict = analyse_gurobi_result(solver_result, config.problem_name, config, report)
        else:
            raise Exception(f"unknown solve method device {config.solve_method_device}")
        write_kpis_to_report(solution_dict, 1, config, report)
        
        report.time_measurements['opt_method_result_analysis_total_time_consumption'] = time.time() - start_time
        return solution_dict, solution_dict
    
    