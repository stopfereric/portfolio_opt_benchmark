# -*- coding: utf-8 -*-
"""
Created on 07.03.2024

@author: Eric Stopfer
"""
import os
import importlib
import sys

from config.config import Config

def create_class_instances(config: Config):
    """
    function that gets the classes of the problem, solver and report
    depending on the information given in the configuration
    """
    # %% get the problem class
    all_problems_folder_path = os.path.join(config.BASE_PATH_SRC, "problems")
    problem_path = all_problems_folder_path + '\\' + config.problem_name
    sys.path.append(problem_path)
    problem_file = importlib.import_module(config.problem_name)
    problem_class = vars(problem_file)[config.problem_name]
    
    # %% get the solver class
    solver_path = os.path.join(config.BASE_PATH_SRC, "solvers")
    sys.path.append(solver_path)
    solver_file = importlib.import_module(config.solve_method)
    solver_class = vars(solver_file)[config.solve_method]
    
    # %% get the report class
    report_path = config.BASE_PATH_SRC
    sys.path.append(report_path)
    report_file = importlib.import_module("report")
    report_class = vars(report_file)["Report"]
    
    # %%   
    return problem_class, solver_class, report_class

