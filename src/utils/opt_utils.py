# -*- coding: utf-8 -*-
"""
Created on 24.05.2024

@author: Eric Stopfer
"""
import os
import time
import pdb
from pyscipopt import Model as scipmodel
from gurobipy import Model as gurmodel, Env, GRB
import copy
import json

from config.config import Config
from src.report import Report
from src.solvers.Opt_Heuristics import Opt_Heuristics
from src.solvers.Greedy_Algorithm import Greedy_Algorithm




class Opt_Model:
    
    def __init__(self, model_type: str, model_name: str, config: Config): 
        ''' initializes an optimization model. currently gurobipy or pyscipopt is implemented '''
        self.model_type = model_type
        self.model_name = model_name
        self._create_initial_model(config)
            
    def _create_initial_model(self, config):
        ''' creates an initial model '''
        if self.model_type == 'gurobipy':
            if config.GUROBI_LICENSE_AVAILABLE == False or (config.solve_method == "MIP_Solver" and config.solve_method_device == "Gurobi_ComputeServer"):
                self.gurobi_env = self._create_gurobi_compute_server_env(config)
                self.model = gurmodel(self.model_name, env=self.gurobi_env)
            else:
                self.model = gurmodel(self.model_name, env=None)
            
            self.model.setParam("LogFile", os.path.join(config.EXPORT_PATH, config.LOG_FILE_NAME))
                
        elif self.model_type == 'pyscipopt':
            self.model = scipmodel(self.model_name)
            self.model.setLogfile(os.path.join(config.EXPORT_PATH, config.LOG_FILE_NAME))
        else:
            raise Exception("invalid model_type. must be either 'gurobipy' or 'pyscipopt'")
            
    @staticmethod
    def _create_gurobi_compute_server_env(config):
        ''' function that creates and returns a gurobi compute server environment '''         
        solver_api_tokens_path = os.path.join(os.path.join(config.BASE_PATH_CONFIG, "config_files"), "solver_api_tokens.json")
        with open(solver_api_tokens_path, "r") as json_file:
            solver_api_tokens_data = json.load(json_file)
            
        env = Env(empty=True)
        env.setParam(GRB.Param.ComputeServer, solver_api_tokens_data['GUROBI_COMPUTE_SERVER_NAME'])
        env.setParam(GRB.Param.ServerPassword, solver_api_tokens_data['GUROBI_COMPUTE_SERVER_PASS'])
        env.setParam(GRB.Param.CSPriority, solver_api_tokens_data['GUROBI_COMPUTE_SERVER_PRIO'])
        env.start()
        
        return env
    
    def close_gurobi_server_env(self):
        ''' function that closes a gurobi compute server environment '''
        self.gurobi_env.close()


    def add_variable(self, name, lb=0.0, ub=None, vtype="C"):
        ''' add variable to opt_model. currently gurobipy or pyscipopt is implemented  '''
        assert vtype in {"C", "I", "B"}, "Invalid variable type, must be 'C', 'I', or 'B'"
        
        if self.model_type == 'pyscipopt':
            var = self.model.addVar(name=name, lb=lb, ub=ub, vtype=vtype)
            return var
            
        elif self.model_type == "gurobipy":
            vartype = {"C": GRB.CONTINUOUS, "I": GRB.INTEGER, "B": GRB.BINARY}[vtype]
            ub = float("inf") if ub==None else ub
            var = self.model.addVar(name=name, lb=lb, ub=ub, vtype=vartype)
            self.model.update()
            return var
            
    
    def add_constraint(self, lhs, sense, rhs, name):
        ''' add constraint to opt_model. currently gurobipy or pyscipopt is implemented  '''
        assert sense in {"<=", ">=", "=="}, "Invalid constraint sense, must be '<=', '>=', or '=='"
        
        if self.model_type == 'pyscipopt':
            if sense == "<=":
                self.model.addCons(lhs <= rhs, name=name)
            elif sense == ">=":
                self.model.addCons(lhs >= rhs, name=name)
            elif sense == "==":
                self.model.addCons(lhs == rhs, name=name)
                
        elif self.model_type == "gurobipy":
            if sense == "<=":
                self.model.addConstr(lhs <= rhs, name=name)
            elif sense == ">=":
                self.model.addConstr(lhs >= rhs, name=name)
            elif sense == "==":
                self.model.addConstr(lhs == rhs, name=name)
                
    
    def set_objective(self, obj, sense = 'min', quadratic_objective = False):
        ''' set objective of opt_model. currently gurobipy or pyscipopt is implemented  ''' 
        assert sense in {"min", "max"}, "Invalid objective sense, must be 'min' or 'max'"
        
        if self.model_type == 'pyscipopt':
            objsense = 'minimize' if sense == "min" else 'maximize'
            if not quadratic_objective:
                self.model.setObjective(obj, sense=objsense)
            else:
                #reformulation necessary because scip can't handle quadratic objective
                quadobjvar = self.model.addVar('quadobjvar', lb=None, vtype='C')
                self.model.setObjective(quadobjvar, sense=objsense)
                if objsense == 'minimize':
                    self.model.addCons(quadobjvar >= obj, name="define_quadobjvar")
                else:
                    self.model.addCons(quadobjvar <= obj, name="define_quadobjvar")
        elif self.model_type == "gurobipy":
            self.model.setObjective(obj, GRB.MINIMIZE if sense == "min" else GRB.MAXIMIZE)
            self.model.update()
           
            
    def solve_model(self, 
                    time_limit: int, 
                    absolute_gap: float,
                    relative_gap: float,
                    presolve_method: int):
        ''' solve the opt_model. currently gurobipy or pyscipopt is implemented  '''
        if self.model_type == 'pyscipopt':
            self.model.setRealParam('limits/absgap', absolute_gap)
            self.model.setRealParam('limits/gap', relative_gap)
            self.model.setRealParam('limits/time', time_limit)
            self.model.setPresolve(presolve_method)
            start_time = time.time()
            self.model.optimize()
            self.scip_solving_time = time.time() - start_time
            
        elif self.model_type == "gurobipy":
            self.model.params.MipGap = relative_gap
            self.model.params.MipGapAbs = absolute_gap
            self.model.params.TimeLimit = time_limit  
            # self.model.params.NonConvex = 0
            
            self.model.optimize()
      
        
    def write_model(self, path, filename):
        ''' write the model to a file '''
        if self.model_type == 'pyscipopt':
            self.model.writeProblem(os.path.join(path, filename))
            
        elif self.model_type == "gurobipy":
            self.model.write(os.path.join(path, filename))

   
    def get_number_of_variables(self):
        ''' calculates and returns the number of variables of self.model'''
        if self.model_type == 'pyscipopt':
            number_of_vars = self.model.getNVars()
        elif self.model_type == "gurobipy":
            number_of_vars = len(self.model.getVars())
        return number_of_vars
           
    
    def get_varnames(self):
        ''' returns a list of the varnames of the model '''
        if self.model_type == 'pyscipopt':
            varnames = [v.name for v in self.model.getVars()]
        elif self.model_type == "gurobipy":
            varnames = [v.VarName for v in self.model.getVars()]
        return varnames        
     
            
    def get_solving_time(self):
        ''' get the solving time of an opt model '''
        if self.model_type == 'pyscipopt':
            solving_time = self.scip_solving_time
        elif self.model_type == "gurobipy":
            solving_time = self.model.Runtime
        return solving_time
        
    def get_opt_status(self):
        ''' get the optimization status '''
        
        if self.model_type == 'pyscipopt':
            opt_status = self.model.getStatus()
        elif self.model_type == "gurobipy":
            #status mapping dict maps to the same wordings as in pyscipopt
            status_mapping_dict = {GRB.OPTIMAL: "optimal",
                                   GRB.INFEASIBLE: "infeasible",
                                   GRB.UNBOUNDED: "unbounded",
                                   GRB.TIME_LIMIT: "timelimit",
                                   GRB.SUBOPTIMAL: "gaplimit"}
            if self.model.Status in status_mapping_dict:
                opt_status = status_mapping_dict[self.model.Status]
            else:
                raise Exception(f"treatment of gurobi solve status {self.model.Status} not implemented")
        return opt_status
    
    
    def get_best_found_obj_value(self):
        ''' get the best found objective value during the optimization '''
        if self.model_type == 'pyscipopt':
            best_obj_value = self.model.getObjVal()
        elif self.model_type == "gurobipy":
            best_obj_value = self.model.ObjVal
        return best_obj_value
        

def solve_opt_model_with_SCIP(
        gurobi_model, 
        time_limit: int, 
        absolute_gap: float,
        relative_gap: float,
        presolve_method: int, 
        config: Config,
        report: Report
        ) -> (dict, float):
    ''' 
    function to solve an optimization model that is read from a LP-file
    with SCIP solver that returns the model in its final state
    '''
    mps_file_path = os.path.join(config.EXPORT_PATH, "MIP_to_be_solved_by_mipsolver.mps")
    gurobi_model.write(mps_file_path)
    
    scip_model = scipmodel()
    scip_model.readProblem(mps_file_path)
    
    scip_model.setRealParam('limits/absgap', absolute_gap)
    scip_model.setRealParam('limits/gap', relative_gap)
    scip_model.setRealParam('limits/time', time_limit)
    scip_model.setPresolve(presolve_method)
    
    scip_model.optimize()
    
    return scip_model
        

def create_gurobi_compute_server_env(config):
    ''' 
    function that creates and returns a gurobi compute server environment
    '''
    solver_api_tokens_path = config.BASE_PATH_CONFIG + "\\config_files\\" + "solver_api_tokens.json"
    with open(solver_api_tokens_path, "r") as json_file:
        solver_api_tokens_data = json.load(json_file)
    
    env = Env(empty=True)
    env.setParam(GRB.Param.ComputeServer, solver_api_tokens_data['GUROBI_COMPUTE_SERVER_NAME'])
    env.setParam(GRB.Param.ServerPassword, solver_api_tokens_data['GUROBI_COMPUTE_SERVER_PASS'])
    env.setParam(GRB.Param.CSPriority, solver_api_tokens_data['GUROBI_COMPUTE_SERVER_PRIO'])
    env.start()
    
    return env


def solve_opt_model_with_gurobi(
        gurobi_model,
        time_limit: int, 
        absolute_gap: float,
        relative_gap: float,
        presolve_method: int, 
        config: Config,
        report: Report
        ) -> (dict, float):
    ''' 
    function to solve an optimization model 
    with Gurobi solver that returns the model in its final state
    '''
    gurobi_model.params.MipGap = relative_gap
    gurobi_model.params.MipGapAbs = absolute_gap
    gurobi_model.params.TimeLimit = time_limit    
    gurobi_model.optimize()
    
    return gurobi_model


def execute_steepest_descent_on_bitstrings(opt_problem: any,
                                           feas_sol_dict: dict, 
                                           config: Config, 
                                           report: Report):
    '''
    function to postprocess optimization result bitstrings with steepest_descent greedy algorithm
    to possibly get even better results
    '''
    if feas_sol_dict == {}:    
        return {}, {}
    else:
        steepest_desc_config = copy.deepcopy(config)
        steepest_desc_config.solve_method_device = "Dwave_GreedyAlgorithm"
        steepest_desc_config.solve_method_config = {"time_limit": 100, 
                                                    "number_of_reads": len(list(feas_sol_dict.keys())),
                                                    "max_calculation_time_exists": False
                                                    }  
        var_dict_list = []
        for bit_tuple, bit_tuple_dict in feas_sol_dict.items():
            var_dict_list.append(bit_tuple_dict['var_dict'])
        
        steepestdescentsolver = Greedy_Algorithm(steepest_desc_config, report)
        greedy_alg_response = steepestdescentsolver.run(
            opt_problem=opt_problem,
            config=steepest_desc_config, 
            report=report, 
            initial_states=var_dict_list)
        greedy_alg_result_dict, greedy_alg_result_dict_feas_analysed = steepestdescentsolver.analyse_solver_result(greedy_alg_response, steepest_desc_config, report)    
        return greedy_alg_result_dict, greedy_alg_result_dict_feas_analysed


def execute_heuristic_for_initial_states(opt_problem: any,
                                         number_of_initial_states: int,
                                         config: Config, 
                                         report: Report):
    '''
    function to preprocess an optimization problem to get initial 
    feasible solutions by a heuristic
    '''
    opt_heuristic_config = copy.deepcopy(config)
    opt_heuristic_config.solve_method = "Opt_Heuristics"
    opt_heuristic_config.solve_method_device = "no device needed"
    opt_heuristic_config.solve_method_config = {"number_of_samples": number_of_initial_states,
                                                "max_calculation_time_exists": True,
                                                "max_calculation_time_in_min": 1,
                                                "heuristic_name": "default"
                                                }
    opt_heuristics = Opt_Heuristics(opt_heuristic_config, report)
    (_, _, discretized_states_from_heuristic_dict_list) = opt_heuristics.run(opt_problem, opt_heuristic_config, report)
    return discretized_states_from_heuristic_dict_list

