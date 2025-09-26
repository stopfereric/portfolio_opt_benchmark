# -*- coding: utf-8 -*-
"""
Created on 24.05.2024

@author: Eric Stopfer
"""
import pdb
import time
import pandas as pd
import numpy as np

import gurobipy
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.problems.constraint import ConstraintSense

from src.utils.error_utils import CustomizedError
from src.report import Report
from config.config import Config


def analyse_SCIP_result(opt_model, problem_name: str, config: Config, report: Report):
    '''
    function that analyses a SCIP solver result and outputs a solution_dict with all the
    important result information if the model isn't infeasible or unbounded
    '''    
    start_time = time.time()
    if opt_model.model.getStatus() not in ['infeasible', 'unbounded']: 
        solution_dict = {} #initialize
        var_dict = {} #initialize
        bit_tuple = () #initialize
        
        solution = opt_model.model.getBestSol()
        obj_value = opt_model.model.getObjVal()
        used_quadobjvar = False # if the objective is quadratic, scip introduces a variable that represents the quadratic objective, we want to delte this variable in the end
        for var in opt_model.model.getVars():
            var_name = var.__repr__()
            if var_name in ['quadobjvar']:
                used_quadobjvar = True
            var_value = solution[var]
            bit_tuple = bit_tuple + (var_value,)
            var_dict[var_name] = var_value
        
        if used_quadobjvar == True:
            bit_str = ', '.join([str(item) for item in bit_tuple[:-1]])
        else:
            bit_str = ', '.join([str(item) for item in bit_tuple])
            
        solution_dict[bit_str] = {}
        solution_dict[bit_str]['var_dict'] = var_dict
        solution_dict[bit_str]['mip_obj_value'] = obj_value
        solution_dict[bit_str]['solution_status'] = opt_model.model.getStatus()
        solution_dict[bit_str]['count'] = 1
        
        report.solution_quality[problem_name+'_'+'best_objective_value'] = obj_value
        report.solution_quality[problem_name+'_'+'optimization_status'] = opt_model.model.getStatus()
        report.solution_quality[problem_name+'_'+'relative_gap'] = opt_model.model.getGap() # |(primalbound - dualbound) / min(|primalbound|, |dualbound|)|
        report.time_measurements[problem_name+'_'+'scip_read_opt_results'] = time.time() - start_time
    else: # if infeasible or unbounded
        report.solution_quality[problem_name+'_'+'optimization_status'] = opt_model.model.getStatus()
        report.time_measurements[problem_name+'_'+'scip_read_opt_results'] = time.time() - start_time
        solution_dict = {}
    
    return solution_dict 
            

def analyse_gurobi_result(opt_model, problem_name: str, config: Config, report: Report):
    '''
    function that analyses a Gurobi solver result and outputs a solution_dict with all the
    important result information if the model isn't infeasible or unbounded
    '''    
    start_time = time.time()
    if opt_model.model.Status not in [gurobipy.GRB.INFEASIBLE, gurobipy.GRB.UNBOUNDED]:
        solution_dict = {}
        var_dict = {}
        bit_tuple = ()
        
        used_quadobjvar = False
        
        for var in opt_model.model.getVars():
            var_name = var.VarName
            if var_name == 'quadobjvar':
                used_quadobjvar = True
            var_value = var.X
            bit_tuple = bit_tuple + (var_value,)
            var_dict[var_name] = var_value
        
        if used_quadobjvar:
            bit_str = ', '.join([str(item) for item in bit_tuple[:-1]])
        else:
            bit_str = ', '.join([str(item) for item in bit_tuple])
        
        solution_dict[bit_str] = {}
        solution_dict[bit_str]['var_dict'] = var_dict
        solution_dict[bit_str]['mip_obj_value'] = opt_model.model.ObjVal
        solution_dict[bit_str]['solution_status'] = opt_model.model.Status
        solution_dict[bit_str]['count'] = 1
        
        report.solution_quality[problem_name + '_' + 'best_objective_value'] = opt_model.model.ObjVal
        report.solution_quality[problem_name + '_' + 'optimization_status'] = opt_model.model.Status
        if opt_model.model.SolCount > 0 and opt_model.model.IsMIP:
            report.solution_quality[problem_name + '_' + 'relative_gap'] = opt_model.model.MIPGap
        else:
            report.solution_quality[problem_name + '_' + 'relative_gap'] = None
        report.time_measurements[problem_name + '_' + 'gurobi_read_opt_results'] = time.time() - start_time
    else:
        report.solution_quality[problem_name + '_' + 'optimization_status'] = opt_model.model.Status
        report.time_measurements[problem_name + '_' + 'gurobi_read_opt_results'] = time.time() - start_time
        solution_dict = {}
    
    return solution_dict

    
def process_dwave_response(dwave_response_list, mip_variable_names_list, config: Config, report: Report) -> dict:
    ''' 
    function to process the response of a dwave sampler into a readable dictionary. 
    the variables of the dwave response are reordered accordingly to the qiskit QuadraticProgram
    '''
    start_time = time.time()
    
    result_dict = {} #will be filled
    qpu_times = [] # initialize
    qpu_chainbreak_fractions = [] # initialize
    qpu_problem_ids = [] # initialize
    for dwave_response in dwave_response_list:                
        for var_dict, energy, count in dwave_response.data(['sample', 'energy', 'num_occurrences']): 
            bitstring = ', '.join([str(var_dict[var_name]) for var_name in mip_variable_names_list])
            if bitstring not in result_dict.keys():
                result_dict[bitstring] = {}
                result_dict[bitstring]['energy'] = float(energy)
                result_dict[bitstring]['count'] = int(count)
                result_dict[bitstring]['var_dict'] = {key: int(val) for key, val in var_dict.items()}
            else:
                result_dict[bitstring]['count'] += int(count)
        try:
            qpu_times.append(dwave_response.info['timing']['qpu_access_time'] / 1000000) # given in microseconds
            qpu_chainbreak_fractions.append(dwave_response.record.chain_break_fraction.mean())
            qpu_problem_ids.append(dwave_response.info['problem_id'])
        except:
            pass # otherwise wasn't executed on a real QC
    
    if len(qpu_times) > 0:
        report.time_measurements['qpu_times'] = qpu_times 
        report.qpu_problem_ids = qpu_problem_ids
        overall_chainbreak_fraction = np.array(qpu_chainbreak_fractions).mean()
        overall_qpu_time = sum(qpu_times)
        report.solution_quality["chain_break_fraction"] = overall_chainbreak_fraction
        report.time_measurements['opt_method_run_total_time_consumption'] = overall_qpu_time
        config.logger.info(f"\t hardware stats: chainbreak fraction: {round(100*overall_chainbreak_fraction, 2)}%")
        config.logger.info(f"\t hardware stats: time spent on QPU: {round(overall_qpu_time, 2)}s")
    
       
    if config.report_config["analyse_only_best_sols"] == True:
        sorted_items = sorted(result_dict.items(), key=lambda item: item[1]['energy'])
        number_of_sols_to_keep = config.report_config["number_of_sols_to_be_analysed"]
        result_dict = dict(sorted_items[:number_of_sols_to_keep])
        config.logger.info(f"We only keep the {number_of_sols_to_keep} best solutions")
    
    report.time_measurements['process_dwave_response'] = time.time() - start_time
    return result_dict


def get_list_for_reordering_bits_in_bitstring(dwave_response, mip_variable_names_list):
    '''
    function to create an index list to reorder a bitstring that resulted from a dwave sampling process. 
    E.g. we have the original variable names of a MIP: ('x0', 'x1', 'x2', ..., 'x10'), 
         then dwave orders the variables alphabetically: ('x0', 'x1', 'x10', 'x2', ...)
         --> we have to reorder the dwave result bitstring
    '''        
    dwave_vars = []
    for var, _ in dwave_response.first[0].items():
        dwave_vars.append(var)
    
    index_list_for_reordering = [dwave_vars.index(element) for element in mip_variable_names_list]
    
    return index_list_for_reordering


def analyse_feasibility_of_solver_solutions( result_dict: dict, 
                                             qubo: QuadraticProgram, 
                                             bp_before_qubo: QuadraticProgram,
                                             config: Config,
                                             report: Report,
                                             detailed_feasibility_analysis: bool = True
                                             ) -> dict:
    '''
    function to analyse the feasibility and objective value of the bitstrings 
    in the result_dict. 
    The results are written inside the result_dict dictionary
    '''
    start_time = time.time()
    config.logger.info("start to analyse feasibility of a solver_result-dictionary")
    counter = 0
    for bit_str, bit_tuple_attr in result_dict.items():
        bit_list = [float(item) if '.' in item else int(item) for item in bit_str.split(', ')]
        qubo_obj_value = qubo.objective.evaluate(bit_list)
        bp_before_qubo_obj_value = bp_before_qubo.objective.evaluate(bit_list)
        violated_constraints = []
        if detailed_feasibility_analysis == True:
            for lin_constr in bp_before_qubo.linear_constraints:
                constr_eval = round(lin_constr.evaluate(bit_list), 6) #left hand side of the constraint is evaluated
                if lin_constr.sense == ConstraintSense.EQ:
                    if constr_eval != lin_constr.rhs:
                        violated_constraints.append((lin_constr.name, constr_eval))
                elif lin_constr.sense == ConstraintSense.LE:
                    if constr_eval > lin_constr.rhs:
                        violated_constraints.append((lin_constr.name, constr_eval))
                elif lin_constr.sense == ConstraintSense.GE:
                    if constr_eval < lin_constr.rhs:
                        violated_constraints.append((lin_constr.name, constr_eval))
                else:
                    raise CustomizedError("invalid constraint type. it is not EQ, LEQ, nor GEQ")
                    
            for quad_constr in bp_before_qubo.quadratic_constraints:
                constr_eval = round(quad_constr.evaluate(bit_list), 6) #left hand side of the constraint is evaluated
                if quad_constr.sense == ConstraintSense.EQ:
                    if constr_eval != quad_constr.rhs:
                        violated_constraints.append((quad_constr.name, constr_eval))
                elif quad_constr.sense == ConstraintSense.LE:
                    if constr_eval > quad_constr.rhs:
                        violated_constraints.append((quad_constr.name, constr_eval))
                elif quad_constr.sense == ConstraintSense.GE:
                    if constr_eval < quad_constr.rhs:
                        violated_constraints.append((quad_constr.name, constr_eval))
                else:
                    raise CustomizedError("invalid constraint type. it is not EQ, LEQ, nor GEQ")
        else:
            feasible = bp_before_qubo.is_feasible(bit_list)
            if not feasible:
                violated_constraints.append('infeasible')
            
        result_dict[bit_str]['qubo_obj_value'] = float(qubo_obj_value)
        result_dict[bit_str]['mip_obj_value'] = float(bp_before_qubo_obj_value)
        result_dict[bit_str]['violated_constraints'] = violated_constraints
        counter +=1 
        
    config.logger.info("finished analysing feasibility of a solver_result-dictionary")    
    report.time_measurements['analyse_feasibility of_opt_results'] = time.time() - start_time
    return result_dict
       

def get_all_feasible_solutions(result_feasibility_analysed: dict, 
                               variable_names: list, 
                               report: Report) -> (dict, pd.DataFrame):
    '''
    function to get all the feasible solutions with their energy and their counts 
    that were sampled, 
    takes the variable names to translate the bitstring into an interpretable format. 
    outputs a dictionary and a dataframe that give the same information.
    '''
    start_time = time.time()
    feasible_solutions_dict = {} # initialize
    feasible_solutions_list = []  # initialize
    for bitstr, bitstr_dict in result_feasibility_analysed.items():
        if len(bitstr_dict['violated_constraints']) == 0: #no violated constraints
            feasible_solutions_dict[bitstr] = bitstr_dict
            if 'var_dict' not in feasible_solutions_dict[bitstr].keys():
                var_dict = {}
                bit_list = [float(item) if '.' in item else int(item) for item in bitstr.split(', ')]
                for i in range(len(bit_list)):
                    var_name = variable_names[i]
                    var_value = bit_list[i]
                    var_dict[var_name] = var_value
                feasible_solutions_dict[bitstr]['var_dict'] = var_dict
            
            solution = {'bitstr': bitstr,
                        'obj_value': bitstr_dict['mip_obj_value'],
                        'count': bitstr_dict['count']
                        }
            feasible_solutions_list.append(solution)
    
    feasible_solutions_df = pd.DataFrame(feasible_solutions_list)
    
    report.time_measurements['getting_all_feasible_solutions'] = time.time() - start_time
    return feasible_solutions_dict, feasible_solutions_df


def write_kpis_to_report(feas_sol_dict: dict, 
                         number_of_reads: int, 
                         config: Config,
                         report: Report):
    ''' 
    function to write some meaningful KPIs to the report-object
    '''
    report.solution_quality['number_of_shots'] = number_of_reads
    if feas_sol_dict != {}:
        report.solution_quality['feasible_solutions_how_many'] = sum(feas_sol['count'] for feas_sol in feas_sol_dict.values()) 
        if config.report_config["analyse_only_best_sols"] == False: #if only best sols were analysed there was no feasibility check done on all samples --> probability doesn't make sense
            report.solution_quality['feasible_solution_probability'] = report.solution_quality['feasible_solutions_how_many'] / number_of_reads
        report.solution_quality['feasible_solutions_how_many_different_ones'] = len(feas_sol_dict.keys())
        report.solution_quality['best_feasible_solution'] = min(feas_sol_dict, key=lambda k: feas_sol_dict[k]['mip_obj_value'])
        report.solution_quality['best_feasible_solution_obj_value'] = feas_sol_dict[report.solution_quality['best_feasible_solution']]['mip_obj_value']
        report.solution_quality['best_feasible_solution_frequency'] = int(feas_sol_dict[report.solution_quality['best_feasible_solution']]['count'])
        report.solution_quality['best_feasible_solution_probability'] =  report.solution_quality['best_feasible_solution_frequency'] / number_of_reads
    else:
        report.solution_quality['feasible_solutions_how_many'] = 0
        report.solution_quality['feasible_solution_probability'] = 0
        report.solution_quality['feasible_solutions_how_many_different_ones'] = 0
    
