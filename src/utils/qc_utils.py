# -*- coding: utf-8 -*-
"""
Created on 24.05.2024

@author: Eric Stopfer
"""
import pdb
from math import floor, ceil
import json
import numpy as np

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import from_gurobipy
from qiskit_optimization.converters import InequalityToEquality, IntegerToBinary, LinearEqualityToPenalty
from qiskit_optimization.problems.constraint import Constraint, ConstraintSense
from qiskit_optimization.problems.quadratic_objective import QuadraticObjective
from qiskit_ibm_runtime import QiskitRuntimeService, Session

from src.utils.error_utils import CustomizedError



def transform_gurobipy_model_to_qubo(opt_model,
                                    penalty_factor: float,
                                    trim_inequ=False, 
                                    max_slack_vars_per_inequ=7
                                    ) -> (dict, QuadraticProgram, QuadraticProgram):
    '''
    Function that maps an opt model to QUBO formulation
    (->Quadratic-Unconstrained-Binary-Optimization).
    Returns a Qiskit-QuadraticProgram and a dictionary with the qubo-coefficients.
    '''
    # %% transform gurobipy model to the qiskit-optimization framework
    mip_qiskit = from_gurobipy(opt_model.model)
    if trim_inequ == True:
        mip_qiskit = trim_inequalities(mip_qiskit, max_slack_vars_per_inequ)
    
    # %% transform inequalities to equalities --> with slacks
    mip_ineq2eq = InequalityToEquality().convert(mip_qiskit)
    
    # %% transform integer variables to binary variables -->split up into multiple binaries
    mip_int2bin = IntegerToBinary().convert(mip_ineq2eq)
    
    # %% transform small quadratic equalities to penalty terms in the objective
    quadrconstr2pen = SmallQuadraticEqualitytoPenalty(bin_quad_prog=mip_int2bin, penalty=penalty_factor)
    
    # %% transform linear equalities to penalty terms in the objective
    qubo = LinearEqualityToPenalty(penalty=penalty_factor).convert(quadrconstr2pen)
    
    # %% squash the quadratic and linear QUBO-coefficients together into a dictionary
    quadr_coeff = qubo.objective.quadratic.to_dict(use_name=True)
    lin_coeff = qubo.objective.linear.to_dict(use_name=True)                
    for var, var_value in lin_coeff.items():
        if (var, var) in quadr_coeff.keys():
            quadr_coeff[(var,var)] += var_value
        else:
            quadr_coeff[(var,var)] = var_value
    qubo_dict = quadr_coeff 
    qubo_dict = rescale_coefficients(qubo_dict)
    # %% 
    return qubo_dict, qubo, mip_int2bin


def SmallQuadraticEqualitytoPenalty(bin_quad_prog: QuadraticProgram, penalty: float) -> QuadraticProgram:
    '''
    function to transform a binary quadratic program to a QUBO. 
    Additionally to the Qiskit_optimization class LinearEqualityToPenalty, 
    terms like var1*var2=rhs can be put into the objective via penalties. 
    -->reason: 
    (binary_var1*binary_var2 - rhs)^2 = binary_var1*binary_var2 - 2*rhs*binary_var1*binary_var2 + rhs^2
    part of the code is copied from documentation: 
    https://qiskit-community.github.io/qiskit-optimization/_modules/qiskit_optimization/converters/linear_equality_to_penalty.html#LinearEqualityToPenalty
    '''
    # create empty QuadraticProgram model
    quadrconstr2pen = QuadraticProgram(name=bin_quad_prog.name)

    # add original variables
    for x in bin_quad_prog.variables:
        quadrconstr2pen.binary_var(name=x.name)
    
    # add original linear constraints
    for lin_constr in bin_quad_prog.linear_constraints:
        lin_constr_dict = lin_constr.linear.to_dict()
        sense = lin_constr.sense
        rhs = lin_constr.rhs
        name = lin_constr.name
        quadrconstr2pen.linear_constraint(lin_constr_dict, sense, rhs, name)    
    
    # get original objective terms
    offset = bin_quad_prog.objective.constant
    linear = bin_quad_prog.objective.linear.to_dict()
    quadratic = bin_quad_prog.objective.quadratic.to_dict()
    sense = bin_quad_prog.objective.sense.value    
    
    # convert small quadratic constraints of type {var1*var2 + var1 + var2 - constant = 0} into penalty terms
    for constraint in bin_quad_prog.quadratic_constraints:
        
        if constraint.sense != Constraint.Sense.EQ:
            raise CustomizedError("An inequality constraint exists. The method supports only equality constraints.")

        constant = constraint.rhs
        linear_part_constr = constraint.linear.to_dict()
        quadratic_part_constr = constraint.quadratic.to_dict()
        
        linear_part_terms = list(linear_part_constr.keys())
        quadratic_part_terms = list(quadratic_part_constr.keys())
        
        all_terms = []
        for quadr_term in quadratic_part_terms:
            all_terms.append(quadr_term)
        for lin_term in linear_part_terms:
            all_terms.append([lin_term])
        all_terms.append([]) # for the constant part
        
        # add the penalties
        for term1 in all_terms:
            if len(term1) == 0:
                coeff1 = constant
            elif len(term1) == 1:
                coeff1 = linear_part_constr[term1[0]]
            else:
                coeff1 = quadratic_part_constr[(term1[0], term1[1])]
                
            for term2 in all_terms:
                if len(term2) == 0:
                    coeff2 = constant
                elif len(term2) == 1:
                    coeff2 = linear_part_constr[term2[0]]
                else:
                    coeff2 = quadratic_part_constr[(term2[0], term2[1])]
                    
                union_of_terms = list(set(term1).union(term2))
                
                if len(union_of_terms) == 0:
                    offset -= sense * penalty * coeff1 * coeff2
                elif len(union_of_terms) == 1:
                    linear_var = union_of_terms[0]
                    linear[linear_var] = linear.get(linear_var, 0.0) + sense * penalty * coeff1 * coeff2
                elif len(union_of_terms) == 2:
                    quadr_var = (union_of_terms[0], union_of_terms[1])
                    quadratic[quadr_var] = quadratic.get(quadr_var, 0.0) + sense * penalty * coeff1 * coeff2
                else: 
                    raise CustomizedError("The quadratic constraint contains linear terms whose variables don't match the quadratic term")
            
    if bin_quad_prog.objective.sense == QuadraticObjective.Sense.MINIMIZE:
        quadrconstr2pen.minimize(offset, linear, quadratic)
    else:
        quadrconstr2pen.maximize(offset, linear, quadratic)

    return quadrconstr2pen
    
    
def trim_inequalities(opt_model, max_slack_vars_per_inequ=7):
    '''
    function to reduce the number of slack variables by trimming the coefficients of inequalities. 
    
    If there is an inequality in the opt_model that has high coefficients, the coefficients
    get adjusted so that only a certain number of slack variables is needed. 
    This is done by dividing an inequality through a certain number and rounding the coefficients. 
    Feasibile solutions of the new inequality are then still feasible for the old inequality.
    '''
    max_rhs = 2**(max_slack_vars_per_inequ) - 1 # the maximum value of the right hand side that we want to achieve by the trimming
    new_lin_constraints = [] #initialize list for new linear constraints
    lin_constraints_to_delete = [] #initialize lists for linear constraints that will be deleted
    new_quadr_constraints = [] #initialize list for new quadratic constraints
    quadr_constraints_to_delete = [] #initialize lists for quadratic constraints that will be deleted
    
    # %% for linear constraints
    for lin_constr in opt_model.linear_constraints:
        if lin_constr.sense == ConstraintSense.EQ: #we only check for inequalities
            pass
        else:
            largest_negative_coefficient = lin_constr.rhs # inital value, set to right-hand-side
            for var, coeff in lin_constr.linear.to_dict().items():
                if coeff <= 0:
                    largest_negative_coefficient += coeff
                    
            if -largest_negative_coefficient >= max_rhs:
                scaling_factor_for_coeffs = max_rhs / (-largest_negative_coefficient)
                lin_constr_dict = {}
                for var, coeff in lin_constr.linear.to_dict().items():
                    if coeff < 0: #if rhs-term, take floor to make the InEQ more strict
                        lin_constr_dict[var] = - floor(-coeff * scaling_factor_for_coeffs)
                    else: #if lhs-term, take ceiling to make the InEQ more strict
                        lin_constr_dict[var] = ceil(coeff * scaling_factor_for_coeffs)
                #identifiers for new constraint
                name_new_constraint = lin_constr.name + '_coeffs_rounded'
                new_lin_constraints.append((lin_constr_dict, name_new_constraint))
                #identifiers for linear constraints to delete
                lin_constraints_to_delete.append(lin_constr.name)
    
    # %% now for quadratic constraints
    for quadr_constr in opt_model.quadratic_constraints:
        if quadr_constr.sense == ConstraintSense.EQ: #we only check for inequalities
            pass
        else:
            largest_negative_coefficient = quadr_constr.rhs # inital value, set to right-hand-side
            for var, coeff in quadr_constr.linear.to_dict().items():
                if coeff < largest_negative_coefficient:
                    largest_negative_coefficient = coeff
                    
            if -largest_negative_coefficient >= max_rhs:
                scaling_factor_for_coeffs = max_rhs / (-largest_negative_coefficient)
                quadr_constr_dict_lin = {}
                quadr_constr_dict_quadr = {}
                for var, coeff in quadr_constr.linear.to_dict().items():
                    if coeff < 0: #if rhs-term, take floor to make the InEQ more strict
                        quadr_constr_dict_lin[var] = - floor(-coeff * scaling_factor_for_coeffs)
                    else: #if lhs-term, take ceiling to make the InEQ more strict
                        quadr_constr_dict_lin[var] = ceil(coeff * scaling_factor_for_coeffs)
                for var, coeff in quadr_constr.quadratic.to_dict().items():
                    if coeff < 0: #if rhs-term, take floor to make the InEQ more strict
                        quadr_constr_dict_quadr[var] = - floor(-coeff * scaling_factor_for_coeffs)
                    else: #if lhs-term, take ceiling to make the InEQ more strict
                        quadr_constr_dict_quadr[var] = ceil(coeff * scaling_factor_for_coeffs)
                #identifiers for new constraint
                name_new_constraint = quadr_constr.name + '_coeffs_rounded'
                new_quadr_constraints.append((quadr_constr_dict_lin, quadr_constr_dict_quadr, name_new_constraint))
                #identifiers for quadratic constraints to delete
                quadr_constraints_to_delete.append(quadr_constr.name)
                
    # %% add the new constraints to the model 
    for (lin_constr_dict, constr_name) in new_lin_constraints:
        opt_model.linear_constraint(
            linear=lin_constr_dict, 
            rhs=0, 
            name=constr_name)
    for (quadr_constr_dict_lin, quadr_constr_dict_quad, constr_name) in new_quadr_constraints:
        opt_model.quadratic_constraint(
            linear=quadr_constr_dict_lin,
            quadratic=quadr_constr_dict_quad,
            rhs=0,
            name=constr_name)
        
    # %% delete the already replaced constraints
    for lin_constr_name in lin_constraints_to_delete:
        opt_model.remove_linear_constraint(lin_constr_name)
    for quadr_constr_name in quadr_constraints_to_delete:
        opt_model.remove_quadratic_constraint(quadr_constr_name)
    # %%
    return opt_model


def rescale_coefficients(qubo_dict: dict, best_range = [2, 10]):
    max_factor = max([abs(x) for x in qubo_dict.values()])
    # min_factor = min(qubo_dict.values())
    rescaling_factor = abs(best_range[1] / max_factor)
    qubo_dict = {key: value*rescaling_factor for key, value in qubo_dict.items()}
    return qubo_dict
         
    
def transform_qubo_to_ising(qubo: dict):
    '''
    function to construct a matrix J, a vector h and an offset c for the Ising formulation
    that is equivalent to the the QUBO: obj = x J x^T + h x + c.
    The Qubo-variables x are in {0; 1}. The problem gets transformed so that
    variables are y in {-1; 1}: x = 1/2 * (1 - y).
    Here 0 gets mapped to a 1    and   a 1 gets mapped to a -1
    '''
    num_qubits = len(qubo.variables)
    ising_matrix = np.zeros((num_qubits, num_qubits), dtype=np.float64)
    ising_vector = np.zeros(num_qubits, dtype=np.float64)
    
    ising_offset = qubo.objective.constant
    
    for idx, coeff in qubo.objective.linear.to_dict().items():
        ising_vector[idx] -= 1/2 * coeff
        ising_offset += 1/2 * coeff
    
    for (i, j), coeff in qubo.objective.quadratic.to_dict().items():
        if i == j:
            ising_offset += 1/2 * coeff #because the quadratic term x_i * x_j reduces to 1 if the x are ising variables in {-1, 1} --> another constant term
        else:
            ising_matrix[i,j] += 1/4 * coeff   
            ising_offset += 1/4 * coeff
        ising_vector[i] -= 1/4 * coeff
        ising_vector[j] -= 1/4 * coeff
    
    return ising_matrix, ising_vector, ising_offset


def calculate_expectation_value_random_sampling_qubo(qubo: dict, qubo_constant: float):
    '''
    function to calculate the expectation value of random sampling the binary variables
    If we assign the variables randomly to 0 and 1, a linear term gets added with probability 1/2 
    and a quadratic term gets added with probability (1/2)^2 = 1/4
    '''
    expectation_value = qubo_constant # initial value
    for (var1, var2), coeff in qubo.items():
        if var1 == var2:
            expectation_value += coeff * 0.5
        else:
            expectation_value += coeff * 0.25
    return expectation_value


def calc_ising_energy_of_bitstring_array(ising_matrix: np.array, ising_vector: np.array, ising_offset: float, bitstring_array: np.array) -> any:
    """
    function that calculates the ising energy of bitstring measurements for a 
    certain ising formulation. 
    The function is compatible for a single bitstring result or multiple bitstring results.
    """
    energy_from_vector = np.dot(bitstring_array, ising_vector)
    if len(bitstring_array.shape) == 1: #if it is only one measurement -->if dim(bitstring_array) = n x 1
        energy_from_matrix = np.dot(bitstring_array, np.dot(ising_matrix, np.transpose(bitstring_array)))
        energy_from_offset = ising_offset
    else: # if it is multiple measurements -->if dim(bitstring_array) = n x m
        energy_from_matrix = np.diag(np.dot(bitstring_array, np.dot(ising_matrix, np.transpose(bitstring_array))))
        energy_from_offset = ising_offset * np.ones(bitstring_array.shape[0])
        
    return energy_from_matrix + energy_from_vector + energy_from_offset
    

def create_ibm_session(solver_api_tokens_path):
    #get the ibm token
    with open(solver_api_tokens_path, "r") as json_file:
        solver_api_tokens_data = json.load(json_file)
    ibm_token = solver_api_tokens_data['IBM_Token']
    if ibm_token == "type_your_ibm_api_token_here": # default value
        raise CustomizedError("for IBM Quantum Computer a valid API token is necessary. it has to be put inside the file: \n C:\...\quopt\config\config_files\solver_api_tokens.json")
    #connect to hardware
    service = QiskitRuntimeService(channel='ibm_quantum', token=ibm_token)
    backend = service.least_busy(operational=True, simulator=False)
    ibm_session = Session(backend=backend) #session is started when the first job is started
    return ibm_session