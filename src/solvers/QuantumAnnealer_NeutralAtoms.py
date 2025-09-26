# -*- coding: utf-8 -*-
"""
Created on 07.03.2024

@author: Eric Stopfer
"""
import time
import pdb
import logging
import matplotlib.pyplot as plt

import networkx as nx
from networkx.algorithms import approximation as approx
from docplex.mp.model import Model
from qiskit_optimization import QuadraticProgram

from config.config import Config
from src.report import Report
from src.utils.error_utils import CustomizedError
from src.utils.opt_utils import solve_opt_model_with_SCIP
from src.utils.solver_result_utils import analyse_SCIP_result



class QuantumAnnealer_NeutralAtoms:
    '''
    ---- from Wikipedia -----
    
    The word ANNEALING describes the process of slowly cooling hot glass objects after they have been formed, 
    to relieve residual internal stresses introduced during manufacture. (https://en.wikipedia.org/wiki/Annealing_(glass) )
    
    QUANTUM ANNEALING (QA) is an optimization process for finding the global minimum of a given objective 
    function over a given set of candidate solutions (candidate states), by a process using quantum fluctuations.
    (https://en.wikipedia.org/wiki/Quantum_annealing )
    '''
    def __init__(self, config: Config, report: Report):
        """
        Function that creates a Quantum Annealing with Neutral Atoms solver instance
        """
        self.device_name = config.solve_method_device
        self.device = self.get_device(self.device_name, report)
        
        logging.info("we are using the following solving method: %s" %config.solve_method)
        logging.info("with the following device: %s" %config.solve_method_device)
        
        return
    
    
    def get_device(self, device_name: str, report: Report) -> any:
        '''
        Function that gets the Quantum Annealing with Neutral Atoms device by a name-string
        '''
        start_time = time.time()
        if device_name == "unknown":
            device = 0
            report.time_measurements['connect_to_unknown'] = time.time() - start_time
        else:
            raise CustomizedError(error_message="Wrong configuration of the device. Check in config.py or config.json")
        return device
        
        
    def run(self, mapped_problem: (QuadraticProgram, dict), config: Config, report: Report) -> dict:
        '''
        Function that runs a Quantum Annealing with Neutral Atoms process for a 
        problem that is mapped to a QUBO dictionary and solving configuration.
        '''
        # %% read the necessary attributes from the configuration
        try:
            num_reads = config.solve_method_config['number_of_reads']
            annealing_time = config.solve_method_config['annealing_time']
        except:
            raise CustomizedError(error_message="wrong or missing configuration of the Quantum Annealer. check in config.py and config.json")
        
        # %% do the anneal
        # start_time = time.time()
        # qubo_dict = mapped_problem[1]
        # logging.info("The Quantum anneal starts right now with %s reads and %sms annealing time" %(num_reads, annealing_time))
        # annealer_response = self.device.sample_qubo(
        #     Q = qubo_dict,
        #     num_reads = num_reads,
        #     annealing_time = annealing_time)
        # report.time_measurements['annealing'] = time.time() - start_time
                
        # # %% analyse the results and write to report
        # logging.info("The Quantum anneal with Neutral Atoms finished. We now analyse the results \n")
        # all_annealing_energies = annealer_response.data_vectors['energy']
        # #for the anneal, constants from the optimization problem were irrelevant, now we want to consider them
        # qubo_constant = mapped_problem[0].objective.constant
        # all_annealing_obj_values = all_annealing_energies + qubo_constant * np.ones(len(all_annealing_energies))
        # #now we want all the unique energy values and their counts
        # unique_obj_values, counts_obj_values = np.unique(all_annealing_obj_values, return_counts=True)
        # obj_values_counts_list = list(zip(unique_obj_values, counts_obj_values))
        # #now we want to get the best variable combination, so to speak the best sample and its objective value
        # best_sample, energy_best_sample, num_of_occurences_best_sample = annealer_response.first 
        # obj_value_best_sample = energy_best_sample + qubo_constant
        
        # report.solution_quality['best_objective_value'] = float(obj_values_counts_list[0][0])
        # report.solution_quality['best_objective_value_frequency'] = float(obj_values_counts_list[0][1])
        # report.solution_quality['best_objective_value_probability'] = float(obj_values_counts_list[0][1] / num_reads)
        
        # # %% create a barplot to see the distribution of the annealing results
        # self.visualize_barplot_obj_values(obj_values_counts_list, config.EXPORT_PATH)
        # # %% 
        # return best_sample



# %% now helperfunctions:
    @staticmethod
    def QUBO_to_WMIS_graph(qubo: dict, config: Config, report: Report) -> nx.Graph:
        """
        function that maps a minimization-QUBO to a Weighted-Maximum-Independent-Set-Problem.
        It takes a minimization-qubo as input and outputs a networkx-graph whose maximum
        independent set is equivalent to the minimum qubo-solution
        """
        start_time = time.time()
        logging.info("start to map QUBO to a maximum weighted independent set graph")
        delta = max(qubo.values()) + 1
        wmis_graph = nx.Graph()
        for (var1, var2), coeff in qubo.items():
            #check if both vars are the same --> it becomes a linear factor which gets added to the variable-node
            if var1 == var2:
                var1 = str(var1)
                #check if nodes already exist -> if not, add a node with certain weight
                if wmis_graph.has_node(var1) == False:
                    wmis_graph.add_node(var1, weight = delta-coeff)
                else:
                    wmis_graph.nodes[var1]["weight"] -= coeff #update weight, because node has been added with only delta before
            else: #if var1 != var2
                var1 = str(var1)
                var2 = str(var2)
                
                #add node for quadratic term with weight
                wmis_graph.add_node(var1+'_'+var2, weight = delta-coeff)
                
                #check if nodes for single variables already exist -> if not, add a node for the single variables without weight
                if wmis_graph.has_node(var1) == False:
                    wmis_graph.add_node(var1, weight = delta) # weight might be adjusted if there also is a linear term for this variable
                    wmis_graph.add_node('a_'+var1, weight = delta)
                    wmis_graph.add_edge(var1, 'a_'+var1)
                    wmis_graph.add_edge('a_'+var1, var1+'_'+var2)
                else: #if the node already exists
                    wmis_graph.add_edge('a_'+var1, var1+'_'+var2)
                wmis_graph.add_node('o_'+var1+var2+'_1', weight = delta)
                wmis_graph.add_edge(var1, 'o_'+var1+var2+'_1')
                wmis_graph.add_edge('o_'+var1+var2+'_1', var1+'_'+var2)
                        
                if wmis_graph.has_node(var2) == False:
                    wmis_graph.add_node(var2, weight = delta) # weight might be adjusted if there also is a linear term for this variable
                    wmis_graph.add_node('a_'+var2, weight = delta)
                    wmis_graph.add_edge(var2, 'a_'+var2)
                    wmis_graph.add_edge('a_'+var2, var1+'_'+var2)
                else: #if the node already exists
                    wmis_graph.add_edge('a_'+var1, var1+'_'+var2)
                wmis_graph.add_node('o_'+var1+var2+'_2', weight = delta)
                wmis_graph.add_edge(var2, 'o_'+var1+var2+'_2')
                wmis_graph.add_edge('o_'+var1+var2+'_2', var1+'_'+var2)
                wmis_graph.add_edge('o_'+var1+var2+'_1', 'o_'+var1+var2+'_2')
                
        report.time_measurements['mapping_to_WMIS_problem'] = time.time() - start_time
        logging.info("finished mapping of QUBO to WMIS-graph")    
        nx.draw(wmis_graph, with_labels=True, node_color='skyblue')
        plt.show()
        return wmis_graph
    
    
    @staticmethod
    def QUBO_to_UDG_WMIS_graph(qubo: dict, config: Config, report: Report) -> nx.Graph:
        """
        function that maps a minimization-QUBO to a Unit-Disk-Graph-Weighted-Maximum-Independent-Set-Problem.
        It takes a minimization-qubo as input and outputs a networkx-graph whose maximum
        independent set is equivalent to the minimum qubo-solution
        Function is based on the method developed by QuEra in the paper:
        https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.4.010316
        """
        start_time = time.time()
        logging.info("start to map QUBO to a unit-disk maximum weighted independent set graph \n")
        # %% analyse variable interactions of the qubo
        qubo_graph = QuantumAnnealer_NeutralAtoms.create_qubo_graph(qubo) 

        qubo_variable_interactions = {node: list(qubo_graph.neighbors(node)) for node in qubo_graph.nodes()}  
        variables_order = list(qubo_variable_interactions.keys())
                    
        # %% reorder variables to reduce unnecessary interactions in the lattices
        variables_order = QuantumAnnealer_NeutralAtoms.optimize_variable_order_for_udg_wmis_graph_with_SCIP(qubo_graph, config, report)
        
        # %% derive the variables interactions in the resulting udg-wmis-graph from the order
        udg_wmis_graph_var_interactions = QuantumAnnealer_NeutralAtoms.get_var_interactions_for_udg_wmis_graph(qubo_graph, variables_order)
        
        # %% create the udg-wmis-graph for the new variable order
        delta = max(qubo.values()) + 1
        udg_wmis_graph = QuantumAnnealer_NeutralAtoms.create_udg_wmis_graph(qubo_graph, udg_wmis_graph_var_interactions, delta)
        
        # %% reroute the variables
        
        # %% simplify the mapping
        
        # %% draw the resulting unit-disk-maximum-weighted-independent-set-graph
        pos = nx.spring_layout(udg_wmis_graph)#, iterations=1000)
        node_colors = [udg_wmis_graph.nodes[node]['weight'] for node in udg_wmis_graph.nodes()]
        nx.draw(udg_wmis_graph, pos=pos, with_labels=True, font_size=5, node_size=50, node_color=node_colors, cmap=plt.cm.Reds)
        #nx.draw(udg_wmis_graph, pos=pos, with_labels=False, font_size=5, node_size=50, node_color=node_colors, cmap=plt.cm.Reds)
        plt.show()
        
        # %%
        report.time_measurements['mapping_to_UDG_WMIS_problem'] = time.time() - start_time
        logging.info("finished mapping of QUBO to UDG-WMIS-graph")
        logging.info("with the optimization of the variable order we reduced the number of needed qubits from %s to %s"%(4*len(variables_order)**2, udg_wmis_graph.number_of_nodes()))
        logging.info("the UDG-WMIS-graph has %s nodes --> %s qubits needed \n"%(udg_wmis_graph.number_of_nodes(), udg_wmis_graph.number_of_nodes()))
        return udg_wmis_graph
    
    
    @staticmethod 
    def create_qubo_graph(qubo: dict) -> nx.Graph:
        '''
        function that creates a networkx-graph that represents a qubo. 
        This graph will have nodes for each variable and edges for each quadratic term. 
        -->node weights = linear coefficients, 
        -->edge weights = quadratic coefficients,
        '''
        logging.info("start to create a graph that represents the QUBO")
        qubo_graph = nx.Graph()
        for (var1, var2), coeff in qubo.items():
            if var1 != var2:
                if not qubo_graph.has_node(var1):
                    qubo_graph.add_node(var1, weight=0)
                if not qubo_graph.has_node(var2):
                    qubo_graph.add_node(var2, weight=0)
                qubo_graph.add_edge(var1, var2, weight=coeff)
            else: #var1 = var2
                if qubo_graph.has_node(var1):
                    qubo_graph.nodes[var1]['weight'] = coeff
                else:
                    qubo_graph.add_node(var1, weight=coeff)
        #plot the graph
        pos = nx.spring_layout(qubo_graph)
        nx.draw(qubo_graph, pos=pos, with_labels=True, font_size=5, node_size=50, node_color='skyblue')
        plt.show() 
        logging.info("finished creating a graph that represents the QUBO \n")
        return qubo_graph
    
    @staticmethod 
    def optimize_variable_order_for_udg_wmis_graph_with_SCIP(qubo_graph: nx.Graph, config: Config, report: Report) -> list:
        '''
        function to optimize the order of the variables for the QUBO-to-UDGWMIS transformation. 
        It creates a MIP and solves it with SCIP classical optimizer.
        '''
        logging.info("start to optimize the order of the variables for a sparse UDG-WMIS-graph \n")
        # %% create the problem
        #vars_ordering_prob = QuantumAnnealer_NeutralAtoms.create_docplex_model_var_reordering_problem_approximated(qubo_graph)
        vars_ordering_prob = QuantumAnnealer_NeutralAtoms.create_docplex_model_var_reordering_problem_exact(qubo_graph)
        
        # %% save the mapped problem to LP-file
        problem_name = "Variable_Reordering_Problem_MIP"
        vars_ordering_prob.export_as_lp(basename=problem_name, path=config.EXPORT_PATH)
        logging.info("The docplex-MIP of the variable reordering problem has been saved to the export-path to a LP-file")
        
        # %% solve the problem --> optimization with pyscipopt
        scip_opt_model = solve_opt_model_with_SCIP(
                problem_path = config.EXPORT_PATH,
                problem_name = problem_name, 
                time_limit = 60 * 15,  # 15 minutes
                allowed_gap = 0,
                presolve_method = 0,
                config = config,
                report = report
                )
        solution_dict = analyse_SCIP_result(scip_opt_model, problem_name, config, report)
        
        # %% analyse the result and derive the variable order
        variables_order = []
        for t in range(qubo_graph.number_of_nodes()):
            for key, value in solution_dict['var_dict'].items():
                if key.endswith("_"+str(t)) and key.startswith("z") and value == 1:
                    last_underscore_index = key.rfind("_")
                    new_str_key = key[2:last_underscore_index]
                    variables_order.append(new_str_key)
        # %%
        return variables_order
    
    @staticmethod
    def create_docplex_model_var_reordering_problem_approximated(qubo_graph: nx.Graph) -> Model:
        ''' 
        function that creates a docplex model for a formulation of the variable reordering
        problem that approximates the optimal variable order
        '''
        vars_ordering_prob = Model('Problem of ordering the QUBO-variables for the UDG-WMIS-graph')
        number_of_nodes = qubo_graph.number_of_nodes()
        names_of_nodes = list(qubo_graph.nodes())
        neighbors_dict = {node: list(qubo_graph.neighbors(node)) for node in qubo_graph.nodes()}   
        # %% add model variables
        node_at_time_vars = vars_ordering_prob.binary_var_matrix(keys1=names_of_nodes, keys2=range(number_of_nodes), name="z")#[f"x_{i}{j}" for i in range(number_of_nodes) for j in range(number_of_nodes)])        
        logging.info("added binary variables z_ij --> =1 if qubo-variable i is placed at spot j")
        
        # %% add model objective --> minimize sum of x_i variables
        objective = vars_ordering_prob.minimize(
            vars_ordering_prob.sum(
                vars_ordering_prob.sum(
                     (number_of_nodes - t) * node_at_time_vars[i, t] * 
                     vars_ordering_prob.sum(
                        vars_ordering_prob.sum(
                            2 * k * node_at_time_vars[j, k])
                        for j in neighbors_dict[i] 
                    for k in range(number_of_nodes)[(t+1):]) 
                for i in names_of_nodes)
            for t in range(number_of_nodes-1)))
        logging.info("added the objective with goal to minimize")
        
        # %% add model constraints --> exactly one variable at each spot
        assignment_constraints_node = vars_ordering_prob.add_constraints((vars_ordering_prob.sum(node_at_time_vars[i, t] for t in range(number_of_nodes)) == 1 for i in names_of_nodes), ["assignment_constraint_node_%d" % i for i in range(number_of_nodes)])
        logging.info("added constraints so that each qubo-variable is placed exactly at one spot")
        
        assignment_constraints_orderspot = vars_ordering_prob.add_constraints((vars_ordering_prob.sum(node_at_time_vars[i, t] for i in names_of_nodes) == 1 for t in range(number_of_nodes)), ["assignment_constraint_orderspot_%d" % i for i in range(number_of_nodes)])
        logging.info("added constraints so that each order spot gets filled with a qubo-variable")  
        # %%  
        return vars_ordering_prob
    
    @staticmethod
    def create_docplex_model_var_reordering_problem_exact(qubo_graph: nx.Graph) -> Model:
        ''' 
        function that creates a docplex model for a formulation of the variable reordering
        problem with which you certainly get the optimal variable order
        '''
        logging.info("start creation of the variable-order-optimization-model")
        number_of_nodes = qubo_graph.number_of_nodes()
        var_names = list(qubo_graph.nodes())
        # %% create a qubo_interaction_dictionary
        qubo_inc_dict = {}
        for var1 in var_names:
            for var2 in var_names:
                if qubo_graph.has_edge(var1, var2):
                    qubo_inc_dict[var1, var2] = 1
                else:
                    qubo_inc_dict[var1, var2] = 0
        
        # %% create the model
        vars_ordering_prob = Model('Problem of ordering the QUBO-variables for the UDG-WMIS-graph')
        
        # %% add model variables
        var_at_orderspot_vars = vars_ordering_prob.binary_var_matrix(keys1=var_names, keys2=range(number_of_nodes), name="z")
        logging.info("added binary variables z_ab --> =1 if qubo-variable a is placed at spot b")
        #node_block_needed_vars = vars_ordering_prob.binary_var_matrix(keys1=range(number_of_nodes), keys2=range(number_of_nodes), name="y")
        node_block_needed_vars = vars_ordering_prob.binary_var_list(keys=[(i, j) for i in range(number_of_nodes) for j in range(number_of_nodes) if i < j], name=[f"y_{i}_{j}" for i in range(number_of_nodes) for j in range(number_of_nodes) if i < j])
        logging.info("added binary variables y_ij --> =1 if  a block of nodes is used for the interaction of the variables in the order spots i,j")
        
        # %% add model objective --> minimize sum of x_i variables
        objective = vars_ordering_prob.minimize(
            vars_ordering_prob.sum(
                vars_ordering_prob.sum(
                    vars_ordering_prob.get_var_by_name(f"y_{i}_{j}")
                for i in range(number_of_nodes) if i < j)
            for j in range(number_of_nodes)))
        logging.info("added the objective with goal to minimize")
        
        # %% add model constraints 
        assignment_constraints_node = vars_ordering_prob.add_constraints(
            (vars_ordering_prob.sum(
                var_at_orderspot_vars[a, b] for b in range(number_of_nodes)) 
                == 1
            for a in var_names), 
            ["assignment_constraint_var_%s" % a for a in var_names])
        logging.info("added constraints so that each qubo-variable is placed exactly at one spot")
        
        assignment_constraints_orderspot = vars_ordering_prob.add_constraints(
            (vars_ordering_prob.sum(
                var_at_orderspot_vars[a, b] for a in var_names)
                == 1 
            for b in range(number_of_nodes)),
            ["assignment_constraint_orderspot_%d" % b for b in range(number_of_nodes)])
        logging.info("added constraints so that each order spot gets filled with a qubo-variable")  
        
        if_interact_then_block_constraints = vars_ordering_prob.add_quadratic_constraints(
            (vars_ordering_prob.get_var_by_name(f"y_{i}_{j}") >=
            vars_ordering_prob.sum(
                vars_ordering_prob.sum(
                    var_at_orderspot_vars[a,i] * var_at_orderspot_vars[b,j] * qubo_inc_dict[a,b] 
                for b in var_names)
            for a in var_names)
            for i in range(number_of_nodes) for j in range(number_of_nodes) if i < j))
        logging.info("added constraints so that a node block is needed if variables interact with each other")
        
        max_number_of_node_blocks = ((number_of_nodes**2) - number_of_nodes) / 2 + 1
        if_above_right_block_then_block_constraints = vars_ordering_prob.add_constraints(
            (max_number_of_node_blocks * vars_ordering_prob.get_var_by_name(f"y_{i}_{j}") >=
            vars_ordering_prob.sum(
                #vars_ordering_prob.sum(
                    vars_ordering_prob.get_var_by_name(f"y_{k}_{l}")
                for l in range(number_of_nodes) if l >= j for k in range(number_of_nodes) if k <= i)
            #for k in range(number_of_nodes) if k >= i)
            for i in range(number_of_nodes) for j in range(number_of_nodes) if i < j),
            ["if_above_right_block_then_block_constraints_%s_%s" %(i,j) for i in range(number_of_nodes) for j in range(number_of_nodes) if i < j])
        logging.info("added constraints so that a node block is needed if above or right of the spot there already exists a node block")
        # %%  
        logging.info("finished creation of the variable-order-optimization-model \n")
        return vars_ordering_prob
    
    
    @staticmethod 
    def get_var_interactions_for_udg_wmis_graph(qubo_graph: nx.Graph, variables_order: list) -> dict:
        '''
        function to get the variable interactions from a certain variables order
        of the udg-wmis-graph
        '''
        logging.info("start of get the variable interactions dict to be able to create the UDG-WMIS-graph")
        number_of_vars = len(variables_order)
        border_vertical = -1 #initial value
        border_horizontal = {} #initial value, will be filled with keys for each line down
        for i in variables_order:
            border_horizontal[i] = 100000000
            
        var_interactions = {}
        already_treated_vars = []
        for i in range(number_of_vars):
            var1 = variables_order[i]
            var_interactions[var1] = []
            for j in range(number_of_vars):
                var2 = variables_order[j]
                if var2 not in already_treated_vars and var1 != var2:
                    if i < border_horizontal[var2] and j > border_vertical:
                        if qubo_graph.has_edge(var1, var2):
                            var_interactions[var1].append(var2)
                            border_vertical = j
                            border_horizontal[var2] = i
                        else:
                            pass
                    else:
                        var_interactions[var1].append(var2)
            already_treated_vars.append(var1)
        
        for key, value in var_interactions.items():
            print( key, value)
        
        logging.info("got the variable interactions dictionary for the variable order \n")
        return var_interactions
    
    @staticmethod 
    def create_udg_wmis_graph(qubo_graph: nx.Graph, udg_wmis_graph_var_interactions: dict, delta: float) -> nx.Graph:
        '''
        function that creates the udg_wmis_graph for a qubo_graph with a certain
        variable order
        '''
        logging.info("start to create the unit disk WMIS-graph")
        variables_order = list(udg_wmis_graph_var_interactions.keys())
        var_copy_counters = {} #initialize the counter dict for each variable line
        
        udg_wmis_graph = nx.Graph() # initialize
        
        # add initial nodes for all variables with weight: (delta + h_i)
        for var in variables_order:
            h_i = -qubo_graph.nodes[var]['weight'] # qubo coefficient
            udg_wmis_graph.add_node(var+'_0', weight=delta+h_i)
            var_copy_counters[var] = 0
            
        # add crossing gadgets for each interaction
        for var1 in variables_order[:-1]:
            copy_gadget, var_copy_counters \
                = QuantumAnnealer_NeutralAtoms.add_copy_gadget(udg_wmis_graph, var1, var_copy_counters, delta)
            udg_wmis_graph.add_nodes_from(copy_gadget.nodes(data=True))
            udg_wmis_graph.add_edges_from(copy_gadget.edges())
            
            var_order_to_interact = udg_wmis_graph_var_interactions[var1]
            for var2 in var_order_to_interact:
                if qubo_graph.has_edge(var1, var2):
                    J_ij = -qubo_graph.edges[var1, var2]['weight'] # qubo coefficient
                else:
                    J_ij = 0
                crossing_gadget, var_copy_counters = \
                    QuantumAnnealer_NeutralAtoms.add_crossing_gadget(udg_wmis_graph, var1, var2, var_copy_counters, delta, J_ij)
                udg_wmis_graph.add_nodes_from(crossing_gadget.nodes(data=True))
                udg_wmis_graph.add_edges_from(crossing_gadget.edges())
            #final node of the variable line to make it even numbered
            udg_wmis_graph.add_node(var1+'_'+str(var_copy_counters[var1]+1), weight=delta)
            udg_wmis_graph.add_edge(var1+'_'+str(var_copy_counters[var1]), var1+'_'+str(var_copy_counters[var1]+1))
            var_copy_counters[var1] += 1
            
        #final node of the last variable's line to make it even numbered -->for-loop didn't cover last element
        last_var = variables_order[-1]
        udg_wmis_graph.add_node(last_var+'_'+str(var_copy_counters[last_var]+1), weight=delta)
        udg_wmis_graph.add_edge(last_var+'_'+str(var_copy_counters[last_var]), last_var+'_'+str(var_copy_counters[last_var]+1))
        var_copy_counters[last_var] += 1
        
        logging.info("finished the creation of the unit-disk WMIS-graph \n")
        return udg_wmis_graph
    
    @staticmethod
    def add_copy_gadget(graph: nx.Graph, var: str, var_copy_counters: dict, delta: float) -> (nx.Graph, dict):
        ''' 
        function to add a copy gadget for a variable to a graph.
        It returns the graph and the updated dictionary of the counters of the variable copies
        '''
        var_copy_counter = var_copy_counters[var]
     
        graph.add_node(var+'_'+str(var_copy_counter+1), weight=2*delta)
        graph.add_node(var+'_'+str(var_copy_counter+2), weight=2*delta)
        graph.add_edge(var+'_'+str(var_copy_counter), var+'_'+str(var_copy_counter+1))
        graph.add_edge(var+'_'+str(var_copy_counter+1), var+'_'+str(var_copy_counter+2))
        
        var_copy_counters[var] +=2
        return graph, var_copy_counters
    
    @staticmethod
    def add_crossing_gadget(graph: nx.Graph, var1: str, var2: str, var_copy_counters: dict, delta: float, weight_interaction: float) -> (nx.Graph, dict):
        ''' 
        function to add a crossing gadget for two variables to a graph.
        It returns the graph and the updated dictionary of the counters of the variable copies
        '''
        var1_copy_counter = var_copy_counters[var1]
        var2_copy_counter = var_copy_counters[var2]
        crossing_name = 'cross_' + var1 + '_' + var2 + '_'
        
        # %% add all additional nodes
        #new variable-copy-entry-nodes
        graph.add_node(var1+'_'+str(var1_copy_counter+1), weight=2*delta)
        graph.add_node(var2+'_'+str(var2_copy_counter+1), weight=2*delta)
        #crossing nodes
        graph.add_node(crossing_name+str(1), weight=4*delta+weight_interaction) # this is the node that gets activated when both the entry variables are =1 --> qubo term gets activated
        graph.add_node(crossing_name+str(2), weight=4*delta)
        graph.add_node(crossing_name+str(3), weight=4*delta)
        graph.add_node(crossing_name+str(4), weight=4*delta)
        #new variable-copy-exit-nodes
        graph.add_node(var1+'_'+str(var1_copy_counter+2), weight=2*delta)
        graph.add_node(var2+'_'+str(var2_copy_counter+2), weight=2*delta)
        var_copy_counters[var1] += 2
        var_copy_counters[var2] += 2
        
        # %% add all edges
        #edges to entry-variable-copy-nodes
        graph.add_edge(var1+'_'+str(var1_copy_counter), var1+'_'+str(var1_copy_counter+1))
        graph.add_edge(var2+'_'+str(var2_copy_counter), var2+'_'+str(var2_copy_counter+1))
        #edges from entry-variable-copy-nodes to the crossing
        graph.add_edge(var1+'_'+str(var1_copy_counter+1), crossing_name+str(1))
        graph.add_edge(var1+'_'+str(var1_copy_counter+1), crossing_name+str(2))
        graph.add_edge(var2+'_'+str(var2_copy_counter+1), crossing_name+str(1))
        graph.add_edge(var2+'_'+str(var2_copy_counter+1), crossing_name+str(3))
        #edges inside the crossing
        graph.add_edge(crossing_name+str(1), crossing_name+str(2))
        graph.add_edge(crossing_name+str(1), crossing_name+str(3))
        graph.add_edge(crossing_name+str(1), crossing_name+str(4))
        graph.add_edge(crossing_name+str(2), crossing_name+str(3))
        graph.add_edge(crossing_name+str(2), crossing_name+str(4))
        graph.add_edge(crossing_name+str(3), crossing_name+str(4))
        #edges outgoing to the exit-variable-copy-nodes
        graph.add_edge(crossing_name+str(3), var1+'_'+str(var1_copy_counter+2))
        graph.add_edge(crossing_name+str(4), var1+'_'+str(var1_copy_counter+2))
        graph.add_edge(crossing_name+str(2), var2+'_'+str(var2_copy_counter+2))
        graph.add_edge(crossing_name+str(4), var2+'_'+str(var2_copy_counter+2))
        # %%
        return graph, var_copy_counters
    
    @staticmethod
    def WMIS_graph_to_MIP(graph: nx.Graph) -> Model:
        '''
        function that maps a graph to a MIP that solves the maximum weighted independent set
        problem on the graph
        '''
        logging.info("start the creation of the WMIS-MIP for the graph \n")
        
        mis_mip = Model("Weighted Maximum Independent Set")
        
        number_of_nodes = graph.number_of_nodes()
        names_of_nodes = list(graph.nodes())
        weights_of_nodes = [graph.nodes[node]["weight"] for node in graph.nodes()]
        edges = list(graph.edges())
        # %% add model variables
        node_variables = mis_mip.binary_var_list(keys=range(number_of_nodes), name=[f"x_{i}" for i in names_of_nodes])        
        logging.info("added binary variables x_i --> =1 if node i is in the WMIS, =0 if not")
       
        # %% add model objective --> maximize sum of weighted x_i variables --> minimize minus-sum
        objective = mis_mip.minimize(- mis_mip.sum([weights_of_nodes[i] * node_variables[i] for i in range(number_of_nodes)]))
        logging.info("added the objective with goal to maximize the weight of the chosen nodes set")
        
        # %% add model constraints
        independency_constr = mis_mip.add_constraints((mis_mip.get_var_by_name("x_"+node_i) + mis_mip.get_var_by_name("x_"+node_j) <= 1 for (node_i, node_j) in edges), ["independency_constraint_edge_%s_%s" % (i, j) for (i,j) in edges])
        logging.info("added constraints so that two incident nodes can't be chosen together for the WMIS \n")
        
        logging.info("finished the creation of the WMIS-MIP \n") 
        return mis_mip
    
    
    @staticmethod
    def postprocess_mis_result(mis_solution: dict) -> dict:
        '''
        function that maps a result from the NeutralAtom-QuantumAnnealer backwards
        to an interpretable form --> because in the transformation to the MIS we 
        added many ancillary nodes whose results are not interesting
        '''
        filtered_solution = {key[2:-2]: mis_solution[key] for key in mis_solution if key.endswith("_0")}
        return filtered_solution
    


# class Config_Test:
#     def __init__(self):
#         self.EXPORT_PATH = "C:\\Users\\stopfer\\Documents\\git\\quopt\\"

# class Report_Test:
#     def __init__(self):
#         self.solution_quality = {}
#         self.time_measurements = {}

# test_qubo = {
#             ("1", "1"): -1,
#             ("2", "2"): -2,
#             ("3", "3"): -3,
#             ("1", "2"): -4,
#             ("2", "3"): 4,
#             }

# test_qubo = {
#             ("1", "1"): -1,
#             ("2", "2"): -2,
#             ("3", "3"): -3,
#             ("4", "4"): -4,
#             #("1", "2"): -4,
#             ("2", "3"): 4,
#             ("1", "4"): 4,
#             ("3", "4"): 4,
#             ("3", "5"): 2,
#             ("4", "5"): 3
#             }

# config = Config_Test()
# report = Report_Test()
# #graph = QuantumAnnealer_NeutralAtoms.QUBO_to_WMIS_graph(test_qubo, config, report)
# graph = QuantumAnnealer_NeutralAtoms.QUBO_to_UDG_WMIS_graph(test_qubo, config, report)

# mip = QuantumAnnealer_NeutralAtoms.WMIS_graph_to_MIP(graph)

# mip.export_as_lp(basename="WMIS-MIP", path=config.EXPORT_PATH)

# scip_opt_model = solve_opt_model_with_SCIP(
#         problem_path = config.EXPORT_PATH,
#         problem_name = "WMIS-MIP", 
#         time_limit = 60 * 15,  # 15 minutes
#         allowed_gap = 0,
#         presolve_method = 0,
#         config = config,
#         report = report
#         )
# sol_dict = analyse_SCIP_result(scip_opt_model, "WMIS-MIP", report)

# max_ind_set = [key[2:] for key, value in sol_dict['var_dict'].items() if value == 1]
# optimal_solution = QuantumAnnealer_NeutralAtoms.postprocess_mis_result(sol_dict)

# node_colors = ["tab:red" if node in max_ind_set else "tab:blue" for node in graph]
# pos = nx.spring_layout(graph, seed=39299899)
# nx.draw(graph, pos=pos, with_labels=True, font_size=5, node_color=node_colors, node_size=50)
# plt.show()

        