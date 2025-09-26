# config validity
This markdown-file explains which values and types the attributes in the config-files may have

## 1. problem_options
contains all the possible problem applications and the attributes that are necessary for its creation

### BinPacking
| attribute_name | type | description |
| ------------- | ----- | --------------------------|
| object_weights | list | contains integers which describe the weights of the objects in the BinPacking problem |
| bin_capacity | int | capacity of the bins |
| incompatible_objects | list | contains tuples which describe which objects aren't allowed to be in the same bin |
| precedence_relations | list | some tuple to make sure that certain objects have to be put in an earlier bin than some others, e.g. (0,1) means that object 1 can't be put in an earlier bin than object 0 |
| random_instance | bool | if true the problem instance is created with random object weights |
| number_of_objects_for_random | int | number of objects for a random object weight creation |
| number_of_incompatibilities_for_random | int | number of incompatibilities for random incompatibility creation |
| number_of_precedences_for_random | int | number of precedences for random precedence creation |

### MarkowitzPortfolio
| attribute_name | type | description |
| ------------- | ----- | --------------------------|
| asset_returns | list | contains floats with the returns of all assets in the portfolio |
| asset_limits | list | contains floats with the limits on the shares of the assets in the portfolio |
| asset_covariance_matrix | list | list of lists of floats representing a quadratic covariance matrix of the assets |
| variable_discretization_n | int | for the quantum optimization, we need binary variables, so we discretize our portfolio-share-variables that lie between 0 and 1, we split this interval up in 2^n parts -->n = variable_discretization_n |
| opt_goal | str | the goal for the optimization might either be max_return or min_risk |
| max_volatility | float | the portfolio has to have a smaller volatility than this if opt_goal=='max_return_with_constrained_volatility' |
| min_return | float | the portfolio has to have a bigger return than this if opt_goal=='min_volatility_with_constrained_return' |
| choose_dax_assets | bool | if true, the specified DAX-assets of the DAX testdata-set in the attribute 'dax_assets' are taken as the assets for the Markowitz-Portfolio |
| dax_assets | list | the DAX-asset-strings that should be chosen if choose_dax_assets=True. Possible values for those DAX-assets can be looked in ...\src\problems\MarkowitzPortfolio\dax_annual_returns.csv |
| random_instance | bool | if True the problem instance is created randomly from a DAX testdata-set consisting of 36 assets. otherwise with preconfigured additional arguments |
| number_of_assets_for_random | int | the number of assets in the portfolio for random instance creation |
| if_mipsolver_then_solvediscretizedproblem | bool | if true, then if mipsolve is chosen as solve_method it's solving the discretized optimization-problem |

### TSP
| attribute_name | type | description |
| ------------- | ----- | --------------------------|
| nodes | list | contains the names of all nodes in a graph |
| edges | list | contains lists of length 2 that represent the edges, e.g. ['Berlin', 'Munich'] |
| distance_matrix | list | list of lists in quadratic matrix form that contains all the distances between the nodes |
| modelling_approach | str | the different implemented modelling approaches of the TSP, which are each fitting for a particular solve method, so far implemented are: 'vars_for_edge_usage', 'vars_for_vertex_at_time_step' |
| random_instance | bool | if true the tsp problem instance is created randomly |
| number_of_nodes_for_random | int | number of nodes for a random graph |
| number_of_edges_for_random | int | number of edges for a random graph |
| max_distance_for_random | int | maximum edge distance for a random graph |

### CVRP
| attribute_name | type | description |
| ------------- | ----- | --------------------------|
| nodes | list | contains the names of all nodes in a graph, here they are customers with demands |
| edges | list | contains lists of length 2 that represent the edges, e.g. ['Berlin', 'Munich'] |
| distance_matrix | list | list of lists in quadratic matrix form that contains all the distances between the nodes |
| demands | list | contains the demands of the nodes, have to be integers |
| vehicles_capacity | int | the maximum capacity of the vehicles that have |
| modelling_approach | str | the different implemented modelling approaches of the CVRP, which are each fitting for a particular solve method, so far implemented are: 'vars_for_routes' |
| random_instance |  bool | if true the cvrp problem instance is created randomly |
| number_of_nodes_for_random | int | number of nodes for a random graph |
| number_of_edges_for_random | int | number of edges for a random graph |
| max_distance_for_random | int | maximum edge distance for a random graph |
| max_demand_for_random | int | maximum customer demand for a random graph |

## 2. problem_mapping_options
| attribute_name | type | description |
| ------------- | ----- | --------------------------|
| choose_penalty_worst_feasible_better_than_best_infeasible | bool | if False: just use the given qubo_penalty_factor,  if True: the qubo penalty factors are chosen so that every feasible solution has a better obj value than aninfeasible solution |
| qubo_penalty_factor | int | penalty factor for the violation of a constraint in the QUBO objective |
| trim_inequalities | bool | # if True: the inequality coefficients are trimmed down, so that there is only a certain number of slack variables possible. this is done by dividing and rounding up/down the coefficients |
| max_number_of_slack_vars_per_inequality | int | if trim_inequalities=True, this is the maximum number of slack variables per inequality that is allowed |


## 3. solve_method_options
contains all the possible solver options and its specific configurations

### MIP_Solver
| attribute_name | type | description |
| ------------- | ----- | --------------------------|
| time_limit | int | the time-limit for the MIP-solver |
| abs_gap | int | the gap that is allowed between primal and dual solution to terminate the solver |
| presolve_method | int | 0: default, 1: aggressive, 2: fast, 3: off |
| initial_lp_algorithm | str | default is s; 's'implex, 'p'rimal simplex, 'd'ual simplex, 'b'arrier, barrier with 'c'rossover |

### QAOA
| attribute_name | type | description |
| ------------- | ----- | --------------------------|
| number_of_layers | int | number of alternating QAOA-circuit-layers of mixing- and problem-unitaries |
| shots_for_sampling | int | number of shots for sampling the QAOA-circuit to get good solutions |
| optimize_params_classically | bool | if true, the parameters of the QAOA-circuit will be optimized with a classical optimizer |
| opt_method | str | the classical optimization method for the QAOA-parameter training if optimize_params_classically=True -> more optimizer options on: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html |
| shots_for_training | int | number of shots for the classical training of the QAOA-circuit |
| training_with_local_simulator | bool | if False: train with same device as sampling; if True: use local simulator for parameter training --> useful if you don't want to use real quantum hardware resources for the training |
| optimize_params_with_grid_search | bool | if true, a grid search method for QAOA depth 1 is used to find the circuit parameters |
| grid_search_size | int | size of the grid of QAOA-circuits of one-dimensional beta and gamma for a grid search for the optimal parameters |
| sample_randomly | bool | if true, a QAOA circuit with parameters beta,gamma=0 will be executed which is basically random sampling |
| postprocess_with_steepest_descent | bool | if True: a steepest descent method (->one-bitflip) is executed at the end of the optimization method to potentially improve results |
| max_calculation_time_exists | bool | if True, the solver may to a certain time limit |
| max_calculation_time_in_min | int | if max_calculation_time_exists=True, this is the timelimit in minutes that the solver has to obey to |

### QuantumAnnealer
| attribute_name | type | description |
| ------------- | ----- | --------------------------|
| number_of_reads | int | how many times it should be annealed |
| annealing_time | int | Total running time per read in milliseconds |
| postprocess_with_steepest_descent | bool | if True: a steepest descent method (->one-bitflip) is executed at the end of the optimization method to potentially improve results |
| max_calculation_time_exists | bool | if True, the solver may to a certain time limit |
| max_calculation_time_in_min | int | if max_calculation_time_exists=True, this is the timelimit in minutes that the solver has to obey to |

### QuantumAnnealer_NeutralAtoms
| attribute_name | type | description |
| ------------- | ----- | --------------------------|
| number_of_reads | int | how many times it should be annealed |
| annealing_time | int | Total running time per read in milliseconds |
| unit_disk_MIS | bool | if True, the problem will be transformed to a unit-disk-maximum-independent-set-Problem (->UD-MIS) |

### SimulatedAnnealer
| attribute_name | type | description |
| ------------- | ----- | --------------------------|
| number_of_reads | int | how many times it should be annealed |
| annealing_time | int | Total running time per read in milliseconds |
| postprocess_with_steepest_descent | bool | if True: a steepest descent method (->one-bitflip) is executed at the end of the optimization method to potentially improve results |
| max_calculation_time_exists | bool | if True, the solver may to a certain time limit |
| max_calculation_time_in_min | int | if max_calculation_time_exists=True, this is the timelimit in minutes that the solver has to obey to |

### Tabu_Search
| attribute_name | type | description |
| ------------- | ----- | --------------------------|
| number_of_reads | int | how many times it should be annealed |
| time_limit | int | Total running time per read in milliseconds |
| postprocess_with_steepest_descent | bool | if True: a steepest descent method (->one-bitflip) is executed at the end of the optimization method to potentially improve results |
| max_calculation_time_exists | bool | if True, the solver may to a certain time limit |
| max_calculation_time_in_min | int | if max_calculation_time_exists=True, this is the timelimit in minutes that the solver has to obey to |

### Greedy_Algorithm
| attribute_name | type | description |
| ------------- | ----- | --------------------------|
| number_of_reads | int | how many times it should be annealed |
| time_limit | int | Total running time per read in milliseconds |
| max_calculation_time_exists | bool | if True, the solver may to a certain time limit |
| max_calculation_time_in_min | int | if max_calculation_time_exists=True, this is the timelimit in minutes that the solver has to obey to |


## 4. device_options
contains all the possible devices for each solve method
| solve_method_name | device_options |
| ------------- | ---------------------------------|
| MIP_Solver | 'SCIP' |
| QAOA | 'Local_Simulator', 'MQT_Simulator', 'Noisy_Local_Simulator', 'IBM_Quantum_Computer' |
| QuantumAnnealer | 'Quantum_Annealer_Dwave' |
| QuantumAnnealer_NeutralAtoms | 'unknown' |
| SimulatedAnnealer | 'Simulated_Annealer_Dwave' |
| Tabu_Search | 'Dwave_TabuSampler' |
| Greedy_Algorithm | 'Dwave_GreedyAlgorithm' |


## 5. report_options
| attribute_name | type | description |
| ------------- | ----- | --------------------------|
| visualize_distribution_of_obj_values | bool | if =True barplots will be saved for the sampling results that will show the distribution of objective values |
| analyse_only_best_sols | bool | if =True only a certain amount of samples will be analysed for feasibility and those will be the solutions with the best QUBO-objective values |
| number_of_sols_to_be_analysed | int | if analyse_only_best_sols=True, this will be the number of solutions that will be analysed for feasibility. This will be the solutions with the x best QUBO-objective values |
| save_feasible_sols_to_excel | bool | if True, a excel file with the feasible solutions will be saved to the export path |
| visualize_feasible_solutions | bool | if True, the solutions that are feasible will be visualized in a problem-specific way in the export path |
                  