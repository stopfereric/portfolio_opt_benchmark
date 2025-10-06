# config validity
This markdown-file explains which values and types the attributes in the config-files may have

## 1. problem_options
contains all the possible problem applications and the attributes that are necessary for its creation

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
| delta_risk_aversion | float | the factor with which portfolio return and volatility are balanced if opt_goal=='max_return_min_volatility' |
| choose_dax_assets | bool | if true, the specified DAX-assets of the DAX testdata-set in the attribute 'dax_assets' are taken as the assets for the Markowitz-Portfolio |
| dax_assets | list | the DAX-asset-strings that should be chosen if choose_dax_assets=True. Possible values for those DAX-assets can be looked in ...\src\problems\MarkowitzPortfolio\dax_annual_returns.csv |
| choose_nasdaq_assets | bool | if true, the specified nasdaq-assets of the nasdaq testdata-set in the attribute 'nasdaq_assets' are taken as the assets for the Markowitz-Portfolio |
| nasdaq_assets | list | the nasdaq-asset-strings that should be chosen if choose_nasdaq_assets=True. Possible values for those nasdaq-assets can be looked in ...\src\problems\MarkowitzPortfolio\nasdaq_annual_returns.csv |
| random_instance | bool | if True the problem instance is created randomly from a DAX/Nasdaq/NYSE testdata-set |
| number_of_assets_for_random | int | the number of assets in the portfolio for random instance creation |
| stock_market_for_random_instance | str | the allowed stock markets that the random assets are taken from, right now allowed: ["dax", "nasdaq", "nyse", ""] |
| if_mipsolver_then_solvediscretizedproblem | bool | if true, then if mipsolve is chosen as solve_method it's solving the discretized optimization-problem |

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
| time_limit | int | the time-limit in seconds for the MIP-solver |
| abs_gap | float | the absolute gap that is allowed between primal and dual solution to terminate the solver |
| rel_gap | float | the relative gap that is allowed between primal and dual solution to terminate the solver |
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
| sample_with_linearramp_without_training | bool | if true, a QAOA circuit with parameters of the linear ramp formula is sampled |
| sample_randomly | bool | if true, a QAOA circuit with parameters beta,gamma=0 will be executed which is basically random sampling |
| postprocess_with_steepest_descent | bool | if True: a steepest descent method (->one-bitflip) is executed at the end of the optimization method to potentially improve results |
| max_calculation_time_exists | bool | if True, the solver may to a certain time limit |
| max_training_time_in_sec | int | if max_calculation_time_exists=True, this is the timelimit in seconds that the solver training has to obey to |
| max_sampling_time_in_sec | int | if max_calculation_time_exists=True, this is the timelimit in seconds that the solver sampling has to obey to |

### QuantumAnnealer
| attribute_name | type | description |
| ------------- | ----- | --------------------------|
| number_of_reads | int | how many times it should be annealed |
| annealing_time | int | Total running time per read in milliseconds |
| use_custom_chain_strength | bool | if true, a customized chain strength value is taken, otherwise the default dwave implementation |
| chain_strength | float | if use_custom_chain_strength == True, this is the used chain strength |
| postprocess_with_steepest_descent | bool | if True: a steepest descent method (->one-bitflip) is executed at the end of the optimization method to potentially improve results |
| max_calculation_time_exists | bool | if True, the solver must obey to a certain time limit |
| max_calculation_time_in_min | float | if max_calculation_time_exists=True, this is the timelimit in minutes that the solver has to obey to |

### SimulatedAnnealer
| attribute_name | type | description |
| ------------- | ----- | --------------------------|
| number_of_reads | int | how many times it should be annealed |
| calc_initial_states_with_heuristic | bool | if True, the simulated annealing is initialized by states that are created by a heuristic |
| number_of_initial_states | int | if calc_initial_states_with_heuristic==True, this is the number of initial states that are created by the heuristic |
| postprocess_with_steepest_descent | bool | if True: a steepest descent method (->one-bitflip) is executed at the end of the optimization method to potentially improve results |
| max_calculation_time_exists | bool | if True, the solver must obey to a certain time limit |
| max_calculation_time_in_min | float | if max_calculation_time_exists=True, this is the timelimit in minutes that the solver has to obey to |

### Tabu_Search
| attribute_name | type | description |
| ------------- | ----- | --------------------------|
| number_of_reads | int | how many times tabu search should be run |
| time_limit | int | Total running time per read in milliseconds |
| calc_initial_states_with_heuristic | bool | if True, the simulated annealing is initialized by states that are created by a heuristic |
| number_of_initial_states | int | if calc_initial_states_with_heuristic==True, this is the number of initial states that are created by the heuristic |
| postprocess_with_steepest_descent | bool | if True: a steepest descent method (->one-bitflip) is executed at the end of the optimization method to potentially improve results |
| max_calculation_time_exists | bool | if True, the solver must obey to a certain time limit |
| max_calculation_time_in_min | float | if max_calculation_time_exists=True, this is the timelimit in minutes that the solver has to obey to |

### Greedy_Algorithm
| attribute_name | type | description |
| ------------- | ----- | --------------------------|
| number_of_reads | int | how many times the greedy algorithm should be run |
| max_calculation_time_exists | bool | if True, the solver may to a certain time limit |
| max_calculation_time_in_min | float | if max_calculation_time_exists=True, this is the timelimit in minutes that the solver has to obey to |

### Opt_Heuristics
| attribute_name | type | description |
| ------------- | ----- | --------------------------|
| number_of_samples | int | the number of samples that should be generated by the Opt-Heuristic |
| max_calculation_time_exists | bool | if True, the solver must obey to a certain time limit |
| max_calculation_time_in_min | float | if max_calculation_time_exists=True, this is the timelimit in minutes that the solver has to obey to |
| heuristic_name | str | the name of the heuristic that should be used, currently implemented: ["default", "steepestdescent_weightadding", "randomized_weightadding"] |

### RandomSamplerQUBO
| attribute_name | type | description |
| ------------- | ----- | --------------------------|
| number_of_reads | int | the number of samples that should be generated by the RandomSampler |
| max_calculation_time_exists | bool | if True, the solver must obey to a certain time limit |
| max_calculation_time_in_min | float | if max_calculation_time_exists=True, this is the timelimit in minutes that the solver has to obey to |

## 4. device_options
contains all the possible devices for each solve method
| solve_method_name | device_options |
| ------------- | ---------------------------------|
| MIP_Solver | 'SCIP', 'Gurobi_LocalLicense', 'Gurobi_ComputeServer' |
| QAOA | 'Local_Simulator', 'MQT_Simulator', 'Noisy_Local_Simulator', 'IBM_Quantum_Computer' |
| QuantumAnnealer | 'Quantum_Annealer_Dwave' |
| QuantumAnnealer_NeutralAtoms | 'unknown' |
| SimulatedAnnealer | 'Simulated_Annealer_Dwave' |
| Tabu_Search | 'Dwave_TabuSampler' |
| Greedy_Algorithm | 'Dwave_GreedyAlgorithm' |
| Opt_Heuristics | 'no device needed' |
| RandomSamplerQUBO | 'no device needed' |

## 5. report_options
| attribute_name | type | description |
| ------------- | ----- | --------------------------|
| visualize_distribution_of_obj_values | bool | if =True barplots will be saved for the sampling results that will show the distribution of objective values |
| analyse_only_best_sols | bool | if =True only a certain amount of samples will be analysed for feasibility and those will be the solutions with the best QUBO-objective values |
| number_of_sols_to_be_analysed | int | if analyse_only_best_sols=True, this will be the number of solutions that will be analysed for feasibility. This will be the solutions with the x best QUBO-objective values |
| save_feasible_sols_to_excel | bool | if True, a excel file with the feasible solutions will be saved to the export path |
| visualize_feasible_solutions | bool | if True, the solutions that are feasible will be visualized in a problem-specific way in the export path |
| calc_qscore | bool | if True, it tries to calculate the qscore of the solving method |
| varnumber_cutoffpoint_for_lpfile_creation | int | the LP-files are only generated if the opt-problem has less variables than here specified |
                  