# -*- coding: utf-8 -*-
"""
Created on 07.03.2024

@author: Eric Stopfer
"""
import pdb
import os
import time
import random
import json
import math
import copy
import numpy as np
import pandas as pd
import dataclasses

from gurobipy import Model
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import from_gurobipy
from qiskit_optimization.converters import InequalityToEquality, IntegerToBinary
from qiskit_optimization.problems.constraint import ConstraintSense

from config.config import Config
from src.report import Report
from src.utils.error_utils import CustomizedError
from src.utils.qc_utils import transform_gurobipy_model_to_qubo, transform_qubo_to_ising
from src.utils.opt_utils import Opt_Model
from src.utils.visualisation_utils import draw_return_vs_volatility


class MarkowitzPortfolio:
    '''
    Portfolio optimization is the process of selecting an optimal portfolio (asset distribution), 
    out of a set of considered portfolios, according to some objective. The objective typically 
    maximizes factors such as expected return, and minimizes costs like financial risk, resulting 
    in a multi-objective optimization problem
    (source: https://en.wikipedia.org/wiki/Portfolio_optimization)
    '''
    def __init__(self, config: Config, report: Report):
        """
        Function that creates the MarkowitzPortfolio-problem-object.
        """
        asset_names, asset_returns, asset_limits, asset_covariance_matrix, opt_goal, max_volatility, min_return, delta_risk_aversion \
            = self.create_problem(config, report)
        self.problem_instance = ProblemInstanceParams(number_of_assets=len(asset_returns), 
                                                      asset_names=asset_names,
                                                      asset_returns=asset_returns, 
                                                      asset_limits=asset_limits, 
                                                      asset_covariance_matrix=asset_covariance_matrix, 
                                                      opt_goal=opt_goal,
                                                      max_volatility=max_volatility,
                                                      min_return=min_return,
                                                      delta_risk_aversion=delta_risk_aversion)        
        self.check_problem_config()
        config.logger.info("Created a MarkowitzPortfolio problem instance with \n \
                     %s assets of mean returns %s. \n \
                     The maximum shares of the assets as part of the portfolio are %s. \n \
                     The covariance_matrix of the assets is specified as: %s."
                     %(self.problem_instance.number_of_assets, self.problem_instance.asset_returns, self.problem_instance.asset_limits, self.problem_instance.asset_covariance_matrix))
        config.logger.info("The opt_goal is '%s' with %s \n"
                     %(self.problem_instance.opt_goal, 
                       "max_volatility "+str(self.problem_instance.max_volatility) if self.problem_instance.opt_goal=="max_return_with_constrained_volatility" else "min_return "+str(self.problem_instance.min_return)))
        return


    def create_problem(self, config: Config, report: Report) -> (list, np.array, np.array, np.array, str, float, float):
        '''
        function that creates a problem instance for a MarkowitzPortfolio.
        If in the config-file specified, the problem instance is created randomly.
        Otherwise it is overtaken from the config-file
        '''
        start_time = time.time()
        # %% read the configuration
        try:
            asset_returns = config.problem_config['asset_returns']
            asset_limits = config.problem_config['asset_limits']
            asset_covariance_matrix = config.problem_config['asset_covariance_matrix']
            opt_goal = config.problem_config['opt_goal']
            max_volatility = config.problem_config['max_volatility']
            min_return = config.problem_config['min_return']
            delta_risk_aversion = config.problem_config['delta_risk_aversion']
            choose_dax_assets = config.problem_config['choose_dax_assets']
            dax_assets = config.problem_config['dax_assets']
            choose_nasdaq_assets = config.problem_config['choose_nasdaq_assets']
            nasdaq_assets = config.problem_config['nasdaq_assets']
            random_instance = config.problem_config['random_instance']
            stock_market_for_random_instance = config.problem_config['stock_market_for_random_instance']
            number_of_assets_for_random = config.problem_config['number_of_assets_for_random']
        except:
            config.logger.error("Error during access of problem configuration. The config didn't contain the necessary attributes")
            raise CustomizedError(error_message="Error during access of problem configuration. The config didn't contain the necessary attributes")
        
        # %% validity check
        if (choose_dax_assets == True or choose_nasdaq_assets == True) and random_instance == True:
            raise CustomizedError("CONFIG-FILE-ERROR: in problem_config only one of 'choose_dax/nasdaq_assets' and 'random_instance' can be true.")
        
        # %% create personalised dax problem instance
        if choose_dax_assets == True or choose_nasdaq_assets == True:
            asset_names = dax_assets if choose_dax_assets == True else nasdaq_assets
            stock_market = "dax" if choose_dax_assets == True else "nasdaq"
            asset_names, asset_returns, asset_limits, asset_covariance_matrix =  \
                self.create_markowitz_instance(random_instance=False, 
                                               asset_names=asset_names,
                                               stock_market=stock_market)
            with open(os.path.join(config.EXPORT_PATH, "asset_names_and_returns.json"), "w") as json_file:
                json.dump(dict(zip(asset_names, [str(round(100*ret, 1))+"%" for ret in asset_returns])), json_file, indent=4)
            config.logger.info("We created a MarkowitzPortfolio instance with %s %s assets from a given list \n"%(len(dax_assets) if choose_dax_assets else len(nasdaq_assets), stock_market))
        
        # %% create random dax problem instance
        elif random_instance == True:
            asset_names, asset_returns, asset_limits, asset_covariance_matrix =  \
                self.create_markowitz_instance(random_instance=True,
                                               number_of_assets=number_of_assets_for_random,
                                               stock_market=stock_market_for_random_instance)
            with open(os.path.join(config.EXPORT_PATH, "asset_names_and_returns.json"), "w") as json_file:
                json.dump(dict(zip(asset_names, [str(round(100*ret, 1))+"%" for ret in asset_returns])), json_file, indent=4)
            config.logger.info("We created a random MarkowitzPortfolio instance with %s %s assets \n"%(number_of_assets_for_random, stock_market_for_random_instance))
            
        # %% overtake basic asset config
        else:
            asset_returns = np.array(asset_returns, dtype=np.float64)
            asset_limits = np.array(asset_limits, dtype=np.float64)
            asset_names = list(range(len(asset_returns)))
            asset_covariance_matrix = np.array(asset_covariance_matrix, dtype=np.float64)
            config.logger.info("We overtake the MarkowitzPortfolio configuration from the config-file \n")  
        # %%
        opt_goal = opt_goal
        if random_instance == True:
            #reset the max_volatility/min_return because it really depends on the randomly chosen assets
            min_return, max_volatility = self.reset_max_vola_and_min_ret(asset_returns, asset_covariance_matrix, target_quantile=0.8)
        else:
            max_volatility = max_volatility
            min_return = min_return
        report.time_measurements['problem_creation'] = time.time() - start_time
        # %%
        
        return asset_names, asset_returns, asset_limits, asset_covariance_matrix, opt_goal, max_volatility, min_return, delta_risk_aversion
    
    
    def check_problem_config(self):
        '''
        function that checks certain elements of the problem config that 
        weren't checked in config.py before
        '''
        for asset_return in self.problem_instance.asset_returns:
            assert isinstance(asset_return, float), "CONFIG_FILE_ERROR: asset_returns has an element that is not a float"
            
        for asset_limit in self.problem_instance.asset_limits:
            assert isinstance(asset_limit, float), "CONFIG_FILE_ERROR: asset_limits has an element that is not a float"
        
        assert len(self.problem_instance.asset_limits) == len(self.problem_instance.asset_returns), "CONFIG_FILE_ERROR: asset_returns doesn't match dimension of asset_limits"
        
        assert self.problem_instance.asset_covariance_matrix.shape[0] == self.problem_instance.asset_covariance_matrix.shape[1], "CONFIG_FILE_ERROR: asset_covariance_matrix has to be a quadratic matrix"
        assert self.problem_instance.asset_covariance_matrix.shape[0] == len(self.problem_instance.asset_returns), "CONFIG_FILE_ERROR: dimension of asset_covariance_matrix doesn't match with asset_returns"
        assert np.issubdtype(self.problem_instance.asset_covariance_matrix.dtype, np.floating), "CONFIG_FILE_ERROR: all elements of asset_covariance_matrix have to be floats"
        
    
    @staticmethod
    def create_markowitz_instance(random_instance: bool,
                                    asset_names: list = [], 
                                    number_of_assets: int = 0,
                                    stock_market: str = ''
                                    ) -> (np.array, np.array, np.array):
        '''
        function that creates a random problem instance for a MarkowitzPortfolio.
        It returns:
            a list-array for the mean returns of the assets
            a list-array for the maximum shares of the assets as part of the portfolio
            a matrix-array for the covariance between the assets
        '''
        #read test dataset
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        if stock_market != '':
            asset_returns_df = pd.read_csv(current_file_directory+'\\'+f'{stock_market}_annual_returns.csv', delimiter='\t')
            covariance_matrix_df = pd.read_csv(current_file_directory+'\\'+f'{stock_market}_annualized_covariance_matrix.csv', delimiter='\t')
        else:
            asset_returns_df = pd.read_csv(current_file_directory+'\\'+'dax_annual_returns.csv', delimiter='\t')
            covariance_matrix_df = pd.read_csv(current_file_directory+'\\'+'dax_annualized_covariance_matrix.csv', delimiter='\t')
        
        if random_instance == True:
            #select a few of those assets
            all_asset_names = asset_returns_df.columns.to_list()
            number_of_assets = min(number_of_assets, len(all_asset_names))
            asset_names = random.sample(all_asset_names, number_of_assets)        
        
        #cut the test dataset to only the selected assets
        selected_assets_returns_df = asset_returns_df[asset_names]
        selected_assets_returns_list = selected_assets_returns_df.values.tolist()[0]
        asset_returns_list = [float(item) for item in selected_assets_returns_list]
        asset_returns = np.array(asset_returns_list, dtype=np.float64)
        selected_assets_covariance_matrix_df = covariance_matrix_df.loc[asset_names, asset_names]
        selected_assets_covariance_matrix_list = selected_assets_covariance_matrix_df.values.tolist()
        cov_matrix_list = [[float(item) for item in row] for row in selected_assets_covariance_matrix_list]
        cov_matrix = np.array(cov_matrix_list, dtype=np.float64)
        
        #create asset limits
        max_asset_share = float(max(0.1, min(1, 3 / len(asset_returns_list))))
        asset_limits = [max_asset_share] * len(asset_returns_list)
                
        return asset_names, asset_returns, asset_limits, cov_matrix
    
    
    @staticmethod 
    def reset_max_vola_and_min_ret(asset_returns: list, 
                                   asset_covariance_matrix: list,
                                   target_quantile: float) -> (float, float):
        ''' 
        function that resets max_volatility and min_return of a portfolio to a target quantile of its
        respective asset returns and asset variances.
        gets called if config is set to random choosing of dax assets
        '''
        assert target_quantile <= 1 and target_quantile >=0; "target quantile has to be between 0 and 1"
        
        min_return = np.percentile(asset_returns, target_quantile*100)
                
        asset_variances =  [asset_covariance_matrix[i][i] for i in range(len(asset_covariance_matrix))]
        max_volatility = np.percentile(asset_variances, target_quantile*100)
        
        return min_return, max_volatility
        
        
    
    def map_problem(self, config: Config, report: Report):
        '''
        Function that maps the MarkowitzPortfolio problem instance to a formulation 
        that fits the configured solver.
        '''
        start_time = time.time()
        
        solve_method = config.solve_method
        if solve_method == "MIP_Solver":
            self.problem_mapping = self.map_problem_to_mip(config)
        elif solve_method in ["QuantumAnnealer", "SimulatedAnnealer", "Tabu_Search", "Greedy_Algorithm", "Opt_Heuristics"]:
            self.problem_mapping = self.map_problem_to_qubo(config)
        elif solve_method in ["QAOA", "RandomSamplerQUBO"]:
            self.problem_mapping = self.map_problem_to_ising(config)
        else:
            raise CustomizedError(error_message="Error during creation of the mapping. check if the wanted mapping is implemented and if the correct solver is accessed")
        
        report.time_measurements['problem_mapping'] = time.time() - start_time
    
    
    def map_problem_to_mip(self, config: Config) -> Model:
        """
        Function that maps the MarkowitzPortfolio problem instance to a MIP formulation
        (->Mixed-Integer-Program).
        The problem instance consists of asset_returns, asset_limits, asset_cov_matrix, risk_tolerance.
        Returns a docplex MIP model.
        """               
        model_type = "pyscipopt" if config.solve_method_device == "SCIP" else "gurobipy"
        
        self.markowitz_mip = self.create_non_discretized_markowitz_problem(lower_bound_on_sum_of_asset_weights_does_not_exist=True, model_type=model_type, config=config)
        num_vars = self.markowitz_mip.get_number_of_variables()
        if num_vars <= config.report_config['varnumber_cutoffpoint_for_lpfile_creation']:
            self.markowitz_mip.write_model(path=config.EXPORT_PATH, filename="MarkowitzPortfolio_MIP.lp")
        
        if config.solve_method == "MIP_Solver" and config.problem_config['if_mipsolver_then_solvediscretizedproblem'] == False:
            return self.markowitz_mip
        
        elif config.solve_method == "MIP_Solver" and config.problem_config['if_mipsolver_then_solvediscretizedproblem'] == True:
            self.markowitz_mip_discretized = self.create_non_discretized_markowitz_problem(lower_bound_on_sum_of_asset_weights_does_not_exist=False, model_type=model_type, config=config)
            num_vars = self.markowitz_mip_discretized.get_number_of_variables()
            if num_vars <= config.report_config['varnumber_cutoffpoint_for_lpfile_creation']:
                self.markowitz_mip_discretized.write_model(path=config.EXPORT_PATH, filename="MarkowitzPortfolio_MIP_of_discretized_problem.lp")
            return self.markowitz_mip_discretized
            
        elif config.solve_method in ["QAOA", "QuantumAnnealer", "SimulatedAnnealer", "Tabu_Search", "Greedy_Algorithm", "Opt_Heuristics", "RandomSamplerQUBO"]:
            self.markowitz_mip_for_qubo = self.create_discretized_markowitz_problem(model_type=model_type, config=config)
            num_vars = self.markowitz_mip_for_qubo.get_number_of_variables()
            if num_vars <= config.report_config['varnumber_cutoffpoint_for_lpfile_creation']:
                self.markowitz_mip_for_qubo.write_model(path=config.EXPORT_PATH, filename="MarkowitzPortfolio_MIPforQUBO.lp")
            return self.markowitz_mip_for_qubo
        else:
            raise CustomizedError("missing solve method for the incompatibility constraint construction")
     
        
    def create_non_discretized_markowitz_problem(self,
                                                 lower_bound_on_sum_of_asset_weights_does_not_exist: bool,
                                                 model_type: str,
                                                 config: Config
                                                 ) -> Model:
        ''' 
        function that creates the general markowitz portfolio problem that has 
        variables for the portfolio asset weights that are NOT discretized. 
        The return is a docplex model
        '''
        # %% initialize the problem data
        asset_returns = self.problem_instance.asset_returns
        asset_limits = self.problem_instance.asset_limits
        asset_cov_matrix = self.problem_instance.asset_covariance_matrix
        max_volatility = self.problem_instance.max_volatility
        min_return = self.problem_instance.min_return
        opt_goal = self.problem_instance.opt_goal
        delta_risk_aversion = self.problem_instance.delta_risk_aversion
        
        # %%
        markowitz_mip = Opt_Model(model_type=model_type, model_name="MarkowitzPortfolio", config=config)
        config.logger.info("start the creation of the MarkowitzPortfolio-MIP \n")
        
        # the decision to be made in the MarkowitzPortfolio optimization are the shares of the assets in the portfolio
        # add continous variables for each asset between 0 and 1
        number_of_assets = self.problem_instance.number_of_assets
        asset_vars = []
        for i in range(number_of_assets):
            asset_vars.append(markowitz_mip.add_variable(
                name = f"w_{i}",
                lb=0.0,
                ub=asset_limits[i],
                vtype="C"
                ))
        config.logger.info("added continous variables w_i ∈ [0, asset_limit] for the weight of asset i in the portfolio")
        self.variables_vector = [asset_vars[i] for i in range(number_of_assets)]
        
        # add constraint that asset weights sum up to 1
        if lower_bound_on_sum_of_asset_weights_does_not_exist:
            markowitz_mip.add_constraint(
                lhs = sum(self.variables_vector), 
                sense = "==", 
                rhs = 1,
                name = "constr_asset_weights_sum_up_to_1"
                )
        else:
            # the following is necessary to benchmark the solver results, because in the discretized model there can be feasible solutions whose weights are less than 1 --> this is necessary so that the MIP-Solver's result is not worse
            markowitz_mip.add_constraint(
                lhs = sum(self.variables_vector), 
                sense = "<=", 
                rhs = 1,
                name = "constr_asset_weights_sum_up_to_1_ct1"
                )
            lower_max_deviation, upper_max_deviation = self.get_max_deviation_of_weights_from_1(config)
            markowitz_mip.add_constraint(
                lhs = sum(self.variables_vector), 
                sense = ">=", 
                rhs = 1 - lower_max_deviation,
                name = "constr_asset_weights_sum_up_to_1_ct2"
                )
        config.logger.info("added constraint so that the asset weights sum up to 1")
        
        if opt_goal == 'max_return_with_constrained_volatility':
            # objective: maximize expected return of portfolio
            neg_portfolio_return = 0
            for i in range(number_of_assets):
                neg_portfolio_return += -asset_returns[i] * asset_vars[i]
            markowitz_mip.set_objective(neg_portfolio_return, sense='min')
            config.logger.info("added the objective with goal to maximize the expected return of asset portfolio")
        
            # constraint: portfolio volatility must not exceed max_volatility
            volatility = 0
            for i in range(number_of_assets):
                for j in range(number_of_assets):
                    volatility += (asset_cov_matrix[i, j] * asset_vars[i] * asset_vars[j])
            markowitz_mip.add_constraint(
                lhs = volatility, 
                sense = "<=", 
                rhs = max_volatility, 
                name="constr_exp_volat_leq_max_volat")
            config.logger.info("added constraint so that the portfolio doesn't pass a certain volatility")
        
        elif opt_goal == 'min_volatility_with_constrained_return':
            # objective: minimize portfolio volatility
            volatility = 0
            for i in range(number_of_assets):
               for j in range(number_of_assets):
                   volatility += asset_cov_matrix[i, j] * asset_vars[i] * asset_vars[j]
            markowitz_mip.set_objective(volatility, sense='min', quadratic_objective=True)
            config.logger.info("added the objective with goal to minimize the volatility of the asset portfolio")
            
            # constraint: expected return must be above min_return
            portfolio_return = 0
            for i in range(number_of_assets):
                portfolio_return += asset_returns[i] * asset_vars[i]
            markowitz_mip.add_constraint(
                lhs = portfolio_return, 
                sense = ">=", 
                rhs = min_return, 
                name="constr_exp_return_geq_min_return")
            config.logger.info("added constraint so that the portfolio return is higher than a certain parameter")
        
        elif opt_goal == "max_return_min_volatility":
            # multi-objective: minimize volatility and maximize return
            obj = 0
            for i in range(number_of_assets):
                for j in range(number_of_assets):
                    obj += delta_risk_aversion * asset_cov_matrix[i, j] * asset_vars[i] * asset_vars[j]
            for i in range(number_of_assets):
                obj += - asset_returns[i] * asset_vars[i]
            markowitz_mip.set_objective(obj, sense='min', quadratic_objective=True)
            config.logger.info("added the multi-objective with goal to minimize the volatility and maximizing the return of the asset portfolio")
        
        else:
            raise CustomizedError(f"invalid opt_goal-configuration for '{opt_goal}'. check config-file")
        
        config.logger.info("finished the creation of the MarkowitzPortfolio-MIP. It will be saved to the export-folder \n")
                
        return markowitz_mip
    
    
    def create_discretized_markowitz_problem(self, model_type: str, config: Config) -> Model:
        ''' 
        function that creates the Markowitz model suitable for QUBO transformation
        the decision to be made in the MarkowitzPortfolio optimization are the shares of the assets in the portfolio
        for quantum methods we need binary variables -->discretize the space between 0 and 1
        we encode the discretization binary --> split the definition area into 2^{n} parts
        for each asset we add 2^{n} binary variables 
        furthermore we limit the maximum share of an asset here with integration of asset_limits
        '''
        # We start to create another MIP that is for the QUBO-transformation for quantum methods.
        config.logger.info("We start to create another MIP that is for the QUBO-transformation for quantum methods. \n")
        
        # %% initialize the problem data
        asset_returns = self.problem_instance.asset_returns
        asset_limits = self.problem_instance.asset_limits
        asset_cov_matrix = self.problem_instance.asset_covariance_matrix
        max_volatility = self.problem_instance.max_volatility
        min_return = self.problem_instance.min_return
        opt_goal = self.problem_instance.opt_goal
        delta_risk_aversion = self.problem_instance.delta_risk_aversion
        
        # %% add variables to the model        
        markowitz_mip_for_qubo = Opt_Model(model_type=model_type, model_name="MarkowitzPortfolio", config=config)
        n_discretization = config.problem_config['variable_discretization_n']
        number_of_assets = self.problem_instance.number_of_assets
        
        asset_vars = {}
        for i in range(number_of_assets):
            for n in range(n_discretization + 1):
                asset_vars[i, n] = markowitz_mip_for_qubo.add_variable(
                    name = f"w_{i}_{n}",
                    vtype="B"
                    )
        self.var_names = [f"w_{i}_{n}" for i in range(number_of_assets) for n in range(n_discretization + 1)]
        config.logger.info("added binary variables w_i_n ∈ [0,1] --> weight in the portfolio of asset i of discretization step n")
        
        # create a variables vector that represents the discretisation
        variable_weight_list = [1 / (2 ** (n + 1)) for n in range(n_discretization)]
        variable_weight_list.append(variable_weight_list[-1]) #add the last element a second time
        self.variables_vector = [
            sum(
                asset_vars[i, n] * variable_weight_list[n] * asset_limits[i] 
                for n in range(n_discretization + 1)
                ) 
            for i in range(number_of_assets)
            ]
        
        # add constraint that asset weights sum up to 1, we have LEQ because because of discretisation it might not be exactly 1
        markowitz_mip_for_qubo.add_constraint(
            lhs = sum(self.variables_vector), 
            sense = "==", 
            rhs = 1,
            name = "constr_asset_weights_sum_up_to_1"
            )
        config.logger.info("added constraint(s) so that the asset weights sum up to 1")
        
        # Set the objective depending on the chosen opt_goal
        if opt_goal in ['max_return_with_constrained_volatility', "max_return_min_volatility"]:
            # add model objective to maximize the expected return
            # we also write the portfolio volatility in the objective directly because as a constraint, it would get ugly for the qubo-transformation--> would not be quadratic anymore
            markowitz_mip_for_qubo.set_objective(
                obj = - np.dot(self.variables_vector, asset_returns) + delta_risk_aversion * np.dot(self.variables_vector, np.dot(asset_cov_matrix, self.variables_vector)), 
                sense = "min"
                )
            config.logger.info("added the objective with goal to maximize the expected return of asset portfolio")
        
        elif opt_goal == 'min_volatility_with_constrained_return':
            # add model objective to minimize the portfolio volatility
            markowitz_mip_for_qubo.set_objective(
                obj = np.dot(self.variables_vector, np.dot(asset_cov_matrix, self.variables_vector)), 
                sense = "min"
                )
            config.logger.info("added the objective with goal to minimize the volatility of the asset portfolio")
        
            # add constraint that the portfolio expected return is higher than the min_return-parameter
            self.return_correction_factor = 0.01
            markowitz_mip_for_qubo.add_constraint(
                lhs = sum([self.variables_vector[i] * asset_returns[i] 
                           for i in range(number_of_assets)]), 
                sense = "==", 
                rhs = min_return + self.return_correction_factor,
                name = "constr_exp_return_geq_min_return"
                )
            config.logger.info("added constraint so that the portfolio return is higher than a certain parameter")
        
        else:
            raise CustomizedError(f"discretised Mip-modelling of opt_goal {opt_goal} hasn't been configured yet")
            
        config.logger.info("finished the creation of the MarkowitzPortfolio-MIPforQUBO. It will be saved to the export-folder \n")
        return markowitz_mip_for_qubo
        
        
    def get_max_deviation_of_weights_from_1(self, config):
        ''' 
        function to calculate the maximum deviation of the sum of the weights from 1.
        The reason for this is that while discretizing the weight variables, it might not
        be possible to get exactly to 1 while summing up the weights.
        '''
        if config.problem_config['choose_dax_assets'] == True or config.problem_config['choose_nasdaq_assets'] == True or config.problem_config['random_instance'] == True:
            smallest_possible_var_factor = self.problem_instance.asset_limits[0] * (1/2**config.problem_config['variable_discretization_n'])
            lower_max_deviation = 1 - math.floor(1//smallest_possible_var_factor) * smallest_possible_var_factor # math.floor(1/smallest_possible_var_factor) is the number of times the var factor fits completely into 1 
            upper_max_deviation = math.ceil(1/smallest_possible_var_factor) * smallest_possible_var_factor - 1
        else:
            smallest_possible_var_factors = [asset_limit * (1/2**config.problem_config['variable_discretization_n']) for asset_limit in self.problem_instance.asset_limits]
            lower_max_deviation = max([1 - math.floor(1/var_factor) * var_factor for var_factor in smallest_possible_var_factors])
            upper_max_deviation = max([math.ceil(1/var_factor) * var_factor - 1 for var_factor in smallest_possible_var_factors])
        return lower_max_deviation, upper_max_deviation
    
    
    def map_problem_to_qubo(self, config: Config) -> (dict, QuadraticProgram, QuadraticProgram):
        """
        Function that maps the problem instance to QUBO formulation (->Quadratic-Unconstrained-Binary-Optimization).
        The problem instance consists of object_weights, bin_capacity and incompatible_objects.
        Returns the Qiskit-QuadraticProgram and a dictionary with the qubo-coefficients.
        """
        # %% create a docplex MIP formulation for the problem instance
        markowitz_mip = self.map_problem_to_mip(config)
        
        # %% read the config to get necessary attributes for the qubo transformation
        try:
            penalty_factor = config.problem_mapping_config["qubo_penalty_factor"]
            choose_penalty_feas_better_than_infeas = config.problem_mapping_config["choose_penalty_worst_feasible_better_than_best_infeasible"]
            trim_inequalities = config.problem_mapping_config["trim_inequalities"]
            max_slack_vars_per_inequ = config.problem_mapping_config["max_number_of_slack_vars_per_inequality"]
        except:
            config.logger.error("Error during access of problem configuration. The config didn't contain the necessary attributes")
            raise CustomizedError("Error during access of problem configuration. The config didn't contain the necessary attributes")
        
        # %% possibly change penalty factors
        if choose_penalty_feas_better_than_infeas == True: #we choose the penaltyfactor so that even the worst feasible solution has a better QUBO obj value than any infeasible solution
            #here this is:   1 / minimum_constraint_coefficient * ( weight_vars_=1_of_least_profitable_assets(=worst feasible) - all_weight_vars_=1(=best infeasible) )
            mip_qiskit = from_gurobipy(markowitz_mip)
            mip_ineq2eq = InequalityToEquality().convert(mip_qiskit)
            mip_int2bin = IntegerToBinary().convert(mip_ineq2eq)
            minimum_constraint_coefficient = 1000000 #initial value
            coeff_list = []
            for constraint in mip_int2bin.linear_constraints:
                coeff_array0 = constraint.linear.to_array()
                coeff_array0_without_zeroes = [abs(x) for x in coeff_array0 if x != 0] #get all the non-zero-coefficients of this array
                coeff_list.extend(coeff_array0_without_zeroes)
            for constraint in mip_int2bin.quadratic_constraints:
                coeff_array1 = constraint.linear.to_array()
                coeff_array1_without_zeroes = [abs(x) for x in coeff_array1 if x != 0] #get all the non-zero-coefficients of this array
                coeff_list.extend(coeff_array1_without_zeroes)
                coeff_array2 = constraint.quadratic.to_array()
                coeff_array2_without_zeroes = [abs(x) for row in coeff_array2 for x in row if x != 0] #get all the non-zero-coefficients of this matrix-array
                coeff_list.extend(coeff_array2_without_zeroes)
            min_coeff = min(coeff_list)
            if min_coeff < minimum_constraint_coefficient:
                minimum_constraint_coefficient = min_coeff
            penalty_factor = (1 / (minimum_constraint_coefficient ** 2)) * (-min(self.problem_instance.asset_returns) - (-sum(self.problem_instance.asset_returns))) # 1 / minimum_constraint_coefficient * ( weight_vars_=1_of_least_profitable_assets(=worst feasible) - all_weight_vars_=1(=best infeasible) )
        else:
            pass
                
        # %% transform the model to a qubo
        if trim_inequalities == True:
            qubo_dict, qubo, mip = transform_gurobipy_model_to_qubo(markowitz_mip, penalty_factor, trim_inequalities, max_slack_vars_per_inequ)
        else:
            qubo_dict, qubo, mip = transform_gurobipy_model_to_qubo(markowitz_mip, penalty_factor)
        
        if qubo.get_num_vars() <= config.report_config['varnumber_cutoffpoint_for_lpfile_creation']:
            qubo.write_to_lp_file(os.path.join(config.EXPORT_PATH, "MarkowitzPortfolio_QUBO"))
        self.markowitz_qubo = qubo
            
        # %% change senses of linear constraints, because for feasibility analysis we want inequalities instead of equalities
        constraints_to_remove = []
        new_constraints = []
        for lin_constr in mip.linear_constraints:
            name = lin_constr.name
            if name == "constr_exp_return_geq_min_return":
                constraints_to_remove.append(name)
                linear_part = lin_constr.linear.to_array()
                rhs = lin_constr.rhs - self.return_correction_factor
                new_constraints.append((linear_part, ConstraintSense.GE , rhs , name))
                
            if name == "constr_asset_weights_sum_up_to_1":
                #we replace this EQ constraint with two constraints: one with GE and one with LE -> reason: with var-discretisation it might not be possible to exactly sum up to 1
                constraints_to_remove.append(name)
                linear_part = lin_constr.linear.to_array()
                lower_max_deviation, upper_max_deviation = self.get_max_deviation_of_weights_from_1(config)
                new_constraints.append((linear_part, ConstraintSense.LE , lin_constr.rhs + upper_max_deviation, name+"_ct1"))
                new_constraints.append((linear_part, ConstraintSense.GE, lin_constr.rhs - lower_max_deviation, name+"_ct2"))
                
        for constr_to_rm in constraints_to_remove:
            mip.remove_linear_constraint(constr_to_rm)
        for new_constr in new_constraints:
            mip.linear_constraint(new_constr[0], new_constr[1], new_constr[2], new_constr[3])
        
        if mip.get_num_vars() <= config.report_config['varnumber_cutoffpoint_for_lpfile_creation']:
            mip.write_to_lp_file(os.path.join(config.EXPORT_PATH, "MarkowitzPortfolio_MIP_for_feasibility_analysis"))
        self.markowitz_mip_for_feasibility_analysis = mip
        
        # %% create a non-discretised markowitz-mip with the replaced weights-sum-up-to-1-constraint --> will be used for qscore calculation of optimal solution
        self.markowitz_mip_for_qscore_calc = copy.copy(self.markowitz_mip)
        asset_weights_sum_up_to_1_constr = self.markowitz_mip_for_qscore_calc.model.getConstrByName("constr_asset_weights_sum_up_to_1")
        
        for constr in self.markowitz_mip_for_qscore_calc.model.getConstrs():
            name = constr.ConstrName
            if name == "constr_asset_weights_sum_up_to_1":
                self.markowitz_mip_for_qscore_calc.model.remove(constr)
        
                expr = self.markowitz_mip_for_qscore_calc.model.getRow(constr)
        
                self.markowitz_mip_for_qscore_calc.model.addConstr(expr <= 1, name=name + "_ct1")
                self.markowitz_mip_for_qscore_calc.model.addConstr(expr >= 1 - lower_max_deviation, name=name + "_ct2")
        
        num_vars = len(self.markowitz_mip_for_qscore_calc.model.getVars())
        if num_vars <= config.report_config['varnumber_cutoffpoint_for_lpfile_creation']:
            self.markowitz_mip_for_qscore_calc.model.write(os.path.join(config.EXPORT_PATH, "MarkowitzPortfolio_MIP_for_Qscore_calc.lp"))
        
        # %%
        config.logger.info("Mapped the problem instance successfully to a QUBO formulation. saved lp-file to export-folder \n")
        config.logger.info(f"This problem would require {qubo.get_num_vars()} qubits \n")
        return qubo_dict, self.markowitz_qubo, self.markowitz_mip_for_feasibility_analysis


    def map_problem_to_ising(self, config: Config) -> (np.array, np.array, float, QuadraticProgram, QuadraticProgram, dict):
        """
        Function that maps the problem instance to an Ising formulation.
        The problem instance consists of object_weights, bin_capacity and incompatible_objects.
        Returns a ising formulation consisting of ising_matrix, -_vector and -_offset as well
        as the correspondingQiskit-QuadraticProgram-QUBO.
        """    
        # %% generate the QUBO with binary variables in {0; 1}
        qubo_dict, qubo, mip = self.map_problem_to_qubo(config)
        
        # %% transform it to an Ising formulation
        ising_matrix, ising_vector, ising_offset = transform_qubo_to_ising(qubo)
            
        # %%   
        config.logger.info("Transformed the QUBO formulation successfully to an Ising formulation")
    
        return qubo_dict, qubo, mip, ising_matrix, ising_vector, ising_offset

    
    
    def postprocess_solver_result(self, 
                                  solver_result_feas_sols: dict,
                                  solver_result_all_sols: dict,
                                  config: Config, 
                                  report: Report
                                  ) -> dict:
        ''' 
        function that postprocesses the solver result 
        - adds interpretable analysis to the solver_result-dict
        - calculates additional metrics
        '''
        solver_result_interpretable = self.make_solver_result_interpretable(
            solver_result_feas_sols, config, report)
        
        if config.report_config['calc_qscore']:
            qscore_metric = self.calc_qscore_metric(
                solver_result_feas_sols, 1000, config, report)
        
        file_path_jsondump = os.path.join(config.EXPORT_PATH, 'solution_dict.json')
        with open(file_path_jsondump, 'w') as json_file:
            json.dump(solver_result_interpretable, json_file, indent=4)
        
        #close all opened gurobi server environments
        existing_models = [m for m in [
                getattr(self, "markowitz_mip", None),
                getattr(self, "markowitz_mip_for_qubo", None),
                getattr(self, "markowitz_mip_discretized", None),
                getattr(self, "markowitz_mip_for_feasibility_analysis", None),
                getattr(self, "markowitz_mip_for_qscore_calc", None)
            ] if m is not None]
        for model in existing_models:
            try:
                model.close_gurobi_server_env()
            except:
                pass
            
        return solver_result_interpretable
        
        
    def make_solver_result_interpretable(self, solver_result: dict, config: Config, report: Report) -> dict:
        '''
        function that processes the optimization result of a solver and transforms it
        to a meaningful interpretable result.
        Often, because of the transformation to Qubo there were many variables added
        that are harmful for the interpretability of the result.
        '''
        start_time = time.time()
        if config.solve_method in ['MIP_Solver']:
            #because no variable discretization was done, the variables are already interpretable, we just add some information
            if len(solver_result.keys()) != 0: # if yes, infeasible, or timelimit exceeded
                for var_values, var_values_dict in solver_result.items():
                    var_values_list = [float(item) for item in var_values.split(', ')]
                    expected_return = np.dot(self.problem_instance.asset_returns, var_values_list)
                    expected_volatility = np.dot(var_values_list, np.dot(self.problem_instance.asset_covariance_matrix, var_values_list))
                    solver_result[var_values]['expected_return'] = expected_return
                    solver_result[var_values]['expected_volatility'] = expected_volatility
                    solver_result[var_values]['expected_volatility_minus_return'] = self.problem_instance.delta_risk_aversion * expected_volatility - expected_return
                    solver_result[var_values]['bought assets'] = {
                        self.problem_instance.asset_names[asset_idx]: {
                            'weight': f"{100*round(asset_weight, 4)}%",
                            'asset_return': f"{100*round(self.problem_instance.asset_returns[asset_idx], 4)}%"
                            }
                        for asset_idx, asset_weight in enumerate(var_values_list)
                        if round(asset_weight, 4) > 0
                        }
                    solver_result[var_values]['ignored assets'] = {
                        self.problem_instance.asset_names[asset_idx]: {
                            'weight': f"{100*round(asset_weight, 4)}%",
                            'asset_return': f"{100*round(self.problem_instance.asset_returns[asset_idx], 4)}%"
                            }
                        for asset_idx, asset_weight in enumerate(var_values_list)
                        if round(asset_weight, 4) <= 0
                        }
            
        elif config.solve_method in ["QAOA", "QuantumAnnealer", "SimulatedAnnealer", "Tabu_Search", "Greedy_Algorithm", "Opt_Heuristics", "RandomSamplerQUBO"]:
            
            n_discretization = config.problem_config['variable_discretization_n']
            variable_factors_from_discretization = [1 / (2 ** (i+1)) for i in range(n_discretization)]
            variable_factors_from_discretization.append(variable_factors_from_discretization[-1])
            
            for bit_tuple, bit_tuple_attr in solver_result.items():
                
                interpretable_var_values = {} # initial dict
                for asset in range(self.problem_instance.number_of_assets): # fill initial dict
                    interpretable_var_values[f"w_{asset}"] = None
                                
                # make the var-value-dict interpretable
                var_dict = bit_tuple_attr['var_dict']
                for var_name in interpretable_var_values.keys():
                    var_index = int(var_name.split("_")[1])
                    asset_weight = 0
                    for i in range(n_discretization+1):
                        var_value = var_dict[var_name + '_' + str(i)]
                        asset_weight += self.problem_instance.asset_limits[var_index] * var_value * variable_factors_from_discretization[i]
                    interpretable_var_values[var_name] = asset_weight
                    
                #rescale the var-values if sum of asset weights > 1
                sum_of_asset_weights_from_opt = sum(interpretable_var_values.values())
                if sum_of_asset_weights_from_opt > 1:
                    for var_name in interpretable_var_values.keys():
                        interpretable_var_values[var_name] /= sum_of_asset_weights_from_opt 
                
                #write the results into the solver_result
                solver_result[bit_tuple]['interpretable_var_dict'] = interpretable_var_values
                solver_result[bit_tuple]['bought assets'] = {
                    self.problem_instance.asset_names[int(var_name[2:])]: {
                        'weight': f"{100*round(asset_weight, 4)}%",
                        'asset_return': f"{100*round(self.problem_instance.asset_returns[int(var_name[2:])], 4)}%"
                        }
                    for var_name, asset_weight in interpretable_var_values.items()
                    if round(asset_weight, 4) > 0
                    }
                solver_result[bit_tuple]['ignored assets'] = {
                    self.problem_instance.asset_names[int(var_name[2:])]: {
                        'weight': f"{100*round(asset_weight, 4)}%",
                        'asset_return': f"{100*round(self.problem_instance.asset_returns[int(var_name[2:])], 4)}%"
                        }
                    for var_name, asset_weight in interpretable_var_values.items()
                    if round(asset_weight, 4) <= 0
                    }
                
                # calculate the expected return and the associated volatility
                expected_return = float(np.dot(self.problem_instance.asset_returns, list(interpretable_var_values.values())))
                expected_volatility = float(np.dot(list(interpretable_var_values.values()), np.dot(self.problem_instance.asset_covariance_matrix, list(interpretable_var_values.values()))))
                solver_result[bit_tuple]['expected_return'] = expected_return
                solver_result[bit_tuple]['expected_volatility'] = expected_volatility
                solver_result[bit_tuple]['expected_volatility_minus_return'] = self.problem_instance.delta_risk_aversion * expected_volatility - expected_return
                solver_result[bit_tuple]['summed_up_portfolio_weights'] = float(sum(interpretable_var_values.values()))
                
        else:
            raise CustomizedError(error_message="Error during backwards mapping of solver result. Option for the used solver not implemented.")
        
        report.time_measurements['making_result_interpretable'] = time.time() - start_time
                    
        return solver_result

        
    def calc_qscore_metric(self, solver_result: dict, 
                           samplesize_random: int, 
                           config: Config, 
                           report: Report):
        ''' 
        function that calculates the Q-Score metric for MarkowitzPortfolio solution method.
        It measures how well the method performs versus a random method and the optimal solution:
        qscore_metric = (C_optmethod - C_rand) / (C_optsol - C_rand)
        '''
        start_time = time.time()
        if len(solver_result.keys()) == 0:
            config.logger.info("ERROR in QSCORE calculation. no feasible solution in opt_method found. \n")
            report.solution_quality['qscore_metric'] = 0
            report.solution_quality['approximation_ratio'] = 0
            report.time_measurements['qscore_calculation'] = time.time() - start_time
            return
        
        # %% calc the necessary numbers for the qscore metric
        C_rand, _, _ = self.calc_C_rand(samplesize_random, config, report)
        C_optmethod = self.calc_C_optmethod('solver_result', config, report, solver_result)
        C_optsol = self.calc_C_optsol(config, report)        
                
        # %% write the qscore-metric to the report     
        qscore_metric = (C_optmethod - C_rand) / (C_optsol - C_rand)
        report.time_measurements['qscore_calculation'] = time.time() - start_time
        report.solution_quality['qscore_metric'] = qscore_metric
        
        # %% write the approximation ratio to the report
        opt_goal = config.problem_config['opt_goal']
        if opt_goal == "max_return_with_constrained_volatility":
            approximation_ratio = C_optsol / C_optmethod
        elif opt_goal == "min_volatility_with_constrained_return":
            approximation_ratio = C_optmethod / C_optsol
        elif opt_goal == "max_return_min_volatility":
            approximation_ratio = C_optmethod / C_optsol
        report.solution_quality['approximation_ratio'] = approximation_ratio
        
        return qscore_metric
    
    
    def calc_C_rand(self, samplesize_random: int, config: Config, report: Report):
        ''' 
        calculate the mean objective value of a random feasible solution of the problem for the Qscore
        '''
        random_portfolio_samples, counter_random_gen = self.get_samples_with_random_heuristic(
            samplesize = samplesize_random,
            config = config,
            report = report)
        C_rand = self.calc_mean_obj_value(model = self.markowitz_mip, 
                                          samples = random_portfolio_samples, 
                                          counts_of_samples = [1]*samplesize_random,
                                          for_random_samples = True,
                                          config = config)
        return C_rand, len(random_portfolio_samples), counter_random_gen
        
    
    def calc_C_optsol(self, config: Config, report: Report):
        ''' 
        calculate the objective value of the optimal solution of the problem for the Qscore with MIPSolver
        '''
        if config.solve_method == "MIP_Solver" and config.problem_config['if_mipsolver_then_solvediscretizedproblem'] == False:
            opt_model_to_solve = self.markowitz_mip
        elif config.solve_method == "MIP_Solver" and config.problem_config['if_mipsolver_then_solvediscretizedproblem'] == True:
            opt_model_to_solve = self.markowitz_mip_discretized                   
        else:
            opt_model_to_solve = self.markowitz_mip_for_qscore_calc
        opt_model_to_solve.solve_model(
            time_limit=20, 
            absolute_gap=0, 
            relative_gap=0,
            presolve_method=0)
        opt_status = opt_model_to_solve.get_opt_status()
        if opt_status not in ['infeasible', 'unbounded']: 
            C_optsol = opt_model_to_solve.get_best_found_obj_value()
        else:
            raise CustomizedError("ERROR in QSCORE calculation. Original MIP is infeasible. \n")
        return C_optsol  
    
    
    def calc_C_optmethod(self, get_from_where: str, config: Config, report: Report, solver_result = None):
        ''' 
        returns the objective value of the best found solution of the opt-run
        '''
        if get_from_where == "report_object":
            C_optmethod = report['solution_quality']['best_feasible_solution_obj_value']
        elif get_from_where == "solver_result":
            opt_goal = config.problem_config['opt_goal']
            if opt_goal == "max_return_with_constrained_volatility":
                best_opt_method_sample_bitstr = max(solver_result, key=lambda k: solver_result[k]['expected_return'])
            elif opt_goal == "min_volatility_with_constrained_return":
                best_opt_method_sample_bitstr = min(solver_result, key=lambda k: solver_result[k]['expected_volatility'])
            elif opt_goal == "max_return_min_volatility":
                best_opt_method_sample_bitstr = min(solver_result, key=lambda k: solver_result[k]['expected_volatility_minus_return'])
            else:
                raise Exception(f"unknown opt_goal: {opt_goal}")
            best_opt_method_sample = [float(bit) for bit in best_opt_method_sample_bitstr.split(', ')]
            if config.solve_method in ['MIP_Solver'] and config.problem_config['if_mipsolver_then_solvediscretizedproblem'] == False:
                C_optmethod = self.markowitz_mip.get_best_found_obj_value()
            elif config.solve_method in ['MIP_Solver'] and config.problem_config['if_mipsolver_then_solvediscretizedproblem'] == True:
                C_optmethod = self.markowitz_mip_discretized.get_best_found_obj_value()
            else:
                opt_method_model = self.markowitz_mip_for_feasibility_analysis
                C_optmethod = self.calc_mean_obj_value(model = opt_method_model, 
                                                       samples = [best_opt_method_sample], 
                                                       counts_of_samples = [1],
                                                       for_random_samples = False,
                                                       config = config)
        else:
            raise CustomizedError(f"allowed values for 'get_from_where': ['report_object', 'solver_result'], but was chosen as {get_from_where}")
            
        return C_optmethod


    def get_samples_with_random_heuristic(self, 
                                         samplesize: int,
                                         config: Config,
                                         report: Report,
                                         consider_only_feasible_ones: bool = True,
                                         max_sampling_time: float = 100.0
                                         ) -> list:
        ''' 
        function that creates a sample of n portfolio weights that sum up to 1. 
        1) take n−1 random numbers from the interval (0,1)
        2) add a 0 and 1 to get a list of n+1 numbers. 
        3) sort the list 
        4) record the differences between two consecutive elements. 
        This gives you a list of n number that will sum up to 1. 
        Moreover this sampling is uniform. 
        This idea can be found in Donald B. Rubin, The Bayesian bootstrap Ann. Statist. 9, 1981, 130-134.
        It is further developed by checking that the weights don't violate the asset_limits
        '''
        start_time = time.time()
        number_of_assets = len(self.problem_instance.asset_limits)
        portfolio_samples = [] # initialize list to be filled
        counter = 0
        for _ in range(samplesize):
            feasible = False
            while feasible == False:
                random_values = np.random.uniform(0, 1, number_of_assets-1)
                random_values = np.append(random_values, [0, 1])
                sorted_values = np.sort(random_values)
                portfolio_weights = list(np.diff(sorted_values))
                check_limits = [diff < limit for diff, limit in zip(portfolio_weights, self.problem_instance.asset_limits)]
                counter += 1
                if consider_only_feasible_ones == True:
                    opt_goal = config.problem_config['opt_goal']
                    if opt_goal == "max_return_with_constrained_volatility":
                        check_vola = np.dot(portfolio_weights, np.dot(self.problem_instance.asset_covariance_matrix, portfolio_weights)) <= self.problem_instance.max_volatility
                        if all(check_limits) and check_vola:
                            feasible = True
                    elif opt_goal == "min_volatility_with_constrained_return":
                        check_return = np.dot(self.problem_instance.asset_returns, portfolio_weights) >= self.problem_instance.min_return
                        if all(check_limits) and check_return:
                            feasible = True
                    elif opt_goal == "max_return_min_volatility":
                        feasible = True
                    else:
                        raise CustomizedError(f"for the opt_goal '{config.problem_config['opt_goal']}' there is no random method configured to create a random feasible sample")
                else:
                    if all(check_limits):
                        feasible = True
            portfolio_samples.append(portfolio_weights)
            if time.time() - start_time > max_sampling_time:
                break
            
        report.time_measurements['qscore_getting_random_samples'] = time.time() - start_time
        report.solution_quality['counter_random_sampling_for_qscore'] = counter
        report.solution_quality['samplesize_randomsample_for_qscore'] = len(portfolio_samples)
        return portfolio_samples, counter


    def calc_mean_obj_value(self, 
                            model: QuadraticProgram, 
                            samples: list, 
                            counts_of_samples: list,
                            for_random_samples: bool,
                            config: Config
                            ) -> float:
        ''' 
        function that calculates the objective values of some samples and returns the mean of those
        '''
        obj_values = []
        opt_goal = config.problem_config['opt_goal']
        if for_random_samples == True:
            for idx in range(len(samples)):
                if opt_goal == "max_return_with_constrained_volatility":
                    obj_values.extend([- np.dot(self.problem_instance.asset_returns, samples[idx])])
                elif opt_goal == "min_volatility_with_constrained_return":
                    obj_values.extend([np.dot(samples[idx], np.dot(self.problem_instance.asset_covariance_matrix, samples[idx]))])
                elif opt_goal == "max_return_min_volatility":
                    obj_values.extend([np.dot(samples[idx], np.dot(self.problem_instance.asset_covariance_matrix, samples[idx])) - 
                                       np.dot(self.problem_instance.asset_returns, samples[idx])])
                else:
                    raise CustomizedError(f"for the opt_goal '{config.problem_config['opt_goal']}' there is no method yet to calculate the mean objective value")
        else:
            for idx in range(len(samples)):
                obj_values.extend([model.objective.evaluate(samples[idx])] * counts_of_samples[idx])
        return sum(obj_values) / len(obj_values)
            
                
    def visualize_result(self, result: dict, config: Config):
        ''' 
        function that visualises the results that are inside of a dict.
        It plots a scatterplot that show the return vs the volatility of the portfolio.
        '''
        store_dir = os.path.join(config.EXPORT_PATH, "opt_result_visualisations")
        filename = "feas_sol_return_vs_volatility"    
        draw_return_vs_volatility(result, store_dir, filename)
        
        

@dataclasses.dataclass
class ProblemInstanceParams:
    """Container for MarkowitzPortfolio problem parameters"""
    number_of_assets: int
    asset_names: list
    asset_returns: np.array
    asset_limits: np.array
    asset_covariance_matrix: np.array
    opt_goal: str
    max_volatility: float
    min_return: float
    delta_risk_aversion: float

