# -*- coding: utf-8 -*-
"""
Created on 07.03.2024

@author: Eric Stopfer
"""
import pdb
import json
import shutil
import logging
import os
import time

from src.utils.error_utils import CustomizedError
from src.utils.logging_utils import configure_logging



class Config:
    """
    class that creates a configuration for the optimization
    if a config-file exists, some of those configurations are overwritten
    """  
    PROJECT_NAME = "QuOpt"
    BASE_PATH_PROJECT = os.path.dirname(os.path.dirname(__file__)) # C:\...\quopt
    BASE_PATH_CONFIG = os.path.dirname(__file__)  # C:\...\quopt\config
    BASE_PATH_SRC = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")  # C:\...\quopt\src
    IMPORT_PATH = os.path.join(os.path.dirname(__file__), "config_files") # C:\...\quopt\config\config_files
    EXPORT_PATH_BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results") # C:\...\quopt\results
    GUROBI_LICENSE_AVAILABLE = False
     
    
    def __init__(self,
                 problem_config_file_path: str, 
                 solver_config_file_path: str, 
                 report_config_file_path: str, 
                 benchmark_export_folder_path: str == '',
                 create_export_folder: bool = True
                 ):
        """
        function that creates default config, loads a problem and a solver config file 
        if it exists and checks for the validity of the configuration
        """
        self.__create_default_config__(benchmark_export_folder_path)
        self.__load_configfile_config__(problem_config_file_path)
        self.__load_configfile_config__(solver_config_file_path)
        self.__load_configfile_config__(report_config_file_path)        
        self.__check_config_validity__()
        
        if create_export_folder == True:
            self.__create_export_folder__()              
            self.logger = configure_logging(self.LOG_CONSOLE_LEVEL,
                                            self.LOG_FILE_LEVEL,
                                            os.path.join(self.EXPORT_PATH, self.LOG_FILE_NAME))
            self.__write_json_to_export_folder__(problem_config_file_path)
            self.__write_json_to_export_folder__(solver_config_file_path)
            self.__write_json_to_export_folder__(report_config_file_path)
            with open(os.path.join(self.EXPORT_PATH, "problemsolverreport_combination.json"), "w") as file:
                json.dump([problem_config_file_path, solver_config_file_path, report_config_file_path], file, indent=4)
        

    def __create_default_config__(self, benchmark_export_folder_path: str = ''):
        """
        function that creates the default configurations 
        """
        self.total_start_time = time.time()
        
        # =============================
        # path config
        # =============================
        if benchmark_export_folder_path == "":
            self.EXPORT_PATH = os.path.join(self.EXPORT_PATH_BASE, f"{time.strftime('%Y_%m_%d_%H_%M_%S')}_Result_{self.PROJECT_NAME}") # C:\...\quopt\results\xxxxxx
        else:
            self.EXPORT_PATH = os.path.join(benchmark_export_folder_path, f"{time.strftime('%Y_%m_%d_%H_%M_%S')}_Result_{self.PROJECT_NAME}") # C:\---benchmark_export_folder_path---\xxxxxx
        
        # =============================
        # logging config
        # =============================
        self.LOG_CONSOLE_LEVEL = logging.INFO
        self.LOG_FILE_LEVEL = logging.INFO
        self.LOG_FILE_NAME = "QuOpt.log"
                

    def __load_configfile_config__(self, config_file_path: str):
        """
        function that loads a config file if it exists
        if the config file does exist, some configurations are overwritten
        """
        if os.path.exists(config_file_path):
            with open(config_file_path) as json_file:
                try:
                    data = json.load(json_file)
                except:
                    raise CustomizedError(f"DecodingError in config.json-file {config_file_path}; \
                                          check the following: \n \
                                            - has to be formatted like a dictionary \n \
                                            - comma placements between key-value-pairs \n \
                                            - no comma after the last key-value-pair \n \
                                            - boolean variables have to written small (->e.g. false instead of False) \n \
                                            - no tuples")
                # go through the keys and values of the config-file and reset the config-variables
                for key, value in data.items():
                    setattr(self, key, value)
        else:
            raise CustomizedError(f"config file path doesn't exist: {config_file_path}")
        return
        

    def __check_config_validity__(self):
        '''
        funtion to check the basic validity of the configuration. 
        there might be some other things that are checked during the problem creation later. 
        if some new problems are added to this framework, their configurations have to be added right here for validity checks. 
        if some new solvers are added to this framework, their configurations have to be added right here for validity checks.
        '''
        # %% loading config-validity-json for problem_options, solver_options, device_options
        config_validity_path = os.path.join(self.BASE_PATH_CONFIG, "config_validity.json")
        try:
            with open(config_validity_path, "r") as json_file:
                config_validity_data = json.load(json_file)
        except:
            raise CustomizedError("DecodingError in config_validity.json-file; check the following: \n \
                - has to be formatted like a dictionary \n \
                - comma placements between key-value-pairs \n \
                - no comma after the last key-value-pair \n \
                - boolean variables have to written small (->e.g. false instead of False) \n \
                - no tuples")   
                                    
        config_validity_data = self.convert_type_strings_to_type_in_dict(config_validity_data)
        
        problem_options = config_validity_data["problem_options"]
        problem_mapping_options = config_validity_data["problem_mapping_options"]
        solve_method_options = config_validity_data["solve_method_options"]
        device_options = config_validity_data["device_options"]
        report_options = config_validity_data["report_options"]
        
        # %% check the problem_config
        assert self.problem_name in problem_options.keys(), f"CONFIG_FILE_ERROR: the given problem_name '{self.problem_name}' is not in the currently configured problem options: {list(problem_options.keys())}"
        
        assert set(list(self.problem_config.keys())) == set(problem_options[self.problem_name]), f"CONFIG_FILE_ERROR: the given problem_config doesn't contain exactly the same keys as needed. should be: {list(problem_options[self.problem_name].keys())}"
        
        for attr, attr_value in self.problem_config.items():
            attr_options = problem_options[self.problem_name][attr]
            assert isinstance(attr_value, attr_options["type"]), f"CONFIG_FILE_ERROR: the given problem_config attribute '{attr}' doesn't have the specified type. should be {attr_options['type']} but was given as: {attr_value}"
            if "geq" in attr_options.keys():
                assert attr_value >= attr_options["geq"], f"CONFIG_FILE_ERROR: the given problem_config attribute '{attr}' should be greater or equal than {attr_options['geq']} but was given as: {attr_value}"
            if "leq" in attr_options.keys():
                assert attr_value <= attr_options["leq"], f"CONFIG_FILE_ERROR: the given problem_config attribute '{attr}' should be less or equal than {attr_options['leq']} but was given as: {attr_value}"
            if "allowed_values" in attr_options.keys():
                assert attr_value in attr_options["allowed_values"], f"CONFIG_FILE_ERROR: the given problem_config attribute '{attr}' should be in {attr_options['allowed_values']} but was given as: {attr_value}"
        
        # %% check the problem_mapping_config
        assert set(list(self.problem_mapping_config.keys())) == set(problem_mapping_options), f"CONFIG_FILE_ERROR: the given problem_mapping_config doesn't contain exactly the same keys as needed. should be: {list(problem_mapping_options.keys())}"
        
        for attr, attr_value in self.problem_mapping_config.items():
            attr_options = problem_mapping_options[attr]
            assert isinstance(attr_value, attr_options["type"]), f"CONFIG_FILE_ERROR: the given problem_mapping attribute '{attr}' doesn't have the specified type. should be {attr_options['type']} but was given as: {attr_value}"
            if "geq" in attr_options.keys():
                assert attr_value >= attr_options["geq"], f"CONFIG_FILE_ERROR: the given problem_mapping attribute '{attr}' should be greater or equal than {attr_options['geq']} but was given as: {attr_value}"
            if "leq" in attr_options.keys():
                assert attr_value <= attr_options["leq"], f"CONFIG_FILE_ERROR: the given problem_mapping attribute '{attr}' should be less or equal than {attr_options['leq']} but was given as: {attr_value}"
            if "allowed_values" in attr_options.keys():
                assert attr_value in attr_options["allowed_values"], f"CONFIG_FILE_ERROR: the given problem_mapping attribute '{attr}' should be in {attr_options['allowed_values']} but was given as: {attr_value}"
        
        # %% check the solve_method_config
        assert self.solve_method in solve_method_options.keys(), f"CONFIG_FILE_ERROR: the given solve_method '{self.solve_method}' is not in the currently configured solving method options: {list(solve_method_options.keys())}"
        
        assert set(list(self.solve_method_config.keys())) == set(solve_method_options[self.solve_method]), f"CONFIG_FILE_ERROR: the given solve_method_config doesn't contain exactly the same keys as needed. should be for {self.solve_method}: {list(solve_method_options[self.solve_method].keys())}"
        
        for attr, attr_value in self.solve_method_config.items():
            attr_options = solve_method_options[self.solve_method][attr]
            assert isinstance(attr_value, attr_options["type"]), f"CONFIG_FILE_ERROR: the given solve_method_config attribute '{attr}' doesn't have the specified type. should be {attr_options['type']} but was given as: {attr_value}"
            if "geq" in attr_options.keys():
                assert attr_value >= attr_options["geq"], f"CONFIG_FILE_ERROR: the given solve_method_config attribute '{attr}' should be greater or equal than {attr_options['geq']} but was given as: {attr_value}"
            if "leq" in attr_options.keys():
                assert attr_value <= attr_options["leq"], f"CONFIG_FILE_ERROR: the given solve_method_config attribute '{attr}' should be less or equal than {attr_options['leq']} but was given as: {attr_value}"
            if "allowed_values" in attr_options.keys():
                assert attr_value in attr_options["allowed_values"], f"CONFIG_FILE_ERROR: the given solve_method_config attribute '{attr}' should be in {attr_options['allowed_values']} but was given as: {attr_value}"
        
        # %% check the solve_method_device_config
        assert self.solve_method_device in device_options[self.solve_method], f"CONFIG_FILE_ERROR: the given solve_method_device '{self.solve_method_device}' is not in the currently configured solving device options: {device_options[self.solve_method]}"
        
        # %% check the report_config
        for attr, attr_value in self.report_config.items():
            attr_options = report_options[attr]
            assert isinstance(attr_value, attr_options["type"]), f"CONFIG_FILE_ERROR: the given report_config attribute '{attr}' doesn't have the specified type. should be {attr_options['type']} but was given as: {attr_value}"
        # %%
    
    
    def convert_type_strings_to_type_in_dict(self, dictionary):
        ''' 
        function that traverses a dictionary and looks into all its values and changes 
        type-strings like "int", "bool", "list" into their original python types
        '''
        type_mapping = {
            "int": int,
            "bool": bool,
            "list": list,
            "str": str,
            "float": float
        }
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.convert_type_strings_to_type_in_dict(value)
            else:
                if isinstance(value, str) and value in type_mapping.keys():
                    dictionary[key] = type_mapping[value]
        return dictionary
     
        
    def __create_export_folder__(self):
        ''' 
        function that creates the export folder for the optimization run
        '''
        if self.problem_name == "BinPacking":
            problem_size = self.problem_config['number_of_objects_for_random'] if self.problem_config['random_instance'] else len(self.problem_config['object_weights'])
        elif self.problem_name == "MarkowitzPortfolio":
            problem_size = len(self.problem_config['dax_assets']) if self.problem_config['choose_dax_assets'] else len(self.problem_config['nasdaq_assets']) if self.problem_config['choose_nasdaq_assets'] else self.problem_config['number_of_assets_for_random'] if self.problem_config['random_instance'] else len(self.problem_config['asset_returns'])
        elif self.problem_name == "TSP":
            problem_size = self.problem_config['number_of_nodes_for_random'] if self.problem_config['random_instance'] else len(self.problem_config['nodes'])
        elif self.problem_name == "CVRP":
            problem_size = self.problem_config['number_of_nodes_for_random'] if self.problem_config['random_instance'] else len(self.problem_config['nodes'])
        else:
            raise CustomizedError(f"didn't configure the problem {self.problem_name} for the naming of the export folder")
        
        path = self.EXPORT_PATH + '_' + self.problem_name + '_size' + str(problem_size) + '_' + self.solve_method + '_' + self.solve_method_device 
        while os.path.exists(path):
            path = path + "_"
        os.makedirs(path, exist_ok=True)
        self.EXPORT_PATH = path
        return
    

    def __write_json_to_export_folder__(self, config_file_path: str):
        """
        function that saves the config-object as a dict to json
        """        
        origin_path = config_file_path
        
        dest_path = self.EXPORT_PATH
        
        shutil.copy2(origin_path, dest_path)
            
        self.logger.info(f"The config '{origin_path}' was saved as JSON-format to the export folder \n")
        return
