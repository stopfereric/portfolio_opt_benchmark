# -*- coding: utf-8 -*-
"""
Created on 12.04.2024

@author: Eric Stopfer
"""
import os
import json

from config.config import Config


class Report:
    """
    this Report-class contains all attributes and metrics
    that are necessary for an evaluation of an optimization run
    """
    def __init__(self, config: Config):
        '''
        function to initialize an optimization report. 
        It assigns variables to store information about the problem, solving method and device.
        It initializes a time measurements and a solution dictionary that will be 
        filled during the optimization.   
        '''
        #store information about the problem that should be solved
        self.problem_name = config.problem_name
        self.problem_config = config.problem_config
        
        #store information about the solving method
        self.solve_method = config.solve_method
        self.solve_method_config = config.solve_method_config
        
        #store information about the solver
        self.solve_method_device = config.solve_method_device
        
        #initialize time measurement dictionary
        self.time_measurements = {}
        
        #initialize the solution quality dictionary
        self.solution_quality = {}
        
        #initialize the calculation statistics dictionary
        self.calculation_statistics = {}
        
        config.logger.info("Created an initial Report-object for the optimization run.")
        config.logger.info("It contains information about the configuration of the optimization-run.")
        config.logger.info("Also dictionaries for time_measurements, solution quality and other calculation statistics are initialized in the Report-object. \n")
        return
    
    def write_to_json(self, config: Config):
        """
        function that saves the report-object as a dict to json
        """
        if not os.path.exists(config.EXPORT_PATH):
            os.makedirs(config.EXPORT_PATH, exist_ok=True)
        
        report_dict = self.__dict__
        json_file_path = os.path.join(config.EXPORT_PATH, "report.json")
        with open(json_file_path, "w") as file:
            json.dump(report_dict, file, indent=4) # indent=4 adds for new keys of the dict 4 spaces -> improves readability of the json
            
        config.logger.info("The report was saved as JSON-format to the export folder\n")
        return