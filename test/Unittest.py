# -*- coding: utf-8 -*-
"""
Created on 17.04.2024

@author: Eric Stopfer
"""
import unittest
import os
import time

from src.run_multiple_optimizations import execute_benchmark_run


class Unittest(unittest.TestCase):
    def __init__(self):
        """
        unittesting function that tests some features of the code
        """        
        # %% Executable-Test 
        test_config_files_path = os.path.join(os.path.dirname(__file__), "unittest_executable_config.json")
        self.test_executable(test_config_files_path, 
                             parallel_calculation=False,
                             number_of_processors_for_parallel = 4)
        
        # %%
        #self.test_feature_x()
        # %%
        return   


    def test_executable(self, 
                        unittest_config_path: str, 
                        parallel_calculation: bool,
                        number_of_processors_for_parallel: int
                        ):
        """
        function that tests if the optimization-run-program 
        terminates without an error for all files in the input-path
        """        
        execute_benchmark_run(config_file_path=unittest_config_path,
                              parallel_calculation=parallel_calculation,
                              number_of_processors_for_parallel = number_of_processors_for_parallel,
                              existing_folder_name=f"unittest_run_{time.strftime('%Y_%m_%d_%H_%M_%S')}",
                              calculate_qscore_afterwards=False
                              )

    
    def test_feature_x(self):
        """
        function that executes the unittest for a certain feature
        """
        a = 4
        b = 4
        self.assertEqual(a, b)
        


if __name__ == "__main__":
    unittest = Unittest()
