# -*- coding: utf-8 -*-
"""
Created on 04.09.2024

@author: Eric Stopfer
"""
import time
import pdb
import logging
import warnings


class OptimizationTimeout(Exception):
    pass

class Callback_StopScipyMinimizer():
    def __init__(self, max_sec=60):
        self.max_sec = max_sec
        self.start = time.time()
    def __call__(self, xk=None):
        elapsed = time.time() - self.start
        if elapsed > self.max_sec:
            print(f"Stop the scipy minimizer because the time limit of {self.max_sec}s was exceeded:")
            raise OptimizationTimeout("Timeout reached")