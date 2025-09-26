# -*- coding: utf-8 -*-
"""
Created on 07.03.2024

@author: Eric Stopfer
"""

class CustomizedError(Exception):
    '''
    this class can be used to raise customized errors for fails that occur during the execution of the code
    '''
    def __init__(self, error_message, additional_info=None):
        super().__init__(error_message)
        self.additional_info = additional_info
        return