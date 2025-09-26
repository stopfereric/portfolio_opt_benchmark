# -*- coding: utf-8 -*-
"""
Created on 07.03.2024

@author: Eric Stopfer
"""
import logging
import os



def configure_logging(log_level_console, log_level_file, log_file_path):
    ''' 
    Configure a logger with a console and file handler. 
    Updates the file handler if the logger already exists.
    '''
    # Create or get the logger
    logger = logging.getLogger('quopt_logger')
    logger.setLevel(logging.DEBUG)   
    logger.propagate = False # Prevent propagation to the root logger
    
    # Check if logger has handlers
    if logger.hasHandlers():
        # Remove existing FileHandlers
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)

    # Add console handler if not already present
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level_console)
        console_handler_formatter = logging.Formatter('%(levelname)-8s:   %(message)s')
        console_handler.setFormatter(console_handler_formatter)
        logger.addHandler(console_handler)

    # Add or replace file handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(log_level_file)
    file_handler_formatter = logging.Formatter('%(asctime)s - %(levelname)-8s:   %(message)s')
    file_handler.setFormatter(file_handler_formatter)
    logger.addHandler(file_handler)
    
    logger.info("#####################################")
    logger.info("New optimization run starts right now")
    logger.info("#####################################")
    logger.info("A log-file was created for an optimization run\n\n")
    return logger
