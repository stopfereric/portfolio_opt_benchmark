# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:16:58 2024

@author: stopfer
"""
import pdb
import pandas as pd
import numpy as np
import csv
from yfinance import Ticker


def construct_stock_dataset(stock_symbols, start_date, end_date, stocksetname):
    ''' 
    function that creates a output stock dataset for markowitz portfolio 
    optimization.
    '''
    historical_prices = get_historical_prices(stock_symbols, start_date, end_date)
    
    daily_returns = calc_daily_returns(historical_prices)
    
    annual_returns = calc_annualized_returns(daily_returns)
    save_dataframe_to_csv(annual_returns, f"{stocksetname}_annual_returns.csv")
    
    annualized_covariance_matrix = calc_annualized_covariance_matrix(daily_returns)
    save_dataframe_to_csv(annualized_covariance_matrix, f"{stocksetname}_annualized_covariance_matrix.csv")
    

def get_historical_prices(stock_symbols: list, 
                         start_date: str,
                         end_date: str) -> pd.DataFrame:
    ''' 
    function that outputs historical prices of some stocks corresponding to 
    their symbols in the time range of a given start and end date
    '''
    historical_prices = pd.DataFrame()    
    for idx in range(len(stock_symbols)):
        try: 
            symbol = stock_symbols[idx]
            stock = Ticker(symbol)
            historical_data = stock.history(interval="1d", 
                                            start=start_date,
                                            end=end_date)['Close']
            stock_prices = historical_data.tolist()
            historical_prices[symbol] = stock_prices
            print(f"successful with {symbol}, at {round(100*idx/len(stock_symbols))}%")
        
        except:
            print(f"error with {symbol}, at {round(100*idx/len(stock_symbols))}%")
        
    
    historical_prices = historical_prices.reset_index(drop=True)        
    return historical_prices


def calc_daily_returns(historical_prices: pd.DataFrame) -> pd.DataFrame:
    ''' 
    function that calculated the daily returns of some historical prices dataframe
    '''
    daily_returns = pd.DataFrame() 
    for symbol in historical_prices.columns:
        daily_returns[symbol] = historical_prices[symbol].pct_change()
        
    daily_returns = daily_returns.drop(daily_returns.index[0]) # first line was filled with nan's
    daily_returns = daily_returns.reset_index(drop=True)
    print("successfully calculated the daily returns")
    return daily_returns


def calc_annualized_returns(daily_returns: pd.DataFrame) -> pd.DataFrame:
    '''
    function that annualizes the daily returns from a dataframe
    '''
    annual_returns = pd.DataFrame()
    annualization_factor = 252 / len(daily_returns)
    for symbol in daily_returns.columns:
        daily_return_product = np.prod(1 + daily_returns[symbol])
        annualized_return_product = daily_return_product ** annualization_factor
        annual_returns[symbol] = [annualized_return_product - 1]
    
    annual_returns = annual_returns.reset_index(drop=True)
    print("successfully calculated the annualized returns")
    return annual_returns


def calc_annualized_covariance_matrix(daily_returns: pd.DataFrame) -> pd.DataFrame:
    ''' 
    function that calculates the annualized covariance matrix for some
    daily returns of stocks
    '''
    mean_daily_price_changes = calc_mean_daily_price_changes(daily_returns)
    
    annualized_covariance_matrix = pd.DataFrame()
    for symbol1 in daily_returns.columns:
        for symbol2 in daily_returns.columns:
            annualized_covariance_matrix.at[symbol1, symbol2] = pd.NA
    print("successfully initialized the annualized covariance matrix")
    
    number_of_symbols = daily_returns.shape[1]
    for index in range(number_of_symbols):
        symbol1 = daily_returns.columns[index]
        for symbol2 in daily_returns.columns:
            if pd.isna(annualized_covariance_matrix.at[symbol1, symbol2]):
                annualization_factor = 252 / len(daily_returns)
                covariance_sum = float(np.sum((daily_returns[symbol1] - mean_daily_price_changes.at[0, symbol1]) * 
                                              (daily_returns[symbol2] - mean_daily_price_changes.at[0, symbol2])))
                cov = annualization_factor * covariance_sum
                annualized_covariance_matrix.at[symbol1, symbol2] = cov
                annualized_covariance_matrix.at[symbol2, symbol1] = cov
        print(f"successfully calculated the covariances for symbol {symbol1}, now at {round(index/number_of_symbols,2)}%")
                    
    return annualized_covariance_matrix


def calc_mean_daily_price_changes(daily_returns: pd.DataFrame) -> pd.DataFrame:
    ''' 
    function that calculates the mean price changes for some
    daily returns of stocks
    '''
    mean_daily_price_changes = pd.DataFrame()
    for symbol in daily_returns.columns:
        mean_change = 1 / len(daily_returns) * np.sum(daily_returns[symbol])
        mean_daily_price_changes[symbol] = [mean_change]
        
    mean_daily_price_changes = mean_daily_price_changes.reset_index(drop=True)
    return mean_daily_price_changes
    
    
# def calc_annualized_covariance_matrix2(daily_returns: pd.DataFrame) -> pd.DataFrame:
#     ''' 
#     function that calculates the annualized covariance matrix for some
#     daily returns of stocks
#     '''
#     annualization_factor = 252
#     return daily_returns.cov() * annualization_factor


def save_dataframe_to_csv(dataframe: pd.DataFrame, 
                           filename: str):
    ''' 
    function that saves a dataframe
    to csv-format to the same folder as this file
    '''
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        
        writer.writerow(dataframe.columns)
        
        for row in dataframe.itertuples(index=True, name=None):
            writer.writerow(row)
    return
    

# stock_symbols_df = pd.read_csv("DAX.txt", delimiter="\t", usecols=["Symbol"])
# stock_symbols_df = pd.read_csv("NYSE.txt", delimiter="\t", usecols=["Symbol"]) # data got from: 'https://www.eoddata.com/symbols.aspx'
stock_symbols_df = pd.read_csv("NASDAQ.txt", delimiter="\t", usecols=["Symbol"]) # data got from: 'https://www.eoddata.com/symbols.aspx'

stock_symbols_list = stock_symbols_df["Symbol"].tolist()
max_number_of_stocks = 10000
stock_symbols_list = stock_symbols_list[:min(len(stock_symbols_list), max_number_of_stocks)]

start_date = '2020-01-01'
end_date = '2023-12-31'

if __name__ == "__main__":
    # construct_stock_dataset(dax_stock_symbols, start_date, end_date, 'dax')
    
    construct_stock_dataset(stock_symbols_list,
                            start_date, 
                            end_date, 
                            'test')
