# -*- coding: utf-8 -*-
"""
Created on 24.05.2024

@author: Eric Stopfer
"""
import pdb
import os
import numpy as np
import math
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, ScalarFormatter
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
import networkx as nx



    
def create_barplot_for_obj_values(obj_values_counts_list, store_dir):
    '''
    function to create a barplot of a list of objective values and saves the
    resulting graphic to the store_dir
    '''
    obj_values, counts = zip(*obj_values_counts_list)
    plt.bar(obj_values, counts)
    plt.xlabel('obj_value')
    plt.ylabel('counts')
    plt.title('barplot distribution of best objective values gernerated during QuantumAnnealing')
    plt.legend()
    plt.savefig(os.path.join(store_dir, "barplot_obj_values.png"), dpi=200)
    plt.clf()
    plt.close()


def create_barplot_obj_values_with_number_of_violated_constraints(result_dict: dict, store_dir: str, filename: str, colors=['blue', 'orange']):
    '''
    function that creates and saves a barplot for solver sampling results where: 
        -x-axis: obj values 
        -y-axis: count of obj values 
        -different bar colours for different number of violated constraints
    '''
    assert len(colors) == 2, "only 2 colors allowed"
    #check if necessary attributes are there
    required_keys_of_bitstr = ['qubo_obj_value', 'count', 'violated_constraints']
    for bitstr, bitstr_dict in result_dict.items():
        assert all(key in bitstr_dict for key in required_keys_of_bitstr), "in the result dict not every bitstring has the following attributes: -- qubo_obj_value, num_of_occurences, violated_constraints --"
                
    # group data by objective values
    grouped_data = defaultdict(list)
    for bitstr, bitstr_dict in result_dict.items():
        grouped_data[bitstr_dict['qubo_obj_value']].append(bitstr_dict)
    
    # define colour depending on the violated constraints
    max_constraints = max(len(bitstr_dict['violated_constraints']) for bitstr_dict in result_dict.values())
    # colors = plt.cm.viridis(np.linspace(0, 1, max_constraints + 1))
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
    colors = custom_cmap(np.linspace(0, 1, max_constraints + 1))
    
    # prepare heights and colours of bars
    bars_dict = defaultdict(lambda: np.zeros(len(grouped_data)))
    obj_value_positions = sorted(grouped_data.keys())
    
    for i, obj_value in enumerate(obj_value_positions):
        for value in grouped_data[obj_value]:
            constraints_length = len(value['violated_constraints'])
            bars_dict[constraints_length][i] += value['count']
    
    # create the barplot
    plt.figure(figsize=(10, 6))
    # plot the bars
    bottom = np.zeros(len(obj_value_positions))
    bar_width = 0.5
    for constraints_length, heights in sorted(bars_dict.items()):
        plt.bar(obj_value_positions, heights, width=bar_width, bottom=bottom, color=colors[constraints_length], label=f'{constraints_length} constraints violated')
        bottom += heights
    
    # axis labels and title
    plt.xlabel('objective values')
    plt.ylabel('count')
    plt.title('counts of QUBO-objective-values')
    
    # add colorbar
    # sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=max_constraints))
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=0, vmax=max_constraints))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('number of violated constraints')
    plt.savefig(os.path.join(store_dir, filename + ".png"), dpi=200)
    plt.clf()
    plt.close()


def visualize_qaoa_parameter_optimization_process(energy_expectations: list, store_dir: str, filename: str):
    '''
    Function to draw the parameter optimization process during the QAOA.
    It plots the expected ising energies per optimization cycle
    '''
    cycles = np.arange(1, len(energy_expectations)+1)
    # generate values for the plot
    plt.plot(cycles, energy_expectations)
    # create and save the plot
    plt.xlabel('optimization cycle')
    plt.ylabel('energy expectation value for each optimization cycle')
    plt.savefig(os.path.join(store_dir, filename + ".png"), dpi=200)
    plt.clf()
    plt.close()
    

def visualize_qaoa_distribution_of_energies(sampling_energies: dict, 
                                            store_dir: str, 
                                            filename: str):
    '''
    Function to draw a probability density of energy lists and saves the resulting graphic.
    This has the goal to get a better understanding of the energy distribution.
    It is possibly helpful to benchmark QAOA results of 
    classical_param_opt vs grid_search_param_opt vs random sampling.
    '''
    pdfs = [] # initialize probability density functions
    energy_lists = [] 
    for sampling_method, energy_list in sampling_energies.items():
        try:
            pdfs.append((sampling_method, gaussian_kde(energy_list))) # pdf = probability density function
            energy_lists.append(energy_list)
        except:
            pass
    
    min_value = min(min(sublist) for sublist in energy_lists)
    max_value = max(max(sublist) for sublist in energy_lists)
    support = np.linspace(start=min_value, stop=max_value)
    
    for (sampling_method, pdf) in pdfs:
        y_on_support = pdf(support)
        plt.plot(support, y_on_support, label=sampling_method)
    
    plt.xlabel('energy')
    plt.ylabel('probability')
    plt.title('probability density of energies')
    plt.legend()
    plt.savefig(os.path.join(store_dir, filename + ".png"), dpi=200)
    plt.clf()
    plt.close()
   
    
def draw_graph_with_edge_weights(nodes: list, 
                                 edges: list,
                                 distance_matrix: list, 
                                 store_dir: str, 
                                 filename: str
                                 ) -> nx.Graph:
    ''' 
    function that draws a networkx-graph with edge_weights. 
    in the end the graphic is saved to a storage-directory
    '''
    graph = create_graph_with_edge_weights(nodes, edges, distance_matrix)
    
    pos = nx.spring_layout(graph)
    plt.figure()
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    
    plt.savefig(os.path.join(store_dir, filename + ".png"), dpi=200)
    plt.clf()
    plt.close()
    return graph


def create_graph_with_edge_weights(nodes: list, 
                                 edges: list,
                                 distance_matrix: list
                                 ) -> nx.Graph:
    ''' 
    function that creates a networkx-graph with edge_weights.
    '''    
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    for [u,v] in edges:
        weight = distance_matrix[u][v]
        graph.add_edge(u, v, weight=weight)
    return graph


def draw_return_vs_volatility(result: dict, 
                              store_dir: str,
                              filename: str):
    ''' 
    function to draw the results of a Markowitz portfolio optimization.
    it plots points in a 2d-system of return versus volatility for the found solutions.
    '''
    if len(result.keys()) != 0:
        if not os.path.exists(store_dir):
            os.makedirs(store_dir, exist_ok=True)
            
        all_exp_returns = []
        all_exp_volatilities = []
        for var_values, var_values_dict in result.items():
            all_exp_returns.append(var_values_dict['expected_return'])
            all_exp_volatilities.append(var_values_dict['expected_volatility'])
            
        plt.figure(figsize=(10, 6))
        plt.grid(True)
        plt.xlabel('volatility')
        plt.ylabel('return')
        plt.scatter(all_exp_volatilities, all_exp_returns, color='blue', marker='o')
        plt.savefig(os.path.join(store_dir, filename + ".png"))            
        plt.clf()
        plt.close()
               
    
def visualize_metric_forproblemsize_foroptmethod(
        problem_formulation_metric_dict: dict, 
        problem_type: str,
        max_problem_size: int,
        min_problem_size: int,
        metric_name: str,
        export_path: str,
        with_errorbars: bool=True,
        logarithmic_scale: bool=False,
        percentage_scale: bool=False,
        colors: list = ["blue", "green", "orange", "violet", "red", "black", "grey", "darkviolet", "maroon", "olive", "pink", "cyan", "magenta", "gold", "teal", "navy", "turquoise", "coral", "lime", "indigo", "brown", "salmon", "orchid"],
        markers: list = ['o', '+', 'x', '1', '2', '3', '4', '|', '_', 'v', '*', 'p', 'h', 's', 'D']
        ):
    '''
    function that draws a graph to visualize values of a metrics for different
    problem sizes and different opt_methods. 
    x-axis: problem size and y-axis: metric value
    '''
    # plt.figure(figsize=(10, 12))
    plt.figure(figsize=(10, 12*0.67816091954))
    graphic_has_at_least_one_plot = False
    color_idx = 0
    more_than_one_formulation = True if len(problem_formulation_metric_dict.keys()) > 1 else False
    xticks_set = set()
    yticks_set = set()
    for problem_formulation, metric_dict in problem_formulation_metric_dict.items():
        marker_idx = 0  
        for solve_method, values in metric_dict['average'].items():
            if values != {}:
                problem_sizes = sorted([i for i in values.keys() if i <= max_problem_size and i >= min_problem_size])
                xticks_set = xticks_set | set(problem_sizes)
                metric_values_avg = [values[i] for i in problem_sizes]
                yticks_set = yticks_set | set(metric_values_avg)
                label = f"{problem_formulation} - {solve_method}" if more_than_one_formulation else f"{solve_method}"
                plt.plot(problem_sizes, 
                         metric_values_avg, 
                         color=colors[color_idx], 
                         label=label, 
                         linestyle='--',
                         linewidth=2,
                         marker=markers[marker_idx], 
                         markersize=12)
                if more_than_one_formulation:
                    marker_idx += 1
                else:
                    color_idx += 1
                graphic_has_at_least_one_plot = True   
        if more_than_one_formulation:
            color_idx += 1
            
    if with_errorbars:
        color_idx = 0
        for problem_formulation, metric_dict in problem_formulation_metric_dict.items():
            for solve_method, values in metric_dict['standard deviation'].items():
                if values != {}:
                    problem_sizes = sorted([i for i in values.keys() if i <= max_problem_size and i >= min_problem_size])
                    metric_values_avg = [metric_dict['average'][solve_method][i] for i in problem_sizes]
                    metric_values_stddev = [values[i] for i in problem_sizes]
                    plt.errorbar(problem_sizes, 
                                 metric_values_avg, 
                                 yerr=metric_values_stddev, 
                                 fmt='none', 
                                 color=colors[color_idx], 
                                 capsize=5)
                    color_idx += 1
            
    
    if graphic_has_at_least_one_plot:
        if logarithmic_scale:
            plt.yscale('log')
        if percentage_scale:
            plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            plt.ylim(bottom=0)            
        xlabel = "number of assets" if problem_type == "MarkowitzPortfolio" \
            else "number of nodes" if problem_type in ['TSP', 'CVRP'] \
            else "number of objects" if problem_type in ['BinPacking'] \
            else 'unknown'
        plt.xlabel(xlabel, fontsize=28)
        plt.ylabel(metric_name, fontsize=28)
        if len(xticks_set) == 0 or max(xticks_set) >= 40:
            plt.xticks(fontsize=28)
        else:
            plt.xticks(sorted(xticks_set), fontsize=28)
        if len(yticks_set) == 0 or max(yticks_set) >= 10:
            plt.yticks(fontsize=28)
        else:
            formatter = ScalarFormatter()
            formatter.set_scientific(False)
            formatter.set_useOffset(False)
            plt.gca().yaxis.set_major_formatter(formatter)
            plt.yticks(list(range(int(min(yticks_set)), math.ceil(max(yticks_set)) + 1)), fontsize=28)
        # plt.title(metric_name)
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize=28)
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(os.path.join(export_path, f"metric analysis {problem_type} - {metric_name}.png"), dpi=200) 
        plt.clf()
        plt.close()
        
        
def visualize_dict(
        data: dict, 
        problem_type: str,
        max_problem_size: int,
        min_problem_size: int,
        metric_name: str,
        export_path: str,
        logarithmic_scale: bool = False,
        percentage_scale: bool = False,
        colors: list = ["blue", "green", "orange", "violet", "red", "black", "grey", "darkviolet", "maroon", "olive", "pink", "cyan", "magenta", "gold", "teal", "navy", "turquoise", "coral", "lime", "indigo", "brown", "salmon", "orchid"],
        markers: list = ['+', 'x', '1', '2', '3', '4', '|', '_', 'o', 'v', '*', 'p', 'h', 's', 'D']
        ):
    ''' 
    visualizes the data inside of a dictionary with matplotlib
    for different problem sizes and different opt_methods. 
    x-axis: problem size and y-axis: metric value
    '''
    # plt.figure(figsize=(10, 12))
    plt.figure(figsize=(10, 12*0.67816091954))
    xticks_set = set()
    idx = 0
    for label, inner_dict in data.items():
        if not inner_dict:
            continue
        x = [i for i in sorted(inner_dict.keys()) if i <= max_problem_size and i >= min_problem_size]
        xticks_set = xticks_set | set(x)
        y = [inner_dict[i]+0.0025*idx for i in x]
        plt.plot(x, 
                 y, 
                 label=label, 
                 color=colors[idx % len(colors)],
                 linestyle='--',
                 linewidth=2,
                 marker='o',#markers[idx % len(markers)], 
                 markersize=12,
                 alpha=0.7)
        idx += 1
    
    xlabel = "number of assets" if problem_type == "MarkowitzPortfolio" \
        else "number of nodes" if problem_type in ['TSP', 'CVRP'] \
        else "number of objects" if problem_type in ['BinPacking'] \
        else 'unknown'
    if logarithmic_scale:
        plt.yscale('log')
    if percentage_scale:
        plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        plt.ylim(bottom=-0.05)
    plt.xlabel(xlabel, fontsize=28)
    plt.ylabel(metric_name, fontsize=28)
    if len(xticks_set) == 0 or max(xticks_set) >= 40:
        plt.xticks(fontsize=28)
    else:
        plt.xticks(sorted(xticks_set), fontsize=28)
    plt.yticks(fontsize=28)
    # plt.title(metric_name)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize=28)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.65), ncol=1, fontsize=28)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(os.path.join(export_path, f"metric analysis {problem_type} - {metric_name}.png"), dpi=200) 
    plt.clf()
    plt.close()