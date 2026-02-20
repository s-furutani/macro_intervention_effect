import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from tqdm import tqdm
import random
import argparse
import sys
import os
from CTIC import run_ctic_simulations, num_spread_nodes_wo_intervention
from load_graph import Nikolov_susceptibility_graph, randomized_nikolov_graph
# from fit_parameters import get_best_parameters
import util
import time

def get_eta_lam(graph_name):
    if graph_name in ['nikolov', 'randomized_nikolov']:
        eta = 0.026
        lam = 0.25
    else:
        eta = 0.2
        lam = 1.0
    return eta, lam

def compare_single_and_combined_intervention(graph_name, n_simulations=100, target_selection='high_susceptible', eta_scale=1.0):
    """単一介入と複合介入の比較"""
    graph = load_graph_by_name(graph_name)
    seed_base = 42
    intervention_type_list = ['none', 'prebunking', 'contextualization', 'nudging', 'combined']

    eta, lam = get_eta_lam(graph_name)
    eta = eta * float(eta_scale)

    prevalences_list = []
    for intervention_type in intervention_type_list:
        print(f"intervention type: {intervention_type}")
        epsilon_pre = 0.0
        epsilon_ctx = 0.0
        epsilon_nud = 0.0
        delta_pre = 0.0
        intervention_threshold = 1.0

        improve_strength = 0.1  
        
        if intervention_type == 'none':
            pass
        elif intervention_type == 'prebunking':
            epsilon_pre = 0.204 + improve_strength
            delta_pre = 0.3
        elif intervention_type == 'contextualization':
            epsilon_ctx = 0.342 + improve_strength
            intervention_threshold = 0.7
        elif intervention_type == 'nudging':
            epsilon_nud = 0.143 + improve_strength
        elif intervention_type == 'combined':
            epsilon_pre = 0.204 + improve_strength
            delta_pre = 0.3
            epsilon_ctx = 0.342 + improve_strength
            epsilon_nud = 0.143 + improve_strength
            intervention_threshold = 0.7
        
        seed_nodes = util.get_seed_users(graph, is_single_seed=True)
        if intervention_type in ['contextualization', 'combined']:
            expected_max_spread = num_spread_nodes_wo_intervention(graph=graph, seed_nodes=seed_nodes, eta=eta, lam=lam)
        else:
            expected_max_spread = 10000000.0
        prevalences = run_ctic_simulations(n_simulations, graph, seed_nodes, eta, lam, 1000.0, expected_max_spread, epsilon_pre, epsilon_ctx, epsilon_nud, delta_pre, intervention_threshold, target_selection, seed_base)
        prevalences_list.append(prevalences)
    
    prevalences_list = np.array(prevalences_list)
    directory = f'results/{graph_name}'
    os.makedirs(directory, exist_ok=True)
    output_path = f'{directory}/compare_single_and_combined_intervention_{target_selection}_improve_both.npy'
    if eta_scale == 2.0:
        output_path = output_path.replace('.npy', f'_double.npy')
    elif eta_scale == 0.5:
        output_path = output_path.replace('.npy', f'_half.npy')
    np.save(output_path, prevalences_list)
    return prevalences_list

def load_graph_by_name(graph_name):
    if graph_name == 'test':
        G = nx.erdos_renyi_graph(200, 0.1, seed=42)
        G = nx.to_directed(G)
        G.graph['graph_name'] = 'test'
        for node in G.nodes():
            G.nodes[node]['suscep'] = np.random.uniform(0.0, 1.0)
        return G
    elif graph_name == 'nikolov':
        G = Nikolov_susceptibility_graph()
        G.graph['graph_name'] = 'nikolov'
        return G
    elif graph_name == 'randomized_nikolov':
        G = randomized_nikolov_graph()
        G.graph['graph_name'] = 'randomized_nikolov'
        return G
    else:
        raise ValueError(f"Unknown graph name: {graph_name}. Available options: 'test', 'nikolov', 'randomized_nikolov'")


if __name__ == "__main__":
    start_time = time.time()
    graph_name = 'nikolov'
    n_simulations = 10
    target_selection = 'high_degree'
    print(f"target selection: {target_selection}")
    prevalences_list = compare_single_and_combined_intervention(graph_name, n_simulations=n_simulations, target_selection=target_selection)
    # util.violin_plot(prevalences_list, graph_name)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")