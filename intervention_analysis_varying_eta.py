import networkx as nx
import numpy as np
from tqdm import tqdm
import argparse
import os
from CTIC import run_ctic_simulations, num_spread_nodes_wo_intervention
from load_graph import Nikolov_susceptibility_graph, randomized_nikolov_graph
import util

def get_eta_lam(graph_name):
    if graph_name in ['nikolov', 'randomized_nikolov']:
        eta = 0.026
        lam = 0.25
    else:
        eta = 0.2
        lam = 1.0
    return eta, lam

def create_heatmap(graph, intervention_type, x_range, y_range, n_simulations=100, seed_base=42, target_selection='random'):
    print('create heatmap for (epsilon, eta)')
    print(f"intervention type: {intervention_type}")
    print(f"# of simulations: {n_simulations}")

    graph_name = graph.graph['graph_name']
    num_x = len(x_range)
    num_y = len(y_range)
    heatmap_data = np.zeros((num_x, num_y))
    
    _, lam = get_eta_lam(graph_name)
    seed_nodes = util.get_seed_users(graph, is_single_seed=True)

    # evaluate the prevalence for each (epsilon, eta) under the fixed delta_pre or intervention_threshold
    epsilon_pre = epsilon_ctx = epsilon_nud = 0.0
    delta_pre = 0.2
    intervention_threshold = 0.8

    pbar = tqdm(total=num_x * num_y, desc=f"{intervention_type.title()} Progress")
    for i, y in enumerate(y_range):
        eta = y
        expected_max_spread = num_spread_nodes_wo_intervention(graph=graph, seed_nodes=seed_nodes, eta=eta, lam=lam)
        for j, x in enumerate(x_range):
            if intervention_type == 'prebunking':
                epsilon_pre = x
            elif intervention_type == 'contextualization':
                epsilon_ctx = x
            elif intervention_type == 'nudging':
                epsilon_nud = x
            else:
                raise ValueError(f"Unknown intervention type: {intervention_type}")
            prevalences = run_ctic_simulations(num_simulations=n_simulations, graph=graph, seed_nodes=seed_nodes, eta=eta, lam=lam, max_time=1000.0, max_spread=expected_max_spread, epsilon_pre=epsilon_pre, epsilon_ctx=epsilon_ctx, epsilon_nud=epsilon_nud, delta_pre=delta_pre, intervention_threshold=intervention_threshold, target_selection=target_selection, seed=seed_base)
            heatmap_data[i, j] = np.mean(prevalences)
            pbar.update(1)
    pbar.close()
    
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    
    return x_grid, y_grid, heatmap_data

def save_heatmap_data(data_dict, graph_name, intervention_type, target_selection='random'):
    """ヒートマップデータを保存"""
    save_dir = os.path.join('results', graph_name)
    os.makedirs(save_dir, exist_ok=True)
    if intervention_type == 'prebunking':
        filename = f'{intervention_type}_{target_selection}'
    else:
        filename = f'{intervention_type}'
    filename = filename + '_vary_eta'
    filename = filename + '.npy'

    save_path = os.path.join(save_dir, filename)
    np.save(save_path, data_dict)
    print(f"heatmap saved: {save_path}")

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
        raise ValueError(f"Unknown graph name: {graph_name}. Available options: 'test', 'politifact', 'gossipcop', 'nikolov', 'randomized_nikolov'")

def run_intervention_analysis(intervention_type, graph_name='test', n_simulations=100, seed_base=42, save_data=False, target_selection='random'):
    print(f"=== {intervention_type.upper()} heatmap generation (CTIC) ===")
    print(f"graph: {graph_name}")
    
    G = load_graph_by_name(graph_name)
    x_range = np.arange(0.0, 1.05, 0.05)  # epsilon_pre or epsilon_ctx
    y_range = np.arange(0.0, 0.105, 0.005)  # eta

    x_grid, y_grid, heatmap_data = create_heatmap(G, intervention_type, x_range, y_range, n_simulations, seed_base, target_selection)
    
    if intervention_type == 'prebunking':
        labels = {
            'xlabel': r'$\varepsilon_{\mathrm{pre}}$', 
            'ylabel': r'$\eta$', 
            'title': f'prebunking',
        }
    elif intervention_type == 'contextualization':
        labels = {
            'xlabel': r'$\varepsilon_{\mathrm{ctx}}$', 
            'ylabel': r'$\eta$', 
            'title': f'contextualization',
        }
    elif intervention_type == 'nudging':
        labels = {
            'xlabel': r'$\varepsilon_{\mathrm{nud}}$', 
            'ylabel': r'$\eta$', 
            'title': f'nudging',
        }
    else:
        raise ValueError(f"Unknown intervention type: {intervention_type}")
        
    data_dict = {
        'x_grid': x_grid,
        'y_grid': y_grid,
        'heatmap_data': heatmap_data,
        'labels': labels,
        'x_range': x_range,
        'y_range': y_range,
        'intervention_type': intervention_type,
        'graph_name': graph_name,
        'n_simulations': n_simulations
    }
    
    if save_data:
        save_heatmap_data(data_dict, graph_name, intervention_type, target_selection)
    # util.plot_heatmap_data(data_dict)

def main():
    parser = argparse.ArgumentParser(description='Create heatmap under intervention')
    parser.add_argument('--intervention', '-i', type=str, required=True, choices=['prebunking', 'contextualization', 'nudging'], help='介入タイプを指定')
    parser.add_argument('--graph', '-g', type=str, default='test', choices=['test', 'nikolov', 'randomized_nikolov'], help='使用するグラフを指定 (デフォルト: test)')
    parser.add_argument('--simulations', '-s', type=int, default=100, help='シミュレーション回数 (デフォルト: 100)')
    parser.add_argument('--seed', type=int, default=42, help='乱数シード (デフォルト: 42)')
    parser.add_argument('--save_data', action='store_true', help='ヒートマップデータをnpyファイルとして保存')
    parser.add_argument('--target_selection', type=str, default='random', choices=['random', 'high_degree', 'high_susceptible', 'cocoon'], help='介入ターゲット選択戦略 (デフォルト: random)')
    args = parser.parse_args()
    
    run_intervention_analysis(
        intervention_type=args.intervention,
        graph_name=args.graph,
        n_simulations=args.simulations,
        seed_base=args.seed,
        save_data=args.save_data,
        target_selection=args.target_selection,
    )

if __name__ == "__main__":
    main()