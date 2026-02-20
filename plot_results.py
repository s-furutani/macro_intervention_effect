# %%
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from load_graph import Nikolov_susceptibility_graph, randomized_nikolov_graph
import util
import load_graph
import importlib
importlib.reload(util)
# importlib.reload(load_graph)


# %%
graph_name = 'nikolov'
suffix_list = ['', '_double', '_half']
for suffix in suffix_list:
    print(suffix)
    prevalences_list = np.load(f'results/{graph_name}/ctic/compare_single_and_combined_intervention_high_degree{suffix}.npy')
    util.violin_plot(prevalences_list, graph_name, relative_suppression=False)# %%

# %%
graph_name = 'nikolov'
target_selection_list = ['high_degree', 'high_susceptible', 'random']
for target_selection in target_selection_list:
    print(target_selection)
    prevalences_list = np.load(f'results/{graph_name}/ctic/compare_single_and_combined_intervention_{target_selection}.npy')
    util.violin_plot(prevalences_list, graph_name)
# %%
# misinformation prevalence with prebunking and contextualization under fixed eta
graph_name = 'nikolov'
print(graph_name)
path1 = f'results/{graph_name}/ctic/prebunking_random.npy'
path2 = f'results/{graph_name}/ctic/contextualization.npy'
path_list = [path1, path2] * 2
relative_flag_list = [False, False, True, True]
title_list = ['Prebunking', 'Contextualization', None, None]
util.plot_multiple_heatmaps(2, 2, path_list, relative_flag_list, title_list)

# %%
# misinformation prevalence with prebunki|ng, contextualization, and nudging under varying eta
graph_name = 'nikolov'
path_list = [f'results/{graph_name}/ctic/nudging_vary_eta.npy', f'results/{graph_name}/ctic/prebunking_random_vary_eta.npy', f'results/{graph_name}/ctic/contextualization_vary_eta.npy'] * 2 
relative_flag_list = [False, False, False, True, True, True]
title_list = ['Nudging', 'Prebunking', 'Contextualization', None, None, None]
util.plot_multiple_heatmaps(2, 3, path_list, relative_flag_list, title_list, plot_critical_curve=True)
# %%
# difference between random and targeted prebunking for each (epsilon_pre, delta_pre)
graph_name = 'nikolov'
print(graph_name)
path1 = f'results/{graph_name}/ctic/prebunking_random.npy'
path2 = f'results/{graph_name}/ctic/prebunking_high_degree.npy'
path3 = f'results/{graph_name}/ctic/prebunking_high_susceptible.npy'
path_pair1 = (path1, path2)
path_pair2 = (path1, path3)
path_list = [path1, path2, path3, path1, path2, path3]
relative_flag_list = [False, False, False, True, True, True]
title_list = ['Random', 'Degree-based', 'Susceptiblity-based', '', '', '']
util.plot_multiple_heatmaps(2, 3, path_list, relative_flag_list, title_list)

path_list = [path_pair1, path_pair2]
relative_flag_list = [False, False]
title_list = ['Random - Degree', 'Random - Susceptiblity']
util.plot_multiple_heatmaps(1, 2, path_list, relative_flag_list, title_list)
# %%
# difference between random and targeted prebunking for each (epsilon_pre, eta)
graph_name = 'nikolov'
print(graph_name)
path1 = f'results/{graph_name}/ctic/prebunking_random_vary_eta.npy'
path2 = f'results/{graph_name}/ctic/prebunking_high_degree_vary_eta.npy'
path3 = f'results/{graph_name}/ctic/prebunking_high_susceptible_vary_eta.npy'
path_pair1 = (path1, path2)
path_pair2 = (path1, path3)

path_list = [path1, path2, path3, path1, path2, path3]
relative_flag_list = [False, False, False, True, True, True]
title_list = ['Random', 'Degree-based', 'Susceptiblity-based', '', '', '']
util.plot_multiple_heatmaps(2, 3, path_list, relative_flag_list, title_list)


path_list = [path_pair1, path_pair2]
relative_flag_list = [False, False]
title_list = ['Random - Degree', 'Random - Susceptiblity']
util.plot_multiple_heatmaps(1, 2, path_list, relative_flag_list, title_list)
########################################################
#             Randomized Nikolov Graph
########################################################
# %%
# difference between nikolov and randomized_nikolov for (epsilon_pre, delta_pre)

graph_name = 'randomized_nikolov'
path_list = [f'results/{graph_name}/ctic/prebunking_random_vary_eta.npy', f'results/{graph_name}/ctic/contextualization_vary_eta.npy', f'results/{graph_name}/ctic/nudging_vary_eta.npy'] * 2 
relative_flag_list = [False, False, False, True, True, True]
title_list = ['Prebunking', 'Contextualization', 'Nudging', None, None, None]
util.plot_multiple_heatmaps(2, 3, path_list, relative_flag_list, title_list)

graph_name = 'nikolov'
graph_name_r = 'randomized_nikolov'
path1 = f'results/{graph_name}/ctic/prebunking_random.npy'
path1_r = f'results/{graph_name_r}/ctic/prebunking_random.npy'
path2 = f'results/{graph_name}/ctic/contextualization.npy'
path2_r = f'results/{graph_name_r}/ctic/contextualization.npy'
path_pair1 = (path1, path1_r)
path_pair2 = (path2, path2_r)
path_list = [path_pair1, path_pair2]
relative_flag_list = [False, False]
title_list = ['Prebunking', 'Contextualization']
util.plot_multiple_heatmaps(1, 2, path_list, relative_flag_list, title_list, diff_vlim=0.1)

# %%
# difference between nikolov and randomized_nikolov for (epsilon_pre, eta)
graph_name = 'nikolov'
graph_name_r = 'randomized_nikolov'
path1 = f'results/{graph_name}/ctic/prebunking_random_vary_eta.npy'
path1_r = f'results/{graph_name_r}/ctic/prebunking_random_vary_eta.npy'
path2 = f'results/{graph_name}/ctic/contextualization_vary_eta.npy'
path2_r = f'results/{graph_name_r}/ctic/contextualization_vary_eta.npy'
path3 = f'results/{graph_name}/ctic/nudging_vary_eta.npy'
path3_r = f'results/{graph_name_r}/ctic/nudging_vary_eta.npy'

path_list = [path1_r, path2_r, path3_r] * 2
relative_flag_list = [False, False, False, True, True, True]
title_list = ['Prebunking', 'Contextualization', 'Nudging', None, None, None]
util.plot_multiple_heatmaps(2, 3, path_list, relative_flag_list, title_list)

path_pair1 = (path1, path1_r)
path_pair2 = (path2, path2_r)
path_pair3 = (path3, path3_r)
path_list = [path_pair1, path_pair2, path_pair3]
relative_flag_list = [False, False, False]
title_list = ['Prebunking', 'Contextualization', 'Nudging']
util.plot_multiple_heatmaps(1, 3, path_list, relative_flag_list, title_list, diff_vlim=0.2)

# %%
def plot_heatmap_relative_diff(row, col, path_list, title_list, vlim=0.5):
    plt.figure(figsize=(12, 3.5), dpi=100)
    for i, path in enumerate(path_list):
        path1, path2 = path
        data_dict1 = np.load(path1, allow_pickle=True).item()
        data_dict1 =util.label_correction(data_dict1)
        data_dict2 = np.load(path2, allow_pickle=True).item()
        data_dict2 = util.label_correction(data_dict2)
        data1 = data_dict1['heatmap_data']
        data2 = data_dict2['heatmap_data']
        data = util.relative_suppression(data1) - util.relative_suppression(data2)
        plt.subplot(row, col, i+1)
        plt.imshow(data, cmap='bwr', aspect='auto', extent=[0, 1, 0, 1], origin='lower', vmin=-vlim, vmax=vlim)
        plt.colorbar()
        plt.xlabel(data_dict1['labels']['xlabel'], fontsize=22)
        plt.ylabel(data_dict1['labels']['ylabel'], fontsize=22)
        plt.title(title_list[i], fontsize=22)
        plt.tight_layout()

graph_name = 'nikolov'
graph_name_r = 'randomized_nikolov'
path1 = f'results/{graph_name}/ctic/prebunking_random_vary_eta.npy'
path1_r = f'results/{graph_name_r}/ctic/prebunking_random_vary_eta.npy'
path2 = f'results/{graph_name}/ctic/contextualization_vary_eta.npy'
path2_r = f'results/{graph_name_r}/ctic/contextualization_vary_eta.npy'
path3 = f'results/{graph_name}/ctic/nudging_vary_eta.npy'
path3_r = f'results/{graph_name_r}/ctic/nudging_vary_eta.npy'
path_list = [(path1, path1_r), (path2, path2_r), (path3, path3_r)]
title_list = ['Prebunking', 'Contextualization', 'Nudging']
plot_heatmap_relative_diff(1, 3, path_list, title_list)

path1 = f'results/{graph_name}/ctic/prebunking_random.npy'
path1_r = f'results/{graph_name_r}/ctic/prebunking_random.npy'
path2 = f'results/{graph_name}/ctic/contextualization.npy'
path2_r = f'results/{graph_name_r}/ctic/contextualization.npy'
path_list = [(path1, path1_r), (path2, path2_r)]
title_list = ['Prebunking', 'Contextualization']
plot_heatmap_relative_diff(1, 2, path_list, title_list, vlim=0.2)


path1 = f'results/{graph_name}/ctic/prebunking_random.npy'
path1_r = f'results/{graph_name_r}/ctic/prebunking_random.npy'
path2 = f'results/{graph_name}/ctic/prebunking_high_degree.npy'
path2_r = f'results/{graph_name_r}/ctic/prebunking_high_degree.npy'
path3 = f'results/{graph_name}/ctic/prebunking_high_susceptible.npy'
path3_r = f'results/{graph_name_r}/ctic/prebunking_high_susceptible.npy'
path_list = [(path1, path1_r), (path2, path2_r), (path3, path3_r)]
title_list = ['Random', 'Degree-based', 'Susceptiblity-based']
plot_heatmap_relative_diff(1, 3, path_list, title_list, vlim=0.3)




# %%
# difference between nikolov and randomized_nikolov for (epsilon_pre, eta)
graph_name = 'nikolov'
graph_name_r = 'randomized_nikolov'
path1 = f'results/{graph_name}/ctic/prebunking_random.npy'
path1_r = f'results/{graph_name_r}/ctic/prebunking_random.npy'
path2 = f'results/{graph_name}/ctic/prebunking_high_degree.npy'
path2_r = f'results/{graph_name_r}/ctic/prebunking_high_degree.npy'
path3 = f'results/{graph_name}/ctic/prebunking_high_susceptible.npy'
path3_r = f'results/{graph_name_r}/ctic/prebunking_high_susceptible.npy'

path_list = [path1_r, path2_r, path3_r] * 2
relative_flag_list = [False, False, False, True, True, True]
title_list = ['Random', 'Degree-based', 'Susceptiblity-based', '', '', '']
util.plot_multiple_heatmaps(2, 3, path_list, relative_flag_list, title_list)

path_pair1 = (path1, path1_r)
path_pair2 = (path2, path2_r)
path_pair3 = (path3, path3_r)
path_list = [path_pair1, path_pair2, path_pair3]
relative_flag_list = [False, False, False]
title_list = ['Random', 'Degree-based', 'Susceptiblity-based']
util.plot_multiple_heatmaps(1, 3, path_list, relative_flag_list, title_list, diff_vlim=0.2)


# %%
# difference between random and targeted prebunking for each (epsilon_pre, eta)
graph_name = 'randomized_nikolov'
print(graph_name)
path1 = f'results/{graph_name}/ctic/prebunking_random_vary_eta.npy'
path2 = f'results/{graph_name}/ctic/prebunking_high_degree_vary_eta.npy'
path3 = f'results/{graph_name}/ctic/prebunking_high_susceptible_vary_eta.npy'
path_pair1 = (path1, path2)
path_pair2 = (path1, path3)

path_list = [path1, path2, path3, path1, path2, path3]
relative_flag_list = [False, False, False, True, True, True]
title_list = ['Random', 'Degree-based', 'Susceptiblity-based', '', '', '']
util.plot_multiple_heatmaps(2, 3, path_list, relative_flag_list, title_list)


path_list = [path_pair1, path_pair2]
relative_flag_list = [False, False]
title_list = ['degree-based', 'susceptiblity-based']
util.plot_multiple_heatmaps(1, 2, path_list, relative_flag_list, title_list)

graph_name = 'randomized_nikolov'
print(graph_name)
path1 = f'results/{graph_name}/ctic/prebunking_random.npy'
path2 = f'results/{graph_name}/ctic/prebunking_high_degree.npy'
path3 = f'results/{graph_name}/ctic/prebunking_high_susceptible.npy'
path_pair1 = (path1, path2)
path_pair2 = (path1, path3)

path_list = [path1, path2, path3, path1, path2, path3]
relative_flag_list = [False, False, False, True, True, True]
title_list = ['Random', 'Degree-based', 'Susceptiblity-based', '', '', '']
util.plot_multiple_heatmaps(2, 3, path_list, relative_flag_list, title_list)


path_list = [path_pair1, path_pair2]
relative_flag_list = [False, False]
title_list = ['degree-based', 'susceptiblity-based']
util.plot_multiple_heatmaps(1, 2, path_list, relative_flag_list, title_list)

# %%


graph_name = 'nikolov'
graph_name_r = 'randomized_nikolov'
path1 = f'results/{graph_name}/ctic/prebunking_random_vary_eta.npy'
path2 = f'results/{graph_name}/ctic/prebunking_high_degree_vary_eta.npy'
path3 = f'results/{graph_name}/ctic/prebunking_high_susceptible_vary_eta.npy'
path1_r = f'results/{graph_name_r}/ctic/prebunking_random_vary_eta.npy'
path2_r = f'results/{graph_name_r}/ctic/prebunking_high_degree_vary_eta.npy'
path3_r = f'results/{graph_name_r}/ctic/prebunking_high_susceptible_vary_eta.npy'

data_dict1 = np.load(path1, allow_pickle=True).item()
data_dict2 = np.load(path2, allow_pickle=True).item()
data_dict3 = np.load(path3, allow_pickle=True).item()
data12 = data_dict1['heatmap_data'] - data_dict2['heatmap_data']
data13 = data_dict1['heatmap_data'] - data_dict3['heatmap_data']
data_dict1_r = np.load(path1_r, allow_pickle=True).item()
data_dict2_r = np.load(path2_r, allow_pickle=True).item()
data_dict3_r = np.load(path3_r, allow_pickle=True).item()
data12_r = data_dict1_r['heatmap_data'] - data_dict2_r['heatmap_data']
data13_r = data_dict1_r['heatmap_data'] - data_dict3_r['heatmap_data']

diff_data12 = data12 - data12_r
diff_data13 = data13 - data13_r
extent = [0, 1, 0, 0.2]
yticks = [0.0, 0.05, 0.1, 0.15, 0.2]
cmap = plt.get_cmap('PuOr')
vlim = 0.1

plt.figure(figsize=(9, 4), dpi=100)
plt.subplot(1, 2, 1)
plt.imshow(diff_data12, cmap=cmap, aspect='auto', extent=extent, origin='lower', vmin=-vlim, vmax=vlim)
plt.colorbar(ticks=[-0.1, -0.05, 0, 0.05, 0.1])
plt.title('degree-based', fontsize=20)
plt.xlabel(data_dict1['labels']['xlabel'], fontsize=22)
plt.ylabel(data_dict1['labels']['ylabel'], fontsize=22)
plt.yticks(yticks, [f"{y:.2f}" for y in yticks])
plt.subplot(1, 2, 2)
plt.imshow(diff_data13, cmap=cmap, aspect='auto', extent=extent, origin='lower', vmin=-vlim, vmax=vlim)
plt.colorbar(ticks=[-0.1, -0.05, 0, 0.05, 0.1])
plt.title('susceptiblity-based', fontsize=20)
plt.xlabel(data_dict1['labels']['xlabel'], fontsize=22)
plt.ylabel(data_dict1['labels']['ylabel'], fontsize=22)
plt.yticks(yticks, [f"{y:.2f}" for y in yticks])
plt.tight_layout()
plt.show()



########################################################
# %%

def relative_suppression(heatmap_data):
    data_rel = np.array([heatmap_data[i, :]/heatmap_data[i, 0] for i in range(heatmap_data.shape[0])]) 
    return data_rel

def plot_lines_side_by_side(fixed_params, relative=True):
    # --- Define data sources ---
    graph_name = 'nikolov'
    nudging_path = f"results/{graph_name}/ctic/nudging_vary_eta.npy"
    prebunking_path = f"results/{graph_name}/ctic/prebunking_random_vary_eta.npy"
    contextualization_path = f"results/{graph_name}/ctic/contextualization_vary_eta.npy"

    # --- Load data ---
    data_dict_nud = np.load(nudging_path, allow_pickle=True).item()
    data_dict_pre = np.load(prebunking_path, allow_pickle=True).item()
    data_dict_ctx = np.load(contextualization_path, allow_pickle=True).item()

    x_pre = data_dict_pre['x_range']
    y_pre = data_dict_pre['y_range']
    data_pre = data_dict_pre['heatmap_data']
    x_ctx = data_dict_ctx['x_range']
    y_ctx = data_dict_ctx['y_range']
    data_ctx = data_dict_ctx['heatmap_data']
    x_nud = data_dict_nud['x_range']
    y_nud = data_dict_nud['y_range']
    data_nud = data_dict_nud['heatmap_data']

    # --- Relative suppression ---
    if relative:
        data_pre_rel = relative_suppression(data_pre)
        data_ctx_rel = relative_suppression(data_ctx)
        data_nud_rel = relative_suppression(data_nud)
    else:
        data_pre_rel = data_pre
        data_ctx_rel = data_ctx
        data_nud_rel = data_nud

    # --- Colors ---
    cmap_b = plt.get_cmap('Blues')
    cmap_r = plt.get_cmap('Reds')
    cmap_g = plt.get_cmap('Greens')
    n_colors = len(fixed_params) + 1
    colors_nud = cmap_b(np.linspace(0, 1, n_colors))
    colors_pre = cmap_r(np.linspace(0, 1, n_colors))
    colors_ctx = cmap_g(np.linspace(0, 1, n_colors))
    colors_nud = list(colors_nud[3:8]) + list(colors_pre[3:])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=200, sharey=True)
    
    for i, val in enumerate(fixed_params):
        idx = np.where(np.isclose(y_pre, val))[0]
        if len(idx) == 0:
            continue
        idx = idx[0]
        result_pre = data_pre_rel[idx, :]
        result_ctx = data_ctx_rel[idx, :]
        result_nud = data_nud_rel[idx, :]
        label = rf"$\eta={val:.2f}$"
        
        axes[0].plot(x_nud, result_nud, marker='o', markerfacecolor='none', label=label, color=colors_nud[i+1])
        axes[1].plot(x_pre, result_pre, marker='o', markerfacecolor='none', label=label, color=colors_nud[i+1])
        axes[2].plot(x_ctx, result_ctx, marker='o', markerfacecolor='none', label=label, color=colors_nud[i+1])
    
    # 各サブプロットの設定
    titles = [r'Nudging', r'Prebunking', r'Contextualization']
    xlabels = [r'$\varepsilon_{\mathrm{nud}}$', r'$\varepsilon_{\mathrm{pre}}$', r'$\varepsilon_{\mathrm{ctx}}$']
    for i, ax in enumerate(axes):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(titles[i], fontsize=22)
        ax.set_xlabel(xlabels[i], fontsize=22)
        ax.tick_params(axis='both', labelsize=15)
    
    axes[0].set_ylabel('Relative Prevalence', fontsize=22)
    axes[2].legend(fontsize=14, frameon=False)
    
    plt.tight_layout()
    plt.show()

# --- Parameters ---
fixed_param = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 1.0]

# --- Draw both plots side by side ---
plot_lines_side_by_side(fixed_param, relative=True)


# %%

# %%
def compute_nearest_neighbor_mean(graph, x):
    x_nn = {}
    for v in graph.nodes():
        neighbor_signals = [x[u] for u in graph.predecessors(v)]
        x_nn[v] = np.mean(neighbor_signals) if neighbor_signals else 0.0
    return x_nn

def plot_nearest_neighbor_mean(node_list, suscep, suscep_nn, suscep_r, suscep_nn_r):

    x, y = [], []
    x_r, y_r = [], []
    for node in node_list:
        x.append(suscep[node])
        y.append(suscep_nn[node])
        x_r.append(suscep_r[node])
        y_r.append(suscep_nn_r[node])
    
    cmap = plt.get_cmap('YlGnBu_r')
    plt.figure(figsize=(10, 5), dpi=200)   
    x_list = [x, x_r]
    y_list = [y, y_r]
    title_list = ['Original', 'Randomized']
    for i in range(2):
        plt.subplot(1, 2, i+1)
        kde = sns.kdeplot(x=x_list[i], y=y_list[i], fill=True, cmap=cmap, thresh=0.000, levels=10)
        plt.xlabel(r'$s_v$', fontsize=22)
        plt.ylabel(r'$s_v^{\mathrm{NN}}$', fontsize=22)
        plt.xlim(0, 0.5)
        plt.ylim(0, 0.5)
        plt.title(title_list[i], fontsize=22)
        plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tight_layout()
    plt.show()

nikolov_graph = Nikolov_susceptibility_graph()
suscep = {node: nikolov_graph.nodes[node]['suscep'] for node in nikolov_graph.nodes}
suscep_nn = compute_nearest_neighbor_mean(nikolov_graph, suscep)
nikolov_graph_r = randomized_nikolov_graph()
suscep_r = {node: nikolov_graph_r.nodes[node]['suscep'] for node in nikolov_graph_r.nodes}
suscep_nn_r = compute_nearest_neighbor_mean(nikolov_graph_r, suscep_r)
node_list = list(nikolov_graph.nodes)

plot_nearest_neighbor_mean(node_list, suscep, suscep_nn, suscep_r, suscep_nn_r)


# %%
importlib.reload(util)
# misinformation prevalence with prebunking and contextualization under fixed eta
graph_name = 'nikolov'
print(graph_name)


print('misinformation prevalence for (epsilon, eta) on the nikolov network')
intervention_types = ['nudging', 'prebunking_random', 'contextualization']
path_list1 = []
for intervention_type in intervention_types:
    print(f"intervention type: {intervention_type}")
    path = f'results/{graph_name}/{intervention_type}_vary_eta.npy'
    path_list1.append(path)
    # util.plot_multiple_heatmaps(2, 1, [path, path], [False, True], [None, None])
util.plot_multiple_heatmaps(2, 3, path_list1, [False]*3+[True]*3, intervention_types+[None]*3)

print('misinformation prevalence for (epsilon, delta) and (epsilon, phi) on the nikolov network')
path_list2 = []
for intervention_type in intervention_types[1:]:
    print(f"intervention type: {intervention_type}")
    path = f'results/{graph_name}/{intervention_type}.npy'
    path_list2.append(path)
    # util.plot_multiple_heatmaps(2, 1, [path, path], [False, True], [None, None])

print('misinformation prevalence for targeted prebunking on the nikolov network')
path_list3 = []
target_strategies = ['random', 'high_degree', 'high_susceptible', 'cocoon']
for target_selection in target_strategies:
    print(f"target selection: {target_selection}")
    path = f'results/{graph_name}/prebunking_{target_selection}.npy'
    path_list3.append(path)
    # util.plot_multiple_heatmaps(2, 1, [path, path], [False, True], [None, None])


print('misinformation prevalence for targeted prebunking on the nikolov network')
path_list4 = []
for target_selection in target_strategies:
    print(f"target selection: {target_selection}")
    path = f'results/{graph_name}/prebunking_{target_selection}_vary_eta.npy'
    path_list4.append(path)
    # util.plot_multiple_heatmaps(2, 1, [path, path], [False, True], [None, None])

util.plot_multiple_heatmaps(2, 3, path_list1 * 2, [False]*3+[True]*3, [None]*6, plot_critical_curve=True)
util.plot_multiple_heatmaps(2, 2, path_list2 * 2, [False]*2+[True]*2, [None]*4, plot_critical_curve=True)
util.plot_multiple_heatmaps(2, 4, path_list3 * 2, [False]*4+[True]*4, [None]*8, plot_critical_curve=True)
util.plot_multiple_heatmaps(2, 4, path_list4 * 2, [False]*4+[True]*4, [None]*8)

# %%

importlib.reload(util)
print('difference between random and targeted prebunking on the nikolov network')
path_random = f'results/{graph_name}/prebunking_random.npy'
target_strategies = ['high_degree', 'high_susceptible', 'cocoon']
path_pair_list1 = []
for target_selection in target_strategies:
    print(f"target selection: {target_selection}")
    path_targeted = f'results/{graph_name}/prebunking_{target_selection}.npy'
    path_pair = [(path_random, path_targeted)]
    path_pair_list1 += path_pair
    # util.plot_multiple_heatmaps(1, 1, path_pair, [False], [None])

print('difference between random and targeted prebunking on the nikolov network')
path_random = f'results/{graph_name}/prebunking_random_vary_eta.npy'
path_pair_list2 = []
for target_selection in target_strategies:
    print(f"target selection: {target_selection}")
    path_targeted = f'results/{graph_name}/prebunking_{target_selection}_vary_eta.npy'
    path_pair = [(path_random, path_targeted)]
    path_pair_list2 += path_pair
    # util.plot_multiple_heatmaps(1, 1, path_pair, [False], [None])

title_list = [None] * 6
util.plot_multiple_heatmaps(1, 3, path_pair_list1, [False]*3, title_list, plot_axline=False, plot_critical_curve=True)
util.plot_multiple_heatmaps(1, 3, path_pair_list2, [False]*3, title_list, plot_axline=False)


# %%

importlib.reload(util)
print('relative prevalence of single and combined interventions on the nikolov network')

print('base')
target_selection = 'high_degree'
path = f'results/{graph_name}/compare_single_and_combined_intervention_{target_selection}.npy'
prevalences_list = np.load(path, allow_pickle=True)
util.violin_plot(prevalences_list, graph_name, relative_suppression=True)

print('improve strength')
target_selection = 'high_degree'
path = f'results/{graph_name}/compare_single_and_combined_intervention_{target_selection}_improve_strength.npy'
prevalences_list = np.load(path, allow_pickle=True)
util.violin_plot(prevalences_list, graph_name, relative_suppression=True)

print('improve reach')
target_selection = 'high_degree'
path = f'results/{graph_name}/compare_single_and_combined_intervention_{target_selection}_improve_reach.npy'
prevalences_list = np.load(path, allow_pickle=True)
util.violin_plot(prevalences_list, graph_name, relative_suppression=True)

print('improve both')
target_selection = 'high_degree'
path = f'results/{graph_name}/compare_single_and_combined_intervention_{target_selection}_improve_both.npy'
prevalences_list = np.load(path, allow_pickle=True)
util.violin_plot(prevalences_list, graph_name, relative_suppression=True)

# %%

lambda_max = 324.0259937461831

delta_pre = 0.5
epsilon = np.linspace(0, 1, 100)
eta_nud = 1 / ((1 - epsilon) * lambda_max)
eta_pre = 1 / ((1 - delta_pre * epsilon) * lambda_max)
plt.plot(epsilon, eta_nud, label='nudging')
plt.plot(epsilon, eta_pre, label='prebunking')
plt.ylim((0, 0.1))
plt.xlabel('epsilon')
plt.ylabel('eta')
plt.title('eta vs epsilon')
plt.legend()
plt.show()

# %%
lambda_max = 324.0259937461831

eta = 0.026
epsilon = np.linspace(0, 1, 100)
delta = (1 / epsilon) * (1 - 1/ (eta * lambda_max))
plt.plot(epsilon, delta)
plt.xlabel('epsilon')
plt.ylabel('delta')
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.title('delta vs epsilon')
plt.legend()
plt.show()

# %%

delta_range = np.arange(0.1, 1.005, 0.01)
eps_degree = [None, None, 0.99609375, 0.98828125, 0.98046875, 0.974609375, 0.96875, 0.962890625, 0.95703125, 0.9521484375, 0.947265625, 0.943359375, 0.939453125, 0.9345703125, 0.9306640625, 0.927734375, 0.923828125, 0.919921875, 0.9169921875, 0.9130859375, 0.9111328125, 0.9091796875, 0.90625, 0.904296875, 0.90234375, 0.8994140625, 0.8984375, 0.896484375, 0.8955078125, 0.89453125, 0.892578125, 0.8916015625, 0.8916015625, 0.890625, 0.8896484375, 0.888671875, 0.888671875, 0.8876953125, 0.8876953125, 0.88671875, 0.88671875, 0.88671875, 0.8857421875, 0.8857421875, 0.8857421875, 0.884765625, 0.884765625, 0.884765625, 0.884765625, 0.8837890625, 0.8837890625, 0.8837890625, 0.8837890625, 0.8837890625, 0.8837890625, 0.8837890625, 0.8837890625, 0.8837890625, 0.8837890625, 0.8837890625, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125]
eps_suscep = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.9716796875, 0.958984375, 0.9296875, 0.916015625, 0.9140625, 0.9130859375, 0.91015625, 0.90625, 0.904296875, 0.90234375, 0.900390625, 0.8984375, 0.896484375, 0.89453125, 0.892578125, 0.8916015625, 0.890625, 0.8896484375, 0.888671875, 0.888671875, 0.8876953125, 0.88671875, 0.8857421875, 0.8857421875, 0.884765625, 0.884765625, 0.884765625, 0.884765625, 0.884765625, 0.8837890625, 0.8837890625, 0.8837890625, 0.8837890625, 0.8837890625, 0.8837890625, 0.8837890625, 0.8837890625, 0.8837890625, 0.8837890625, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125]
eps_cocoon = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.994140625, 0.9912109375, 0.9833984375, 0.9794921875, 0.9765625, 0.9697265625, 0.96484375, 0.9599609375, 0.94921875, 0.9423828125, 0.9345703125, 0.931640625, 0.92578125, 0.921875, 0.91796875, 0.912109375, 0.908203125, 0.900390625, 0.896484375, 0.890625, 0.8828125, 0.8828125, 0.8828125]
eps_random = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.9892578125, 0.9794921875, 0.9716796875, 0.9580078125, 0.9541015625, 0.94140625, 0.9326171875, 0.9228515625, 0.9150390625, 0.90625, 0.89453125, 0.8828125]

plt.plot(eps_degree, delta_range, '.-', label='degree-based')
plt.plot(eps_suscep, delta_range, '.-', label='susceptibility-based')
plt.plot(eps_cocoon, delta_range, '.-', label='cocoon-based')
plt.plot(eps_random, delta_range, '.-', label='random-based')
plt.xlabel('epsilon')
plt.ylabel('delta')
plt.title('delta vs epsilon')
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.legend()
plt.show()
# %%
