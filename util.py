import glob
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import BoundaryNorm, ListedColormap

plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams["mathtext.fontset"] = 'cm'
# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.family'] = 'Arial'


def relative_suppression(heatmap_data):
    data_rel = np.array([heatmap_data[i, :]/heatmap_data[i, 0] for i in range(heatmap_data.shape[0])]) 
    return data_rel

def get_seed_users(graph, is_single_seed=True):
    if graph.graph['graph_name'] == 'test':
        out_degrees = dict(graph.out_degree())
        max_node = max(out_degrees, key=out_degrees.get)
        return [max_node]
    if is_single_seed:
        # suscep=1かつout_degreeが最大のノード
        seed_user = 131989
        seed_users = [seed_user]
        # print(f"seed user: {seed_user}, suscep: {graph.nodes[seed_user]['suscep']}, out_degree: {graph.out_degree(seed_user)}")
    else:
        # suscep=1のノードを全て
        suscep_items = [(node, graph.nodes[node].get('suscep', 0)) for node in graph.nodes]
        seed_users = [node for node, suscep in suscep_items if suscep == 1]
        # for seed_user in seed_users:
        #     print(f"seed user: {seed_user}, suscep: {graph.nodes[seed_user]['suscep']}, out_degree: {graph.out_degree(seed_user)}")
    return seed_users

def violin_plot(prevalences_list, graph_name, relative_suppression=False):
    plt.figure(figsize=(6, 4), dpi=300)
    # 略称ラベル: None, Nud, Pre, Ctx, All
    intervention_type_list = ['None', 'Pre', 'Ctx', 'Nud', 'All']
    prevalences_dict = {
        'None': prevalences_list[0],
        'Pre': prevalences_list[1],
        'Ctx': prevalences_list[2],
        'Nud': prevalences_list[3],
        'All': prevalences_list[4],
    }
    intervention_type_list_sort = ['None', 'Nud', 'Pre', 'Ctx', 'All']
    
    prevalences_list = [prevalences_dict[intervention_type] for intervention_type in intervention_type_list_sort]
    palette = {
        'None': 'darkgray',
        'Pre': 'steelblue',
        'Ctx': 'darkorange',
        'Nud': 'darkseagreen',
        'All': 'indianred',
    }
    if relative_suppression:
        # 'None' intervention is always the first column
        rho0 = np.mean(prevalences_dict['None'])
        relative_supp = (prevalences_list) / rho0
        df = pd.DataFrame(relative_supp.T, columns=intervention_type_list_sort)
        ylabel = 'Relative Prevalence'
        plt.ylim((0.62, 1.1))
        plt.yticks([0.7, 0.8, 0.9, 1.0])
    else:
        df = pd.DataFrame(prevalences_list.T, columns=intervention_type_list_sort)
        ylabel = 'Prevalence'
    # 各カテゴリごとに平均値を計算し，print()
    means = df.mean()
    print(means)
    for category, val in means.items():
        i = intervention_type_list_sort.index(category)
        plt.text(i+0.15, val+0.03, f"{val:.2f}", ha="center", va="bottom", fontsize=20, fontweight="bold", color="black")
        # print(f"{category} 平均: {val}")
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=20)
    sns.violinplot(data=df, palette=palette, alpha=0.9)
    plt.ylabel(ylabel, fontsize=24)
    os.makedirs(f'results/{graph_name}/png', exist_ok=True)
    plt.savefig(f'results/{graph_name}/png/compare_single_and_combined_intervention_dpi300.png')
    plt.show()

def plot_multiple_heatmaps(row, col, path_list, relative_flag_list, title_list, diff_cmap='bwr', diff_vlim=0.3, plot_axline=True, plot_critical_curve=False, lambda_max=324.0259937461831, delta_pre=0.5):
    # figsizeをrow, colから自動調整
    base_width_per_col = 4
    base_height_per_row = 3.5
    # fisizeに基づきフォントサイズを調整
    fontsize = 20 * ((col+3) / (2+3))
    # print(fontsize)
    figsize = (base_width_per_col * col, base_height_per_row * row)
    plt.figure(figsize=figsize, dpi=300)
    for i, path in enumerate(path_list):
        plt.subplot(row, col, i+1)
        relative = relative_flag_list[i]
        title = title_list[i]
        if len(path) == 2:
            path1, path2 = path
            if path1.endswith('vary_eta.npy'):
                extent = [0, 1, 0, 0.1]
                yticks = [0.0, 0.05, 0.1]
                plt.yticks(yticks, [f"{y:.2f}" for y in yticks])
            else:
                extent = [0, 1, 0, 1]
            data_dict = np.load(path1, allow_pickle=True).item()
            data_dict2 = np.load(path2, allow_pickle=True).item()
            data = data_dict['heatmap_data'] - data_dict2['heatmap_data']            
            # levels = np.array([-0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
            # reds_cmap = plt.get_cmap('Reds')
            # blues_cmap = plt.get_cmap('Blues_r')
            # reds_colors = reds_cmap(np.linspace(0, 1, 6))
            # blues_colors = blues_cmap(np.linspace(0, 1, 6))
            # colors_descrete_reds = ListedColormap(reds_colors).colors.tolist()
            # colors_descrete_blues = ListedColormap(blues_colors).colors.tolist()
            # colors = colors_descrete_blues[-3:-1] + colors_descrete_reds
            # cmap = ListedColormap(colors)
            # norm = BoundaryNorm(boundaries=levels, ncolors=len(colors))
            # im = plt.imshow(data, cmap=cmap, norm=norm, aspect='auto', extent=extent, origin='lower')
            im = plt.imshow(data, cmap='PuOr', aspect='auto', extent=extent, origin='lower', vmin=-diff_vlim, vmax=diff_vlim)
        else:
            if path.endswith('vary_eta.npy'):
                extent = [0, 1, 0, 0.1]
                yticks = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]
                plt.yticks(yticks, [f"{y:.2f}" for y in yticks])
            else:
                extent = [0, 1, 0, 1]
            data_dict = np.load(path, allow_pickle=True).item()
            if relative:
                data = relative_suppression(data_dict['heatmap_data'])
                levels = np.array([0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0])
                norm = BoundaryNorm(boundaries=levels, ncolors=256)
                im = plt.imshow(data, cmap='Spectral_r', norm=norm, aspect='auto', extent=extent, origin='lower')
                im.set_alpha(0.8)
            else:
                plt.imshow(data_dict['heatmap_data'], cmap='YlGnBu_r', aspect='auto', extent=extent, origin='lower', vmin=0)
        plt.colorbar()
        x = {'prebunking': 0.204, 'contextualization': 0.342, 'nudging': 0.143,  'debunking': 0.342}
        # if plot_axline:
        #     plt.axvline(x=x[data_dict['intervention_type']], color='w', linestyle='--', linewidth=2)
        #     if data_dict['labels']['ylabel'] == r'$\eta$':
        #         plt.axhline(y=0.026, color='w', linestyle='--', linewidth=2)
        
        # 臨界曲線をプロット（vary_etaヒートマップのみ）
        if plot_critical_curve:
            if len(path) == 2:
                path1 = path[1]
            else:
                path1 = path
            if path1.endswith('vary_eta.npy'):
                epsilon_range = np.linspace(0.01, 1.0, 100)
                intervention_type = data_dict['intervention_type']
                if intervention_type == 'nudging':
                    # nudging: eta_c = 1 / ((1 - epsilon) * lambda_max)
                    eta_critical = 1 / ((1 - epsilon_range) * lambda_max)
                elif intervention_type == 'prebunking':
                    # prebunking: eta_c = 1 / ((1 - delta_pre * epsilon) * lambda_max)
                    eta_critical = 1 / ((1 - delta_pre * epsilon_range) * lambda_max)
                else:
                    eta_critical = None
                
                if eta_critical is not None:
                    # eta範囲内のみプロット
                    mask = eta_critical <= 0.1
                    plt.plot(epsilon_range[mask], eta_critical[mask], 'r:', linewidth=3, label=r'$\eta_c$')
            elif path1.endswith('random.npy'):
                delta = np.arange(0.1, 1.005, 0.01)
                eps_random = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.9892578125, 0.9794921875, 0.9716796875, 0.9580078125, 0.9541015625, 0.94140625, 0.9326171875, 0.9228515625, 0.9150390625, 0.90625, 0.89453125, 0.8828125]
                plt.plot(eps_random, delta, 'r:', linewidth=3)
            elif path1.endswith('high_degree.npy'):
                delta = np.arange(0.1, 1.005, 0.01)
                eps_degree = [None, None, 0.99609375, 0.98828125, 0.98046875, 0.974609375, 0.96875, 0.962890625, 0.95703125, 0.9521484375, 0.947265625, 0.943359375, 0.939453125, 0.9345703125, 0.9306640625, 0.927734375, 0.923828125, 0.919921875, 0.9169921875, 0.9130859375, 0.9111328125, 0.9091796875, 0.90625, 0.904296875, 0.90234375, 0.8994140625, 0.8984375, 0.896484375, 0.8955078125, 0.89453125, 0.892578125, 0.8916015625, 0.8916015625, 0.890625, 0.8896484375, 0.888671875, 0.888671875, 0.8876953125, 0.8876953125, 0.88671875, 0.88671875, 0.88671875, 0.8857421875, 0.8857421875, 0.8857421875, 0.884765625, 0.884765625, 0.884765625, 0.884765625, 0.8837890625, 0.8837890625, 0.8837890625, 0.8837890625, 0.8837890625, 0.8837890625, 0.8837890625, 0.8837890625, 0.8837890625, 0.8837890625, 0.8837890625, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125]
                plt.plot(eps_degree, delta, 'r:', linewidth=3)
                # plt.plot(eps_random, delta, 'k:')
            elif path1.endswith('high_susceptible.npy'):
                delta = np.arange(0.1, 1.005, 0.01)
                eps_suscep = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.9716796875, 0.958984375, 0.9296875, 0.916015625, 0.9140625, 0.9130859375, 0.91015625, 0.90625, 0.904296875, 0.90234375, 0.900390625, 0.8984375, 0.896484375, 0.89453125, 0.892578125, 0.8916015625, 0.890625, 0.8896484375, 0.888671875, 0.888671875, 0.8876953125, 0.88671875, 0.8857421875, 0.8857421875, 0.884765625, 0.884765625, 0.884765625, 0.884765625, 0.884765625, 0.8837890625, 0.8837890625, 0.8837890625, 0.8837890625, 0.8837890625, 0.8837890625, 0.8837890625, 0.8837890625, 0.8837890625, 0.8837890625, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125, 0.8828125]
                plt.plot(eps_suscep, delta, 'r:', linewidth=3)
                # plt.plot(eps_random, delta, 'k:')
            elif path1.endswith('cocoon.npy'):
                delta = np.arange(0.1, 1.005, 0.01)
                eps_cocoon = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 0.994140625, 0.9912109375, 0.9833984375, 0.9794921875, 0.9765625, 0.9697265625, 0.96484375, 0.9599609375, 0.94921875, 0.9423828125, 0.9345703125, 0.931640625, 0.92578125, 0.921875, 0.91796875, 0.912109375, 0.908203125, 0.900390625, 0.896484375, 0.890625, 0.8828125, 0.8828125, 0.8828125]
                plt.plot(eps_cocoon, delta, 'r:', linewidth=3)
                # plt.plot(eps_random, delta, 'k:')
        
        plt.xlabel(data_dict['labels']['xlabel'], fontsize=fontsize )
        plt.ylabel(data_dict['labels']['ylabel'], fontsize=fontsize)
        if title is not None:
            plt.title(title, fontsize=fontsize)
    plt.tight_layout()
    plt.show()
