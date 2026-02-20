import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import LinearOperator
import networkx as nx
from load_graph import Nikolov_susceptibility_graph, randomized_nikolov_graph
import util
import random
import time
import tqdm
import matplotlib.pyplot as plt

def prebunking_targets(graph, delta_pre, susceptibility, target_selection='random'):
    N = len(graph.nodes())
    k = max(1, int(N * delta_pre))
    seed_nodes = util.get_seed_users(graph, is_single_seed=True)
    inactive_nodes = set(graph.nodes()) - set(seed_nodes)

    random.seed(42)
    np.random.seed(42)
    if target_selection == 'high_degree':
        cand = sorted(inactive_nodes, key=lambda v: graph.out_degree(v), reverse=True)
    elif target_selection == 'high_susceptible':
        cand = sorted(inactive_nodes, key=lambda v: susceptibility[v], reverse=True)
    elif target_selection == 'cocoon':
        out_neighbors = set()
        for s in seed_nodes:
            out_neighbors.update([nbr for nbr in graph.successors(s) if nbr in inactive_nodes])
        nbr_list = list(out_neighbors)
        random.shuffle(nbr_list)
        cand = nbr_list[:k]
        if len(cand) < k:
            two_hop_neighbors = set()
            for s in seed_nodes:
                for nbr in graph.successors(s):
                    two_hop_neighbors.update([nbr2 for nbr2 in graph.successors(nbr) if nbr2 in inactive_nodes])
            two_hop_neighbors = list(two_hop_neighbors)
            random.shuffle(two_hop_neighbors)
            kk = k - len(cand)
            cand += two_hop_neighbors[:kk]
            if len(cand) < k:
                rest = list(inactive_nodes - set(cand))
                random.shuffle(rest)
                cand_extra = rest[:k-len(cand)]
                cand += cand_extra
    else:
        cand = list(inactive_nodes)[:]
        random.shuffle(cand)
    targets = cand[:k]
    return targets
    

def qmf_threshold_eta_linearop(A_csr: sp.csr_matrix, s: np.ndarray,
                               tol=1e-6, maxiter=5000):
    n = A_csr.shape[0]

    def matvec(x):
        # (A @ diag(s)) x = A @ (s * x)
        return A_csr @ (s * x)

    M_op = LinearOperator((n, n), matvec=matvec, dtype=np.float64)

    vals, _ = spla.eigs(M_op, k=1, which='LM', tol=tol, maxiter=maxiter)
    lambda_max = float(np.real(vals[0]))
    eta_c = 1.0 / lambda_max
    return eta_c, lambda_max

def qmf_lambda_max(A_csr, s_vec, tol=1e-6, maxiter=5000):
    n = A_csr.shape[0]
    def matvec(x):
        return A_csr @ (s_vec * x)  # A @ diag(s) @ x
    M_op = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
    vals, _ = spla.eigs(M_op, k=1, which='LM', tol=tol, maxiter=maxiter)
    return float(np.real(vals[0]))

def s_after_targeted_prebunking(graph, s_dict, targets, eps):
    s_new = s_dict.copy()
    for v in targets:
        s_new[v] = (1.0 - eps) * s_new[v]
    s_new = np.array([s_new[v] for v in graph.nodes()])
    return s_new

def critical_eps_for_delta(A_csr, s, s_dict, graph, eta, delta, strategy,
                           tol_eps=1e-3, tol_eigs=1e-6, maxiter=5000):
    # 1) δ固定でターゲット固定
    targets = prebunking_targets(graph, delta, s_dict, target_selection=strategy)
    targets = set(targets)

    # 2) f(eps) = lambda_max - 1/eta
    inv_eta = 1.0 / eta

    def f(eps):
        s_new = s_after_targeted_prebunking(graph, s_dict, targets, eps)
        lam = qmf_lambda_max(A_csr, s_new, tol=tol_eigs, maxiter=maxiter)
        return lam - inv_eta, lam

    f0, lam0 = f(0.0)
    f1, lam1 = f(1.0)

    if f0 <= 0:
        # 介入なしで既に臨界以下
        return 0.0, lam0, "already_subcritical"
    if f1 > 0:
        # 最大介入でも臨界に届かない
        return None, lam1, "cannot_reach"

    # 3) 二分探索
    lo, hi = 0.0, 1.0
    lam_mid = None
    while (hi - lo) > tol_eps:
        mid = 0.5 * (lo + hi)
        fm, lam_mid = f(mid)
        if fm > 0:
            lo = mid
        else:
            hi = mid

    # hiが「臨界に入る最小のeps」に近い
    return hi, lam_mid, "ok"


G = Nikolov_susceptibility_graph()
G.graph['graph_name'] = 'nikolov'
# G = randomized_nikolov_graph()
A = sp.csr_matrix(nx.adjacency_matrix(G))
s = np.array([G.nodes[node]['suscep'] for node in G.nodes()])
s_dict = {node: G.nodes[node]['suscep'] for node in G.nodes()}
start_time = time.time()
eta_c, lambda_max = qmf_threshold_eta_linearop(A, s)
end_time = time.time()
print(f"eta_c: {eta_c:.3f}, lambda_max: {lambda_max:.3f}")
print(f"Time taken: {end_time - start_time} seconds")

print('Critical eps for delta')
delta_range = np.arange(0.1, 1.005, 0.01)
target_strategies = ['high_degree', 'high_susceptible', 'cocoon', 'random']
colors_dict = {'high_degree': 'red', 'high_susceptible': 'green', 'cocoon': 'blue', 'random': 'purple'}
label_dict = {'high_degree': 'Degree-based', 'high_susceptible': 'Susceptibility-based', 'cocoon': 'Distance-based', 'random': 'Random'}
marker_dict = {'high_degree': 'o', 'high_susceptible': 's', 'cocoon': '^', 'random': 'x'}
for target_strategy in target_strategies:
    print(target_strategy)
    eta_pred = 0.026
    eps_c_list = []
    start_time = time.time()
    for delta_pre in tqdm.tqdm(delta_range):
        hi, lam_mid, status = critical_eps_for_delta(A, s, s_dict, G, eta_pred, delta_pre, strategy=target_strategy)
        # if hi is not None:
        #     # print(f"delta: {delta_pre:.3f}, eps_c: {hi:.3f}, lam_mid: {lam_mid:.3f}, status: {status}")
        #     plt.plot(hi, delta_pre, marker_dict[target_strategy], color=colors_dict[target_strategy], label=label_dict[target_strategy], markersize=10, markerfacecolor='none')
        # else:
        #     print(f"delta: {delta_pre:.3f}, eps_c: N/A, lam_mid: {lam_mid:.3f}, status: {status}")
        eps_c_list.append(hi)
    print(eps_c_list)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    # plt.xlim((0.8, 1))
    # plt.ylim((0, 1))
    # plt.xlabel('epsilon')
    # plt.ylabel('delta')
    # plt.title('Critical eps for delta')
# plt.legend()
# plt.show()
# epsilon_range = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
# delta_range = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
# Lambda_max_data = np.zeros((len(epsilon_range), len(delta_range)))
# for delta_pre in tqdm.tqdm(delta_range):
#     targets = prebunking_targets(G, delta_pre, s_dict, target_selection='high_degree')
#     for epsilon_pre in epsilon_range:
#         s_prebunked = []
#         for node in G.nodes():
#             suscep_v = s_dict[node]
#             if node in targets:
#                 suscep_v = (1 - epsilon_pre) * suscep_v
#             s_prebunked.append(suscep_v)
#         s_prebunked = np.array(s_prebunked)
#         _, lambda_max = qmf_threshold_eta_linearop(A, s_prebunked)

#         epsilon_idx = epsilon_range.index(epsilon_pre)
#         delta_idx = delta_range.index(delta_pre)
#         Lambda_max_data[epsilon_idx, delta_idx] = lambda_max

# print(Lambda_max_data)

