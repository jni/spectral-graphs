from scipy import sparse
import numpy as np
import networkx as nx
import warnings


def affinity_view(A, C, D, L):
    Dinv2 = D.copy()
    Dinv2.data = Dinv2.data ** (-.5)
    Q = Dinv2 * L * Dinv2
    eigvals, vec = sparse.linalg.eigsh(Q, k=3, which='SM')
    _, x, y = (Dinv2 * vec).T
    return x, y


def processing_depth(A, C, L):
    b = C.multiply((A - A.T).sign()).sum(axis=1)
    z, error = sparse.linalg.isolve.cg(L, b, maxiter=int(1e4))
    if error > 0:
        warnings.warn('CG convergence failed after %s iterations' % error)
    elif error < 0:
        warnings.warn('CG illegal input or breakdown')
    return z

def node_coordinates(graph, remove_nodes=None, nodelist=None):
    conn = max(nx.connected_components(graph.to_undirected()),
               key=len)
    graph = graph.subgraph(conn)
    if remove_nodes is not None:
        graph.remove_nodes_from(remove_nodes)
    graph.remove_edges_from(graph.selfloop_edges())
    if nodelist is None:
        names = graph.nodes()
    else:
        names = nodelist
    A = nx.to_scipy_sparse_matrix(graph, nodelist=names)
    C = (A + A.T) / 2
    degrees = np.ravel(C.sum(axis=0))
    D = sparse.diags([degrees], [0]).tocsr()
    L = D - C
    x, y = affinity_view(A, C, D, L)
    z = processing_depth(A, C, L)
    return x, y, z, A, names


from matplotlib import pyplot as plt
from matplotlib import colors

def plot_connectome(neuron_x, neuron_y, links, labels, types):
    colormap = colors.ListedColormap([(1, 0, 0),
                                      (0, 0, 1),
                                      (0, 1, 0)])
    # plot neuron locations:
    plt.scatter(neuron_x, neuron_y, c=types, cmap=colormap, zorder=1)

    # add text labels:
    for x, y, label in zip(neuron_x, neuron_y, labels):
        plt.text(x, y, '  ' + label,
                 horizontalalignment='left', verticalalignment='center',
                 fontsize=5, zorder=2)

    # plot links
    pre, post = np.nonzero(links)
    for src, dst in zip(pre, post):
        plt.plot(neuron_x[[src, dst]], neuron_y[[src, dst]],
                 c=(0.85, 0.85, 0.85), lw=0.2, alpha=0.5, zorder=0)

    plt.show()


def plot_dependencies(xs, ys, A, names, values=None, subsample=10,
                      attenuation=2):
    if values is None:
        values = np.full(len(xs), 1/len(xs))
    values = values ** (1 / attenuation)
    values /= np.sum(values)  # normalize to sum to 1 for probabilities
    if subsample:
        indices = np.random.choice(np.arange(len(xs)), p=values,
                                   size=len(xs) // subsample, replace=False)
    else:
        indices = np.arange(len(xs))
    values = values[indices]
    indices = indices[np.argsort(values)]  # plot low values first
    values = np.sort(values) / np.max(values)  # normalize to max-1 for scaling
    xs = xs[indices]
    ys = ys[indices]
    A = A[indices][:, indices]
    names = [names[i] for i in indices]
    colormap = plt.cm.plasma_r
    plt.scatter(xs, ys, s=values*50, c=values,
                cmap=colormap, alpha=0.5, zorder=1)

    # add text labels
    for x, y, label, val in zip(xs, ys, names, values):
        plt.text(x, y, '   ' + label,
                 horizontalalignment='left', verticalalignment='center',
                 fontsize=12 * val, alpha=val, zorder=2)

    pre, post = np.nonzero(A)
    for src, dst in zip(pre, post):
        plt.plot(xs[[src, dst]], ys[[src, dst]],
                 c=(0.85, 0.85, 0.85), lw=0.2, alpha=0.5, zorder=0)
    plt.show()
