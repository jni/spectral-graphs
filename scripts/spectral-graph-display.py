import numpy as np
from scipy import io, sparse, linalg
# run this from elegant scipy chapter

chem = np.load('chem-network.npy')
gap = np.load('gap-network.npy')
neuron_types = np.load('neuron-types.npy')
neuron_ids = np.load('neurons.npy')
A = chem + gap
n = A.shape[0]
c = (A + A.T) / 2
d = sparse.diags([np.sum(c, axis=0)], [0])
d = d.toarray()
L = np.array(d - c)
b = np.sum(c * np.sign(A - A.T), axis=1)
z = np.linalg.pinv(L) @ b
# IPython log file
dinv2 = np.copy(d)
diag = (np.arange(n), np.arange(n))
dinv2[diag] = dinv[diag] ** (-.5)
q = dinv2 @ L @ dinv2
eigvals, vec = linalg.eig(q)
x = dinv2 @ vec[:, 1]
x.shape
from matplotlib import pyplot as plt
from matplotlib import colors
ii, jj = np.nonzero(c)
plt.scatter(x, z, c=neuron_types, cmap=colors.ListedColormap(((1, 0, 0), (0, 1, 0), (0, 0, 1))), zorder=1)
for src, dst in zip(ii, jj):
    plt.plot(x[[src, dst]], z[[src, dst]], c=(0.85, 0.85, 0.85), lw=0.2, alpha=0.5, zorder=0)
for x0, z0, neuron_id in zip(x, z, neuron_ids):
    plt.text(x0, z0, '  ' + neuron_id,
             horizontalalignment='left', verticalalignment='center',
             fontsize=4, zorder=2)
    
