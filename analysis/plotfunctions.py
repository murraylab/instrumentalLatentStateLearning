"""
Copyright 2021, Warren Woodrich Pettine

This contains code pertaining to neural networks used in "Pettine, W. W., Raman, D. V., Redish, A. D., Murray, J. D.
“Human latent-state generalization through prototype learning with discriminative attention.” December 2021. PsyArXiv

https://psyarxiv.com/ku4fr

This incomplete module contains code used to plot figures in the paper.
"""

from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from simulations.experiments import getPredictedRDMs
rcParams['axes.unicode_minus'] = False #Use hypens in labels


def removeSpines(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return ax

def plotRDM(rdm, fontsize=14, ax=None, save_bool=False, ttl="ANN RDM", fig_dir='plots/', fig_name='nn_rdm.png',
            version=2,cbar_shrink=0.65):
    rdm_mats = [rdm] + list(getPredictedRDMs(version=version))
    if version == 2:
        ttls = [ttl, 'Exemplar', 'Prototype', 'Discriminative']
        n_plots = 4
    elif version == 1:
        ttls = [ttl, 'Exemplar', 'Prototype', 'Discriminative 1', 'Discriminative 2']
        n_plots = 5
    else:
        raise ValueError("Version must be 1 or 2")
    if ax is None:
        fig, ax = plt.subplots(1, n_plots, figsize=(4*n_plots, 4))
    for i in range(n_plots):
        ax[i] = sns.heatmap(rdm_mats[i], square=True, cmap='OrRd', ax=ax[i], cbar_kws={'shrink': cbar_shrink},
                            yticklabels=np.arange(1, rdm_mats[i].shape[0] + 1),
                            xticklabels=np.arange(1, rdm_mats[i].shape[0] + 1))
        ax[i].collections[0].colorbar.set_label("Euclidean Distance", fontsize=fontsize)
        ax[i].invert_yaxis()
        ax[i].set_xlabel("Stimulus", fontsize=fontsize)
        ax[i].set_ylabel("Stimulus", fontsize=fontsize)
        ax[i].set_title(ttls[i], fontsize=fontsize + 2)
        ax[i].tick_params(labelsize=fontsize - 2)
    fig.tight_layout()
    if save_bool:
        plt.savefig(os.path.join(fig_dir, f'{fig_name}'), dpi=300)
    else:
        return ax
