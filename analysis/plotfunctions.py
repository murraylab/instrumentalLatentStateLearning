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


def plotNovelExampleGenLCs(performance_categorization_mat, performance_learned_mat, performance_novel_mat, ttl='',
                           fontsize=12,color_learned='#008000', color_novel='#7FFF00', figsize=(7, 3),fig=None,ax=None,
                           label_base='',generalization_overlay=True,mean_only=False,legend_bool=True,axiswidth=1,
                           linewidth_chance=1):
    if ax is None:
        if generalization_overlay:
            fig, ax = plt.subplots(1, 2, figsize=figsize)
        else:
            fig, ax = plt.subplots(2, 2, figsize=(7, 6))
            ax = ax.reshape(-1)

    plotLC(ax[0], performance_categorization_mat, color=color_learned, label=label_base, n_trial_lim=None, linestyle='-',
           ttl='Initial Learning', trial_start=0,fontsize=fontsize,axiswidth=axiswidth,linewidth_chance=linewidth_chance)
    if label_base != '':
        ax[0].legend()
    n_cat_trials = performance_categorization_mat.shape[1]
    if generalization_overlay:
        ax[1] = plotLC(ax[1], performance_learned_mat, color=color_learned, label=f'Learned{label_base}', n_trial_lim=None,
                   linestyle='-', ttl='Novel Example Generalization', trial_start=n_cat_trials,mean_only=mean_only,
                   fontsize=fontsize,axiswidth=axiswidth,linewidth_chance=linewidth_chance)
        ax[1] = plotLC(ax[1], performance_novel_mat, color=color_novel, label=f'Novel{label_base}', n_trial_lim=None,
                   ttl='Novel Example Generalization', trial_start=n_cat_trials,mean_only=mean_only,fontsize=fontsize,
                   axiswidth=axiswidth, linestyle='--',linewidth_chance=linewidth_chance)
    else:
        ax[1] = plotLC(ax[1], performance_learned_mat, color=color_learned, label=f'Learned{label_base}', n_trial_lim=None,
                   linestyle='-', ttl='Generalization BLock (Initallly Learned Examples)', trial_start=n_cat_trials,
                   mean_only=mean_only,fontsize=fontsize,linewidth_chance=linewidth_chance)
        ax[3] = plotLC(ax[3], performance_novel_mat, color=color_novel, label=f'Novel{label_base}', n_trial_lim=None, linestyle='--',
                   ttl='Generalization BLock (Novel Examples)', trial_start=n_cat_trials,mean_only=mean_only,
                   fontsize=fontsize,axiswidth=axiswidth,linewidth_chance=linewidth_chance)
        ax[2].set_visible(False)
    if legend_bool:
        ax[1].legend()
    ax[1].set_xlabel('Trial (Segregated)',fontsize=fontsize)
    ax[1].yaxis.set_visible(False)
    ax[1].spines['left'].set_visible(False)
    if fig is not None:
        fig.suptitle(ttl, fontsize=fontsize + 2)
        fig.tight_layout()
    return fig, ax


def plotLC(ax,y_data=None,y_mn=None,y_se=None,color='k',label=None,n_trial_lim=None,linestyle='-',ttl='',fontsize=12,trial_start=0,
           smooth_window=11,mean_only=False,axiswidth=1,linewidth_chance=1):
    if n_trial_lim is None:
        if y_data is not None:
            n_trials = y_data.shape[1]-1
        else:
            n_trials = len(y_mn)
    else:
        n_trials = n_trial_lim
    x = np.arange(n_trials)+trial_start
    if y_data is not None:
        y = smooth(np.mean(y_data,axis=0),window_len=smooth_window)[:n_trials]
#     error = forwardSmooth((np.std(data_dict_model[block],axis=0)/ \
#             np.sqrt(data_dict_model[block].shape[0]))[:n_trials])
        error = smooth((np.std(y_data,axis=0)/ \
            np.sqrt(y_data.shape[0])))[:n_trials]
    else:
        if (y_mn is None) or (y_se is None):
            raise ValueError('Must pass value for data')
        y, error = y_mn, y_se
    if not mean_only:
        ax.fill_between(x, y-error, y+error, alpha=0.5, edgecolor=color, facecolor=color)
    ax.plot(x,y,color=color,linewidth=.5,label=label,linestyle=linestyle)
    ax.plot(x,x*0+.25,linestyle=':',color='k',linewidth=linewidth_chance)
    ax.set_ylim([0,1.1])
    ax.set_xlabel('Trial',fontsize=fontsize)
    ax.set_ylabel('P(Largest Chosen)',fontsize=fontsize)
    ax.set_title(ttl,fontsize=fontsize+1)
    ax=removeSpines(ax)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(axiswidth)
    ax.tick_params(bottom=True,left=True,labelsize=fontsize,width=axiswidth,length=2)
    return ax