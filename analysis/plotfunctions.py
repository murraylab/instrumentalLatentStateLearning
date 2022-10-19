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
import pandas as pd
import os
from simulations.experiments import getPredictedRDMs
from analysis.general import smooth, figurePrototypesConversion, contextGenModelConfMats, orderConfMat
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


def plotConfMat(confusion_mat,conversion=None,ax=None,ttl='',fontsize=13,vmin=0, vmax=1,cmap='magma',axis_labels_bool=True):
    if conversion is not None:
        confusion_mat = orderConfMat(confusion_mat,conversion)
    if ax is None:
        fig, ax = plt.subplots()
    if confusion_mat.ndim==3:
        confusion_mat = np.sum(confusion_mat,axis=2) / np.sum(np.sum(confusion_mat,axis=2),axis=1)[:,None]
    confusion_mat[np.isnan(confusion_mat)] = 0
    ax.imshow(confusion_mat,origin='lower',cmap=cmap,vmin=vmin, vmax=vmax)
    ax.set_yticks(np.arange(8))
    ax.set_yticklabels(np.arange(1,9))
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(['A','B','C','D'])
    if axis_labels_bool:
        ax.set_ylabel('Stimulus',fontsize=fontsize)
        ax.set_xlabel('Action',fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.set_title(ttl,fontsize=fontsize+1)
    return ax


def plotContextGenIdealizedConfMats(save_bool=False,fig_dir=None,extension='svg',\
    fig_scale_factor=0.8,fontsize=13,conversion=None,context_gen_version=1,fig_name='conf_mats_idealized'):

    if conversion is None:
        conversion = figurePrototypesConversion(experimental_set='humans', context_gen_version=context_gen_version)

    if context_gen_version == 1:
        confusion_mat_pred_att, confusion_mat_pred_proto, confusion_mat_pred_discrim_1att,\
            confusion_mat_pred_discrim_2att, confusion_mat_pred_state_bias, confusion_mat_pred_random = \
            contextGenModelConfMats(context_gen_version=1,experimental_set='humans',method='winner',return_random=True)

        sns.set_style(style='white')
        fig, ax = plt.subplots(1,6,figsize=np.array([9/5*6,4])*fig_scale_factor)
        ax[0] = plotConfMat(confusion_mat_pred_discrim_1att,conversion=conversion,ax=ax[0],ttl='(D1)\nFeature',fontsize=fontsize)
        ax[1] = plotConfMat(confusion_mat_pred_discrim_2att,conversion=conversion,ax=ax[1],ttl='(D2)\nFeatures',
            fontsize=fontsize,axis_labels_bool=False)
        ax[2] = plotConfMat(confusion_mat_pred_proto,conversion=conversion,ax=ax[2],ttl='(P)rototype\nCovariance',
            fontsize=fontsize,axis_labels_bool=False)
        ax[3] = plotConfMat(confusion_mat_pred_att,conversion=conversion,ax=ax[3],ttl='(A)ll Features',fontsize=fontsize
            ,axis_labels_bool=False)
        ax[4] = plotConfMat(confusion_mat_pred_state_bias,conversion=conversion,ax=ax[4],ttl='(S)tate Bias',
            fontsize=fontsize,axis_labels_bool=False)
        ax[5] = plotConfMat(confusion_mat_pred_random,conversion=conversion,ax=ax[5],ttl='(I)ntercept\nNull Attention',
            fontsize=fontsize,axis_labels_bool=False)

    elif context_gen_version == 2:
        confusion_mat_pred_att, confusion_mat_pred_proto, confusion_mat_pred_discrim_1att,\
            confusion_mat_pred_state_bias, confusion_mat_pred_random = contextGenModelConfMats(context_gen_version=2,\
                                    experimental_set='humans',method='winner',return_random=True)
        sns.set_style(style='white')
        fig, ax = plt.subplots(1,5,figsize=np.array([9/5*6,4])*fig_scale_factor)
        ax[0] = plotConfMat(confusion_mat_pred_discrim_1att,conversion=conversion,ax=ax[0],ttl='(D1)\nFeature',fontsize=fontsize)
        ax[1] = plotConfMat(confusion_mat_pred_proto,conversion=conversion,ax=ax[1],ttl='(P)rototype\nCovariance',
            fontsize=fontsize,axis_labels_bool=False)
        ax[2] = plotConfMat(confusion_mat_pred_att,conversion=conversion,ax=ax[2],ttl='(A)ll Features',fontsize=fontsize
            ,axis_labels_bool=False)
        ax[3] = plotConfMat(confusion_mat_pred_state_bias,conversion=conversion,ax=ax[3],ttl='(S)tate Bias',
            fontsize=fontsize,axis_labels_bool=False)
        ax[4] = plotConfMat(confusion_mat_pred_random,conversion=conversion,ax=ax[4],ttl='(I)ntercept\nNull Attention',
            fontsize=fontsize,axis_labels_bool=False)

    fig.suptitle('Idealized Attention Model Confusion Matrices',fontsize=fontsize+4)
    fig.tight_layout()
    if save_bool:
        fig.savefig(os.path.join(fig_dir,f'{fig_name}.{extension}'))

    return fig, ax


def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct


###################################
# Human subject-specific functions
###################################


def plotExampleGenMetrics(bar_df, replication=False,save_bool=False,fig_dir='plots/subjects/',extension='svg',
    fig_name='example_gen_metrics'):

    figsize=(12,4)
    fig, ax = plt.subplots(1,3,figsize=figsize)

    ax[0] = sns.histplot(ax=ax[0],data=bar_df,x='P(Correct | Novel, First Gen Appearance) -\nP(Correct | Learned, First Gen Appearance)',
                        bins=np.arange(-0.875,0.375,0.125),stat='probability',color='k')
    ax[1] = sns.histplot(ax=ax[1],data=bar_df,x='P(Error | Novel) - P(Error | Learned)',bins=10,stat='probability',color='k')
    ax[2] = sns.histplot(ax=ax[2],data=bar_df,x='P(Choose D | Novel Stim)',stat='probability',color='k',bins=np.linspace(-0.1,1,9))
    ax[0].axvline(0,linestyle=':',color='k')
    ax[0].axvline(-0.3,linestyle=':',color='k')
    ax[1].axvline(0,linestyle=':',color='k')
    ax[2].axvline(.25,linestyle=':',color='k')
    # ax[0] = formatAx(ax[0],ttl='First Generalization Appearance')
    ax[0] = removeSpines(ax[0])
    ax[0].set_title('First Generalization Appearance')

    ax[1] = removeSpines(ax[1])
    ax[1].set_title('Paired Generalization Difference')

    ax[2] = removeSpines(ax[2])
    ax[2].set_title('Exploration Errors')

    # ax[1] = formatAx(ax[1],ttl='Paired Generalization Difference')
    # ax[2] = formatAx(ax[2],ttl='Exploration Errors')

    fig.tight_layout()
    if save_bool:
        fig.savefig(os.path.join(fig_dir,f'{fig_name}.{extension}'),dpi=300)

    return fig, ax


def plotContextGenSubjectConfMats(confusion_matrix,indx_de,indx_ie,conversion=None,context_gen_version=2,save_bool=False,\
    fig_dir='plots/subjects',extension='svg',fontsize=13,fig_scale_factor=0.8):

    if conversion is None:
        conversion = figurePrototypesConversion(experimental_set='humans', context_gen_version=context_gen_version)

    sns.set_style(style='white')
    fig, ax = plt.subplots(1,6,figsize=np.array([11,4])*fig_scale_factor)
    ax[0] = plotConfMat(confusion_matrix,conversion=conversion,ax=ax[0],ttl='All',fontsize=fontsize)
    ax[1] = plotConfMat(confusion_matrix[:,:,indx_de],conversion=conversion,ax=ax[1],ttl='High DE',fontsize=fontsize,
        axis_labels_bool=False)
    ax[2] = plotConfMat(confusion_matrix[:,:,(indx_de<1) * indx_ie],conversion=conversion,ax=ax[2],ttl='Low DE\nHigh IE',
        fontsize=fontsize,axis_labels_bool=False)
    ax[3] = plotConfMat(confusion_matrix[:,:,(indx_de<1) * (indx_ie<1)],conversion=conversion,ax=ax[3],ttl='Low DE\nLow IE',
        fontsize=fontsize,axis_labels_bool=False)
    ax[4] = plotConfMat(confusion_matrix[:,:,(indx_de) * indx_ie],conversion=conversion,ax=ax[4],ttl='High DE\nHigh IE',
        fontsize=fontsize,axis_labels_bool=False)
    ax[5] = plotConfMat(confusion_matrix[:,:,(indx_de) * (indx_ie<1)],conversion=conversion,ax=ax[5],ttl='High DE\nLow IE',
        fontsize=fontsize,axis_labels_bool=False)

    fig.suptitle('Subject Confusion Matrices',fontsize=fontsize+4)
    fig.tight_layout()
    if save_bool:
        fig.savefig(os.path.join(fig_dir,f'cgv1_conf_mats_groups.{extension}'),dpi=300)

    return fig, ax