import sys
import os
import numpy as np
import statsmodels.api as sm
import itertools
from analysis.general import contextGenModelConfMats
import pandas as pd
import arviz as az
import bambi as bmb


def fitBayesianWithAssessment(confusion_matrix,context_gen_version=2,priors='half_normal',fitvars=['D','P','A'],
                              conf_mat_method='winner',experimental_set='humans',cores=None,fit_init='adapt_diag'):
    if (context_gen_version != 1) and (context_gen_version != 2):
        raise ValueError('context_gen_version must be either 1 or 2')
    if (fitvars is None) and (context_gen_version==1):
        fitvars = ['D1','D2','P','A']
    elif (fitvars is None) and (context_gen_version==2):
        fitvars = ['D','P','A']
    for var in fitvars:
        if var not in ['D','D1','D2','P','A','S']:
            raise ValueError('fitvars must be a list containing only D, D1, D2, P, A, or S')
    if 'S' in fitvars:
        include_state_bias=True
    else:
        include_state_bias=False
    # Fit Model
    print(f'Fitting Model')
    conf_mat_df, model, fitted = fitBayesian(confusion_matrix,context_gen_version=context_gen_version,fitvars=fitvars,\
                                             priors=priors,include_state_bias=include_state_bias,conf_mat_method=\
                                             conf_mat_method,experimental_set=experimental_set,cores=cores,fit_init=fit_init)
    #Posterior Predictive
    print('Calculating Posterior predictive')
    posterior_predictive = model.predict(fitted, kind="pps")
    # Partial Correlations
    print("Partial correlations")
    pcorr_samples = calcBayesianCoefficients(conf_mat_df, model, fitted)
    #Model comparison (PASS conf_mat_method)
    print('Model comparison')
    models_dic, fitted_dict = predictorDroppingFit(conf_mat_df, model, fitted, priors=priors, version=2,cores=cores,fit_init=fit_init,
                        conf_mat_method=conf_mat_method)
    df_compare = az.compare(fitted_dict, ic="LOO")
    return_dict = {
        'conf_mat_df': conf_mat_df,
        'models_dic': models_dic,
        'fitted_dict': fitted_dict,
        'posterior_predictive': posterior_predictive,
        'df_compare': df_compare,
        'pcorr_samples': pcorr_samples,
    }
    return return_dict


def fitBayesian(confusion_matrix,indx=None,priors='half_normal',context_gen_version=1,fitvars=None,cores=None,
                conf_mat_method='normed',include_state_bias=True,experimental_set='humans',no_intercept=False,fit_init='adapt_diag'):
    if include_state_bias:
        fitvars_base = ['P','A','S']
    else:
        fitvars_base = ['P','A']
    if type(confusion_matrix) is np.ndarray:
        if confusion_matrix.ndim==3:
            if indx is None:
                indx = np.ones(confusion_matrix.shape[2]).astype(bool)
            confusion_matrix = confusion_matrix[:,:,indx]
        if context_gen_version==1:
            if fitvars is None:
                fitvars = ['D1','D2'] + fitvars_base
            confusion_mat_pred_att, confusion_mat_pred_proto, confusion_mat_pred_discrim_1att,\
                confusion_mat_pred_discrim_2att, confusion_mat_pred_state_bias, confusion_mat_pred_random =\
                contextGenModelConfMats(context_gen_version=1,experimental_set=experimental_set,method=conf_mat_method,
                                          return_random=True)
            indx_null = confusion_mat_pred_random.flatten()==0
            conf_mat_df=prepNovelContextGenModelFitDF(confusion_matrix,confusion_mat_pred_proto,confusion_mat_pred_att,
                        confusion_mat_pred_discrim_1att,confusion_mat_pred_state_bias=confusion_mat_pred_state_bias,
                        confusion_mat_pred_discrim_2att=confusion_mat_pred_discrim_2att,indx_null=indx_null)
        elif context_gen_version==2:
            if fitvars is None:
                fitvars = ['D'] + fitvars_base
            confusion_mat_pred_att, confusion_mat_pred_proto, confusion_mat_pred_discrim_1att,\
                      confusion_mat_pred_state_bias, confusion_mat_pred_random = contextGenModelConfMats(context_gen_version=2,
                                experimental_set=experimental_set,method=conf_mat_method,return_random=True)
            indx_null = confusion_mat_pred_random.flatten() == 0
            conf_mat_df = prepNovelContextGenModelFitDF(confusion_matrix,confusion_mat_pred_proto,confusion_mat_pred_att,
                confusion_mat_pred_discrim_1att,confusion_mat_pred_state_bias=confusion_mat_pred_state_bias,indx_null=indx_null)
    elif type(confusion_matrix) is pd.core.frame.DataFrame:
        conf_mat_df = confusion_matrix
    else:
        raise ValueError('Value passed for confusion_matrix must be either a numpy matrix or pandas dataframe')
    conf_mat_df = conf_mat_df.loc[:,['Observed']+[var for var in fitvars if var != '1']]
    if no_intercept:
        fitvars = ['0']+fitvars
    if priors == 'full_normal':
        model = bmb.Model(f"Observed ~ {'+'.join(fitvars)}", conf_mat_df)
    elif priors == 'half_normal':
        priors = {}
        for var in fitvars:
            priors[var] = bmb.Prior("HalfNormal")
        model = bmb.Model(f"Observed ~ {'+'.join(fitvars)}", conf_mat_df, priors=priors)
    try:
        fitted = model.fit(tune=2000, draws=2000, init=fit_init,target_accept=0.9,idata_kwargs={'log_likelihood': True},cores=cores)
    except:
        print('Initial fitting failed. Trying again using the advi initialization.')
        fitted = model.fit(tune=2000, draws=2000, init='advi',target_accept=0.9,idata_kwargs={'log_likelihood': True},cores=cores)
    return conf_mat_df, model, fitted


def predictorDroppingFit(conf_mat,model,fitted,priors='half_normal',version=2,conf_mat_method='normed',
                         experimental_set='humans',cores=None,fit_init='predictorDroppingFit'):
    # Get variables
    varnames = model.formula.split(' ')[-1].split('+')
    vals_list, fitvars_list = [], []
    for var in varnames:
        vals_list.append(var)
        fitvars_list.append([val for val in varnames if (val != var)])

    #Loop through and calculate fits
    models_dic = {'All': model}
    fitted_dict = {'All': fitted}
    for val, fitvars in zip(vals_list, fitvars_list):
        _, model_, fitted_ = fitBayesian(conf_mat, indx=None, priors=priors,conf_mat_method=conf_mat_method,cores=cores,
                                        context_gen_version=version, fitvars=fitvars,experimental_set=experimental_set,
                                        fit_init=fit_init)
        models_dic[f'No {val}'], fitted_dict[f'No {val}'] = model_, fitted_
    # Fit the Null model
    _, model_, fitted_ = fitBayesian(conf_mat, indx=None, priors=priors,conf_mat_method=conf_mat_method,cores=cores,
                                        context_gen_version=version, fitvars=['1'],experimental_set=experimental_set,
                                        fit_init=fit_init)
    models_dic['Null'], fitted_dict['Null'] = model_, fitted_
    
    return models_dic, fitted_dict


def calcBayesianCoefficients(conf_mat_df,model,fitted):
    # if 'D1' in conf_mat_df.columns:
    #     varnames=['D1','D2','P','A', 'S']
    # else:
    #     varnames=['D','P','A', 'S']
    varnames = model.formula.split(' ')[-1].split('+')
    samples = fitted.posterior
    pcorr_samples = partialCorrelationCoefficients(model,conf_mat_df,samples,varnames=varnames,dependant_var='Observed')

    return pcorr_samples


def partialCorrelationCoefficients(model, data, samples, varnames=['D', 'P', 'A'], dependant_var='Observed'):
    # For Bayesian model fitting, compute the needed statistics like R-squared when each predictor is response and all the
    # other predictors are the predictor.

    # x_matrix = common effects design matrix (excluding intercept/constant term)
    terms = [t for t in model.common_terms.values() if t.name != "Intercept"]
    x_matrix = [pd.DataFrame(x.data, columns=x.levels) for x in terms]
    x_matrix = pd.concat(x_matrix, axis=1)

    dm_statistics = {
        'r2_x': pd.Series(
            {
                x: sm.OLS(
                    endog=x_matrix[x],
                    exog=sm.add_constant(x_matrix.drop(x, axis=1))
                    if "Intercept" in model.term_names
                    else x_matrix.drop(x, axis=1),
                )
                    .fit()
                    .rsquared
                for x in list(x_matrix.columns)
            }
        ),
        'sigma_x': x_matrix.std(),
        'mean_x': x_matrix.mean(axis=0),
    }

    r2_x = dm_statistics['r2_x']
    sd_x = dm_statistics['sigma_x']
    r2_y = pd.Series([sm.OLS(endog=data[dependant_var],
                             exog=sm.add_constant(data[[p for p in varnames if p != x]])).fit().rsquared
                      for x in varnames], index=varnames)
    sd_y = data[dependant_var].std()

    slope_constant = (sd_x[varnames] / sd_y) * ((1 - r2_x[varnames]) / (1 - r2_y)) ** 0.5

    pcorr_samples = (samples[varnames] * slope_constant).stack(samples=("draw", "chain"))

    return pcorr_samples


def calcPartialCorrelationProbs(pcorr_samples):
    varnames = list(pcorr_samples.to_dict()['data_vars'].keys())

    indx_comp = np.array(list(itertools.product(np.arange(len(varnames)), repeat=2)))
    indx_comp = indx_comp[np.abs(indx_comp[:,0]-indx_comp[:,1])>0,:]

    prob_greater = {
        'Comparison': [],
        'Probability': []
    }

    for i in range(indx_comp.shape[0]):
        # NEED TO LOOK CLOSER AT THIS OUTPUT. SEEMS TO BE GIVING OPPOSITE RESULT OF WHAT'S EXPECTED
        prob_greater['Probability'].append((pcorr_samples[varnames[indx_comp[i,0]]]**2 > pcorr_samples[varnames[indx_comp[i,1]]]**2).mean().item())
        prob_greater['Comparison'].append(f'{varnames[indx_comp[i,0]]}>{varnames[indx_comp[i,1]]}')

    prob_greater_df = pd.DataFrame(prob_greater)

    return prob_greater_df


def prepNovelContextGenModelFitDF(confusion_matrix,confusion_mat_pred_proto,confusion_mat_pred_att,confusion_mat_pred_discrim_1att,
                    confusion_mat_pred_state_bias,confusion_mat_pred_discrim_2att=None,indx_null=None):
    if indx_null is None:
        indx_null = np.round(confusion_mat_pred_att,2).flatten()==0
    if confusion_matrix.ndim == 3:
        confusion_matrix_vec = (np.sum(confusion_matrix, axis=2) /
                            np.sum(np.sum(confusion_matrix, axis=2), axis=1)[:, None]).flatten()[indx_null < 1]
    else:
        confusion_matrix_vec = (confusion_matrix / np.sum(confusion_matrix, axis=1)[:, None]).flatten()[indx_null < 1]

    confusion_matrix_vec[np.isnan(confusion_matrix_vec)] = 0

    conf_mat_df = pd.DataFrame({
            'Observed': confusion_matrix_vec,
            'P': confusion_mat_pred_proto.flatten()[indx_null<1],
            'A': confusion_mat_pred_att.flatten()[indx_null<1],
            'S': confusion_mat_pred_state_bias.flatten()[indx_null<1],
            'I': np.ones(len(confusion_matrix_vec))*.3
        })
    if confusion_mat_pred_discrim_2att is None:
        conf_mat_df['D'] = confusion_mat_pred_discrim_1att.flatten()[indx_null<1]
    else:
        conf_mat_df['D1'] = confusion_mat_pred_discrim_1att.flatten()[indx_null<1]
        conf_mat_df['D2'] = confusion_mat_pred_discrim_2att.flatten()[indx_null<1]
    return conf_mat_df