import warnings
import numpy as np
from scipy.stats import weibull_min as W
from scipy.stats import genextreme as GEV
from scipy.optimize import fmin
import pandas as pd
from functools import partial


def W2_NS_fit(x, covar=None, shape_var = None, scale_var = None):
    

    """
    
    Fit a non stationary Weibull distribution to the data x. 

    Parameters:
    x: pandas.Series 
        time series of the variable to fit

    covar: pandas.DataFrame
        DataFrame containing the covariates. Please note that the covariates must be standardized ex: (x - x.mean())/x.std()

    shape_var: list of str
        list of the names of the columns of covar that will be used to fit the shape parameter.
    
    scale_var: list of str
        list of the names of the columns of covar that will be used to fit the scale parameter.
    
    Returns:
    param_dict: dict
        a dictionary containing the parameters of the non stationary distribution and the aic of the fit. 
        a0 + a1*x1 + a2*x2 + ... + an*len(scale_var); b0 + b1*x1 + b2*x2 + ... + bn*len(shape_var)
        a0 and b0 should be close to the parameters of the stationary distribution.

    aic: dict
        a dictionary containing the aic of the fit.

    Note: 
    only linear combination of the covariates are used to fit the parameters. None linear effet can be added by adding interaction terms to the covariates. 
    if no covariate is provided, the stationary distribution will be fitted.
    if shape_var is None and scale_var is None the stationary distribution will be fitted with a warning if covar is provided.
    the optimal parameters are found using the fmin function from scipy.optimize.


    """
    
    
    if x.eq(0).sum() > 0:
        warnings.warn("x contains zero values. The log-likelihood will be infinite. Please remove the zero values.")
    
    if not isinstance(covar, pd.DataFrame) and covar is not None:
        raise ValueError("covar must be a pandas DataFrame")
        
    if isinstance(scale_var, str|int|tuple|list) :
         scale_var = pd.Series(scale_var)
    if isinstance(shape_var, str|int|tuple|list) :
         shape_var = pd.Series(shape_var)


    if np.any(shape_var.astype('str').values != 'None') and shape_var.isin(covar.columns).sum() != len(shape_var):
            raise ValueError("shape_var must be columns names of covar")
    if np.any(scale_var.astype('str').values != 'None') and scale_var.isin(covar.columns).sum() != len(scale_var):
            raise ValueError("scale_var must be columns names of covar")
        
    #fit a stationary distribution
    b, _, a = W.fit(x, floc=0)


    if covar is None or (shape_var is None and scale_var is None):
        #if no covariate is provided, return the parameters of the stationary distribution
        if covar is not None:
            warnings.warn("covar is provided but shape_var and scale_var are None. The stationary distribution will be fitted.")
        
        _aic = 4 - 2*W.logpdf(x, scale=a, c=b, loc=0).sum()
      
        return ({'a0': a, 'b0': b}, {'aic': _aic})
    
    if np.any(scale_var.astype('str').values == 'None'):
         sc_v = []
    else:
        sc_v = scale_var
    if np.any(shape_var.astype('str').values == 'None'):
        sh_v = []
    else:
        sh_v = shape_var

    def likelihood(param, args):

        y = args[0]

        X = args[1]

        if len(sc_v) == 0:
             param_a = np.array([param[0]]).reshape(-1, 1)
             X_a = np.ones(X.shape[0]).reshape(-1, 1)
             
        else:
            param_a = param[0:len(sc_v)+1]
         
            X_a = X[sc_v].values.reshape(-1, len(sc_v))
            #X_scale = ((X_scale - X_scale.mean(axis=0))/X_scale.std(axis=0))
            X_a = np.hstack([np.ones(X_a.shape[0]).reshape(-1, 1), X_a])


        if len(sh_v) == 0:
            param_b = np.array([param[-1]]).reshape(-1, 1)
            X_b = np.ones(X.shape[0]).reshape(-1, 1)
        else:
            param_b = param[-len(sh_v)-1:]
            X_b = X[sh_v].values.reshape(-1, len(sh_v))
            #X_shape= ((X_shape - X_shape.mean(axis=0))/X_shape.std(axis=0))
            X_b = np.hstack([np.ones(X_b.shape[0]).reshape(-1, 1), X_b])    

        tol_round = 8
        a_vec = (X_a@param_a).round(tol_round)
        
        b_vec = (X_b@param_b).round(tol_round)
        return - np.sum(W.logpdf(y, scale=a_vec.reshape(-1, ), c=b_vec.reshape(-1, ), loc=0))
    
    
            
    param0 = np.array([a, *[0 for _ in range(len(sc_v))], b, *[0 for _ in range(len(sh_v))]])
    res, fopt, *_ =  fmin(partial(likelihood, args=(x, covar)), x0 =  param0, disp = False, full_output=True)

    _aic = 2*len(res) + 2*fopt
    p1 = {'a0':res[0]}
    p2 = {f'a{i}':res[i] for i in range(1, len(sc_v)+1)}
    p3 = {'b0':res[len(sc_v) + 1]}
    p4 = {f'b{i - (len(sc_v)+1)}':res[i] for i in range(len(sc_v)+2, len(res))}
    param_dict = p1 | p2 | p3 | p4


    return param_dict , {'aic': _aic}
        
def W3_NS_fit(x, covar=None, k_var = None, sg_var = None, mu_var = None, **kwargs):
    

    """
    
    Fit a non stationary 3 parameters Weibull distribution to the data x. 

    Parameters:
    x: pandas.Series 
        time series of the variable to fit

    covar: pandas.DataFrame
        DataFrame containing the covariates. Please note that the covariates must be standardized ex: (x - x.mean())/x.std()

    k_var: str | list of str 
        list of the names of the columns of covar that will be used to fit the shape parameter (k).
    
    sg_var: str | list of str 
        list of the names of the columns of covar that will be used to fit the scale parameter (sigma).
    
    mu_var: str | list of str
        list of the names of the columns of covar that will be used to fit the location parameter (mu).
    
    Returns:
    param_dict: dict
        a dictionary containing the parameters of the non stationary distribution and the aic of the fit. 
        ex: mu0 + mu1*x1 + mu2*x2 + ... + mun* xn when n = len(mu_var);
        mu0, k0 and sigma0 should be close to the parameters of the stationary distribution.

    aic: dict
        a dictionary containing the aic of the fit.

    Note: 
    only linear combination of the covariates are used to fit the parameters. None linear effet can be added by adding interaction terms to the covariates. 
    if no covariate is provided, the stationary distribution will be fitted.
    if mu_var, sg_var and k_var are None, the stationary distribution will be fitted with a warning if covar is provided.
    the optimal parameters are found using the fmin function from scipy.optimize.

    Additional parameters:
    **kwargs:
        additional parameters to pass to the fit method of the scipy.stats.weibull_min distribution. ex: floc=0 => the distribution is truncated at 0.


    """
    #check the inputs

    if isinstance(mu_var, tuple|list) and len(mu_var) == 1:
        mu_var = mu_var[0]
    if isinstance(sg_var, tuple|list) and len(sg_var) == 1:
        sg_var = sg_var[0]
    if isinstance(k_var, tuple|list) and len(k_var) == 1:
        k_var = k_var[0]
        
         
    
    if not isinstance(covar, pd.DataFrame) and covar is not None:
        raise ValueError("covar must be a pandas DataFrame")
        
    if isinstance(mu_var, str|int|tuple|list) :
         mu_var = pd.Series(mu_var)

    if isinstance(sg_var, str|int|tuple|list) :
         sg_var = pd.Series(sg_var)

    if isinstance(k_var, str|int|tuple|list) :
         k_var = pd.Series(k_var)

    #check if the covariates are in the DataFrame covar
    #np.any(shape_var.astype('str').values != 'None')
    if np.any(k_var.astype('str').values != 'None') and k_var.isin(covar.columns).sum() != len(k_var):
            raise ValueError("k_var must be columns names of covar")
    if np.any(sg_var.astype('str').values != 'None') and sg_var.isin(covar.columns).sum() != len(sg_var):
            raise ValueError("sg_var must be columns names of covar")
    if np.any(mu_var.astype('str').values != 'None') and mu_var.isin(covar.columns).sum() != len(mu_var):
            raise ValueError("mu_var must be columns names of covar")
        
    #fit a stationary distribution: this is used to initialize the parameters of the non stationary distribution
    k, mu, sigma = W.fit(x, **kwargs)


    if covar is None or (k_var is None and sg_var is None and mu_var is None):
        #if no covariate is provided, return the parameters of the stationary distribution
        if covar is not None:
            warnings.warn("covar is provided but k_var, sg_var and mu_var are None. The stationary distribution will be fitted.")
        
        _aic = 6 - 2*W.logpdf(x, scale=sigma, c=k, loc=mu).sum()
      
        return ({'sigma0': sigma, 'k0': k, 'mu0':mu}, {'aic': _aic})
    
    if np.any(sg_var.astype('str').values == 'None'):
         sg_v = []
    else:
        sg_v = sg_var

    if np.any(k_var.astype('str').values == 'None'):
        k_v = []
    else:
        k_v = k_var
    if np.any(mu_var.astype('str').values == 'None'):
        mu_v = []
    else:
        mu_v = mu_var

    def likelihood(param, args):

        """"
        log likelihood function of the non stationary distribution
        param: list
            list of the parameters of the non stationary distribution
        args: tuple
            list of the arguments of the likelihood function (x, and covariates)
        
        """
        
        y = args[0]

        X = args[1]

       

        if len(mu_v) == 0:
             param_mu = np.array([param[0]]).reshape(-1, 1)
             X_mu = np.ones(X.shape[0]).reshape(-1, 1)   
        else:
            param_mu = param[0:len(mu_v)+1]
            X_mu = X[mu_v].values.reshape(-1, len(mu_v))
            #X_scale = ((X_scale - X_scale.mean(axis=0))/X_scale.std(axis=0))
            X_mu = np.hstack([np.ones(X_mu.shape[0]).reshape(-1, 1), X_mu])


        if len(sg_v) == 0:
            param_sigma = np.array([param[len(mu_v) + 1]]).reshape(-1, 1)
            X_sigma = np.ones(X.shape[0]).reshape(-1, 1)
        else:
            param_sigma = param[len(mu_v)+1:len(mu_v)+len(sg_v)+2]
            X_sigma = X[sg_v].values.reshape(-1, len(sg_v))
            #X_shape= ((X_shape - X_shape.mean(axis=0))/X_shape.std(axis=0))
            X_sigma = np.hstack([np.ones(X_sigma.shape[0]).reshape(-1, 1), X_sigma])

        if len(k_v) == 0:
            param_k = np.array([param[-1]]).reshape(-1, 1)
            X_k = np.ones(X.shape[0]).reshape(-1, 1)
        else:
            param_k = param[-len(k_v)-1:]
            X_k = X[k_v].values.reshape(-1, len(k_v))
            #X_shape= ((X_shape - X_shape.mean(axis=0))/X_shape.std(axis=0))
            X_k = np.hstack([np.ones(X_k.shape[0]).reshape(-1, 1), X_k])   
        
        tol_round = 5

        mu_vec = (X_mu@param_mu).round(tol_round)
        sigma_vec = (X_sigma@param_sigma).round(tol_round)
        k_vec = (X_k@param_k).round(tol_round)

        return - np.sum(W.logpdf(y, scale=sigma_vec.reshape(-1, ), c=k_vec.reshape(-1, ), loc=mu_vec.reshape(-1,)))
    
    
            
    param0 = np.array([mu, *[0 for _ in range(len(mu_v))], 
                       sigma, *[0 for _ in range(len(sg_v))], 
                       k, *[0 for _ in range(len(k_v))]])
    
    res, fopt, *_ =  fmin(partial(likelihood, args=(x, covar)), x0 =  param0, disp = False, full_output=True)
    
    #####
    #options = dict(disp = False, xatol=1e-3, fatol=1e-3)
    #res=  minimize(partial(likelihood, args=(x, covar)), method='Nelder-Mead', x0 =  param0, options=options, tol=1e-3)
    #fopt = res.fun
    #res = res.x
    #####
    _aic = 2*len(res) + 2*fopt
    p1 = {'mu0':res[0]}
    p2 = {f'mu{i}':res[i] for i in range(1, len(mu_v)+1)}
    p3 = {'sigma0':res[len(mu_v) + 1]}
    p4 = {f'sigma{i - (len(mu_v)+1)}':res[i] for i in range(len(mu_v)+2, len(mu_v)+2 + len(sg_v))}
    p5 = {'k0':res[-len(k_v)-1]}
    p6 = {f'k{(i + len(k_v)+1) }':res[i] for i in range(-len(k_v), 0, 1)}
    param_dict = p1 | p2 | p3 | p4 | p5 | p6


    return param_dict , {'aic': _aic}
        

def GEV_NS_fit(x, covar=None, k_var = None, sg_var = None, mu_var = None, **kwargs):
    

    """
    
    Fit a non stationary 3 parameters GEV distribution to the data x. 

    Parameters:
    x: pandas.Series 
        time series of the variable to fit

    covar: pandas.DataFrame
        DataFrame containing the covariates. Please note that the covariates must be standardized ex: (x - x.mean())/x.std()

    k_var: str | list of str 
        list of the names of the columns of covar that will be used to fit the shape parameter (k).
    
    sg_var: str | list of str 
        list of the names of the columns of covar that will be used to fit the scale parameter (sigma).
    
    mu_var: str | list of str
        list of the names of the columns of covar that will be used to fit the location parameter (mu).
    
    Returns:
    param_dict: dict
        a dictionary containing the parameters of the non stationary distribution and the aic of the fit. 
        ex: mu0 + mu1*x1 + mu2*x2 + ... + mun* xn when n = len(mu_var);
        mu0, k0 and sigma0 should be close to the parameters of the stationary distribution.

    aic: dict
        a dictionary containing the aic of the fit.

    Note: 
    only linear combination of the covariates are used to fit the parameters. None linear effet can be added by adding interaction terms to the covariates. 
    if no covariate is provided, the stationary distribution will be fitted.
    if mu_var, sg_var and k_var are None, the stationary distribution will be fitted with a warning if covar is provided.
    the optimal parameters are found using the fmin function from scipy.optimize.

    Additional parameters:
    **kwargs:
        additional parameters to pass to the fit method of the scipy.stats.genextreme distribution. ex: floc=0 => the distribution is truncated at 0.


    """
    #check the inputs

    if isinstance(mu_var, tuple|list) and len(mu_var) == 1:
        mu_var = mu_var[0]
    if isinstance(sg_var, tuple|list) and len(sg_var) == 1:
        sg_var = sg_var[0]
    if isinstance(k_var, tuple|list) and len(k_var) == 1:
        k_var = k_var[0]
        
         
    
    if not isinstance(covar, pd.DataFrame) and covar is not None:
        raise ValueError("covar must be a pandas DataFrame")
        
    if isinstance(mu_var, str|int|tuple|list) :
         mu_var = pd.Series(mu_var)

    if isinstance(sg_var, str|int|tuple|list) :
         sg_var = pd.Series(sg_var)

    if isinstance(k_var, str|int|tuple|list) :
         k_var = pd.Series(k_var)

    #check if the covariates are in the DataFrame covar
    if np.any(k_var.astype('str').values != 'None') and k_var.isin(covar.columns).sum() != len(k_var):
            raise ValueError("k_var must be columns names of covar")
    if np.any(sg_var.astype('str').values != 'None') and sg_var.isin(covar.columns).sum() != len(sg_var):
            raise ValueError("sg_var must be columns names of covar")
    if np.any(mu_var.astype('str').values != 'None') and mu_var.isin(covar.columns).sum() != len(mu_var):
            raise ValueError("mu_var must be columns names of covar")
        
    #fit a stationary distribution: this is used to initialize the parameters of the non stationary distribution
    k, mu, sigma = GEV.fit(x, **kwargs)


    if covar is None or (k_var is None and sg_var is None and mu_var is None):
        #if no covariate is provided, return the parameters of the stationary distribution
        if covar is not None:
            warnings.warn("covar is provided but k_var, sg_var and mu_var are None. The stationary distribution will be fitted.")
        
        loglik = GEV.logpdf(x, scale=sigma, c=k, loc=mu).sum()
      
        return ({'sigma0': sigma, 'k0': k, 'mu0':mu}, {'loglik': loglik})
    
    if np.any(sg_var.astype('str').values == 'None'):
         sg_v = []
    else:
        sg_v = sg_var

    if np.any(k_var.astype('str').values == 'None'):
        k_v = []
    else:
        k_v = k_var
    if np.any(mu_var.astype('str').values == 'None'):
        mu_v = []
    else:
        mu_v = mu_var

    def likelihood(param, args):

        """"
        log likelihood function of the non stationary distribution
        param: list
            list of the parameters of the non stationary distribution
        args: tuple
            list of the arguments of the likelihood function (x, and covariates)
        
        """
        
        y = args[0]

        X = args[1]

       

        if len(mu_v) == 0:
             param_mu = np.array([param[0]]).reshape(-1, 1)
             X_mu = np.ones(X.shape[0]).reshape(-1, 1)   
        else:
            param_mu = param[0:len(mu_v)+1]
            X_mu = X[mu_v].values.reshape(-1, len(mu_v))
            X_mu = np.hstack([np.ones(X_mu.shape[0]).reshape(-1, 1), X_mu])


        if len(sg_v) == 0:
            param_sigma = np.array([param[len(mu_v) + 1]]).reshape(-1, 1)
            X_sigma = np.ones(X.shape[0]).reshape(-1, 1)
        else:
            param_sigma = param[len(mu_v)+1:len(mu_v)+len(sg_v)+2]
            X_sigma = X[sg_v].values.reshape(-1, len(sg_v))
        
            X_sigma = np.hstack([np.ones(X_sigma.shape[0]).reshape(-1, 1), X_sigma])

        if len(k_v) == 0:
            param_k = np.array([param[-1]]).reshape(-1, 1)
            X_k = np.ones(X.shape[0]).reshape(-1, 1)
        else:
            param_k = param[-len(k_v)-1:]
            X_k = X[k_v].values.reshape(-1, len(k_v))
            X_k = np.hstack([np.ones(X_k.shape[0]).reshape(-1, 1), X_k])   
         
        mu_vec = X_mu@param_mu
        sigma_vec = X_sigma@param_sigma
        k_vec = X_k@param_k

        return - np.sum(GEV.logpdf(y, scale=sigma_vec.reshape(-1, ), c=k_vec.reshape(-1, ), loc=mu_vec.reshape(-1,)))
    
    
            
    param0 = np.array([mu, *[0 for _ in range(len(mu_v))], 
                       sigma, *[0 for _ in range(len(sg_v))], 
                       k, *[0 for _ in range(len(k_v))]])
    
    res, fopt, *_ =  fmin(partial(likelihood, args=(x, covar)), x0 =  param0, disp = False, full_output=True)

    loglik = -fopt
    p1 = {'mu0':res[0]}
    p2 = {f'mu{i}':res[i] for i in range(1, len(mu_v)+1)}
    p3 = {'sigma0':res[len(mu_v) + 1]}
    p4 = {f'sigma{i - (len(mu_v)+1)}':res[i] for i in range(len(mu_v)+2, len(mu_v)+2 + len(sg_v))}
    p5 = {'k0':res[-len(k_v)-1]}
    p6 = {f'k{(i + len(k_v)+1) }':res[i] for i in range(-len(k_v), 0, 1)}
    param_dict = p1 | p2 | p3 | p4 | p5 | p6


    return param_dict , {'loglik': loglik}
        

