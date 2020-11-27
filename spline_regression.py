import numpy as np
import scipy.interpolate as intrp
import scipy.stats as stats
import scipy.linalg as la
from sklearn.linear_model import LinearRegression as OLS

#%%
def spline_covariates(x1, x2, y, num_basis=10, compute_before=False, use_residuals=False, sandwich=False):
    """
    A semi-parametric test for whether x1 predicts y conditional on x2.
    
    In:
        x1 (n_sample, ): the predictor variable (e.g. in/out phrase)
        x2 (n_sample, ): the confounding variable (e.g. PMI)
        y (n_sample, ): the predicted variable (e.g. CCA)
        num_basis (default 10): maximum number of b-splines 
        compute_before (default False): whether to do the regular regression
        use_residuals (default False): regress on the residuals?
        sandwich (default False): Estimate nonlinearity-robust standard errors? 
    Out:
        (beta_x1, p_x1): regression coef. w/ associated p-value 
                        of x1 under the model y ~ x1 (if compute_before=True)
        (beta_x1_ctrl, p_x1_ctrl): coef. and p-value of x1, but for y ~ x1 + f(x2)
    """

    # without confounder'
    if compute_before:
        ols = MyOLS(fit_intercept=True)
        ols.fit(x1[...,None],y)
    
    # with confounder
    these_knots = np.unique(np.quantile(x2,np.linspace(0,1,num_basis)))
    
    # generate spline bases
    buff_0 = np.repeat(these_knots.min(),3)
    buff_1 = np.repeat(these_knots.max(),3)
    numpyknots = np.concatenate((buff_0,these_knots,buff_1)) # because??
    bases = np.zeros((x2.shape[0], len(these_knots)+1))
    bases[:,0] = x2
    for i in range(1,len(these_knots)+1):
        bases[:,i] = intrp.BSpline(numpyknots, 
                                   (np.arange(len(these_knots)+2)==i).astype(float), 
                                   3, extrapolate=False)(x2)
    
    if use_residuals:
        # Fit (Y - E[Y|x2]) ~ w*(x1 - E[x1|x2])
        # this is equivalent but less efficient -- just use for sanity check
        ols2 = MyOLS(fit_intercept=True)
        ols2.fit(bases,y)
        e_y = ols2.predict(bases)
        ols2.fit(bases,x1[:,None])
        e_x = ols2.predict(bases)
        # print(y-e_y)
        # print(x1-e_x)
        ols2.fit(x1[:,None], y-e_y)
    else:
        # concatenate data
        X = np.append(x1[:,None], bases, axis=1)
        ols2 = MyOLS(fit_intercept=True)
        ols2.fit(X,y)

    # if sandwich:
    #     # estimate var(beta) by (X'X)^-1 * X'DX * (X'X)^-1
    #     y_lin = y - X[:,1:]@ols2.coef_[1:] - ols2.intercept_
    #     # D = np.diag((y_lin - X[:,0])**2) # squared residuals
    #     errs = (y_lin - ols2.coef_[0]*X[:,0])**2
    #     # bread = la.inv(X[:,:1].T@X[:,:1])
    #     # meat = X[:,:1].T@D@X[:,:1]
    #     bread = 1/np.sum(X[:,0]**2)
    #     meat = np.sum(errs*X[:,0]**2)
    #     stand_err = np.sqrt(bread*meat*bread) 
    #     return stand_err, ols2.se

    if compute_before:
        # return  (ols.coef_[0], ols.p[0,0]), (ols2.coef_[0], ols2.p[0,0])
        return ols.statistics(st_err=sandwich, these_regs=0), ols2.statistics(st_err=sandwich, these_regs=0)
    else:
        return ols2.statistics(st_err=sandwich, these_regs=0)


#%% helper class for computing p values
class MyOLS(OLS):
    """
    Taken from https://gist.github.com/brentp
    
    LinearRegression class after sklearn's, but calculate t-statistics
    and p-values for model coefficients (betas).
    Additional attributes available after .fit()
    are `t` and `p` which are of the shape (y.shape[1], X.shape[1])
    which is (n_features, n_coefs)
    This class sets the intercept to 0 by default, since usually we include it
    in X.
    """

    def __init__(self, *args, **kwargs):
        if not "fit_intercept" in kwargs:
            kwargs['fit_intercept'] = False
        super(MyOLS, self)\
                .__init__(*args, **kwargs)

    def fit(self, X, y, n_jobs=1):
        self = super(MyOLS, self).fit(X, y, n_jobs)

        res = (y-self.predict(X))**2

        # sum squared error
        sse = np.sum(res, axis=0, keepdims=True) / float(X.shape[0] - X.shape[1])
        
        # homoskedastic standard error
        se = np.array([
            np.sqrt(np.diagonal(sse[i] * np.linalg.inv(np.dot(X.T, X))))
                                                    for i in range(sse.shape[0])
                    ])

        self.t = self.coef_ / se
        self.p = 2 * (1 - stats.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1]))
        # return self

        # nonlinearity-robust standard errors (sandwich)
        bread = la.inv(X.T@X)
        meat = X.T@np.diag(res)@X
        self.se = np.sqrt(np.diag(bread@meat@bread))

    def statistics(self, pval=True, st_err=True, these_regs=None):
        if these_regs is None:
            these_regs = np.arange(len(self.coef_))
        outputs = (self.coef_[these_regs],)
        if pval:
            outputs += (self.p[these_regs,these_regs],)
        if st_err:
            outputs += (self.se[these_regs],)

        return outputs

