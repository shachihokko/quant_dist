import pandas as pd
import numpy as np
from scipy.stats import t
from scipy.stats import norm
from scipy.optimize import minimize
import math

###############################################################################
#%% Probability density function of a TSkew distribution
###############################################################################
def tskew_pdf(x, df, loc, scale, skew):
    """
    Density function of the tskew distribution
    Based on the formula in Giot and Laurent (JAE 2003 pp. 650)
    - x = the value to evaluate
    - df: degrees of freedom (>1)
    - location: mean of the distribution
    - scale: standard deviation of the distribution
    - skew: skewness parameter (>0, if ==1: no skew, <1: left skew, >1 right)

    NB: I had to parametrize the formula differently to get consistent results

    """
    cons = (2/(skew + (1/skew)))/scale
    norm_x =  (x-loc)/scale
    if x < loc :
        pdf = cons*t.pdf(skew*norm_x, df, loc=0, scale=1) # Symmetric t pdf
    elif x >= loc:
        pdf = cons*t.pdf(norm_x/skew, df, loc=0, scale=1) # Symmetric t pdf
    else:
        raise ValueError('Incorrect parameters')

    return(pdf)

###############################################################################
#%% Percentage point function (quantile function) of a TSkew distribution
###############################################################################
def tskew_ppf(tau, df, loc, scale, skew):
    """
    Quantile function of the tskew distribution
    Based on the formula in Giot and Laurent (JAE 2003 pp. 650)
    - tau = the quantile
    - df: degrees of freedom (>1)
    - location: mean of the distribution
    - scale: standard deviation of the distribution (>0)
    - skew: skewness parameter (>0, if ==1: no skew, <1: left skew, >1 right)

    NB: I had to parametrize the formula differently (was wrong in their paper)
    """

    threshold = 1/(1+np.power(skew,2))
    if tau < threshold:
        adj_tau = (tau/2)*(1+np.power(skew,2))
        non_stand_quantile = (1/skew)*t.ppf(adj_tau, df=df, loc=0, scale=1)
    elif tau >= threshold:
        adj_tau = ((1-tau)/2)*(1+(1/np.power(skew,2)))
        non_stand_quantile = -skew*t.ppf(adj_tau, df=df, loc=0, scale=1)
    else:
        raise ValueError('Parameters misspecified')

    quantile = loc + (non_stand_quantile*scale) # Pay attention to this one !

    return(quantile)


###############################################################################
#%% Cumulative distribution of a TSkew distribution:
###############################################################################
def tskew_cdf(x, df, loc, scale, skew):
    """
    Density function of the tskew distribution
    Based on the formula in Giot and Laurent (JAE 2003 pp. 650)
    and Lambert and Laurent (2002) pp. 10
    - x = real value on the support to evaluate
    - df: degrees of freedom (>1)
    - location: mean of the distribution
    - skew: skewness parameter (>0, if ==1: no skew, <1: left skew, >1 right)

    NB: I had to parametrize it differently in order to get consistent results

    """

    sk2 = np.power(skew, 2); inv_sk2 = 1/sk2
    norm_x1 = (x-loc)/scale
    #norm_x2 = x-(loc/scale)
    if x < loc:
        # t.cdf() is the symmetric t cdf
        cdf = (2/(1+sk2))*t.cdf(skew*norm_x1, df, loc=0, scale=1)
    elif x >= loc:
        cdf = 1 - (2/(1+inv_sk2))*t.cdf(-norm_x1/skew, df, loc=0, scale=1)
    else:
        raise ValueError('Incorrect parameters')

    return(cdf)

###############################################################################
#%% get mean value of t-skew
###############################################################################
def tskew_mean(df, loc, scale, skew):
    """
    Note by C. Wang
    Formula from Equation 5 of
    On Bayesian Modeling of Fat Tails and Skewness
    Carmen Fernandez and Mark F. J. Stee, 1998 JASA
    """
    cons1=skew-1/skew
    Mr=math.gamma((df+1)/2)/(math.sqrt(df*math.pi)*math.gamma(df/2))*2*df*scale
    return cons1*Mr+loc


###############################################################################
#%% calculate the distance between ideal and empirical
###############################################################################
def tskew_distance(quantile_list, cond_quant,
                   df, loc, scale, skew):
    """ Return the distance between theoretical and actual quantiles"""

    def tskew_tau(tau):
        """ Function which only depends on a given tau """
        return(tskew_ppf(tau, df=df, loc=loc, scale=scale, skew=skew))

    tskew_ppf_vectorized = np.vectorize(tskew_tau, otypes=[np.float])

    theoretical_quant = tskew_ppf_vectorized(quantile_list)

    diff = np.subtract(theoretical_quant, cond_quant)
    diff2 = np.power(diff,2)
    msse = np.sum(diff2)

    loc_tskew=loc
    for i in range(len(quantile_list)):
        if quantile_list[i]==0.25:
            lowq=cond_quant[i]
        if quantile_list[i]==0.75:
            highq=cond_quant[i]
    alpha=10
    if loc_tskew<=highq and loc_tskew>=lowq:
        penalty=0
    else:
        penalty=alpha*min((lowq-loc_tskew)**2,(highq-loc_tskew)**2)
    mssepen=msse+penalty
    return(mssepen)

###############################################################################
#%% Optimal TSkew fit based on a set of conditional quantiles and a location
###############################################################################
def tskew_fit(conditional_quantiles, fitparams):
    """
    Optimal TSkew fit based on a set of conditional quantiles and a location
    Inputs:
        - conditional_quantiles: quantiles & conditional value
        - loc: location. Can be estimated as a conditional mean via OLS
    Output:
        - A dictionary with optimal scale and skewness, as well as df and loc
    """

    ######################
    #Generate Parameters##
    ######################

    ## Interquartile range (proxy for volatility)
    IQR = np.absolute(conditional_quantiles[0.75] - conditional_quantiles[0.25])

    # Good lower bound approximation
    scale_down = np.sqrt(IQR)/2 +0.1
    scale_up = IQR/1.63 + 0.2 # When skew=1, variance exactly = IQR/1.63

    # Default lower bound approximation
    skew_low = 0.1
    skew_high = 3

    x0_f = [IQR/1.63 + 0.1, 1]
    loc=conditional_quantiles[0.5]
    o_df = 2

    ## Two values optimizer: on both conditional variance and skewness
    def mult_obj_distance(x): # x is a vector
      """ Multiple parameters estimation """
      ## Unpack the vector
      scale = x[0]
      skew = x[1]

      # Run the optimizer
      obj = tskew_distance(quantile_list=quantile_list,
                           cond_quant=cond_quant,
                           df=o_df, loc=cond_mean, scale=scale, skew=skew)
      return(obj)

    ## Run the optimizer
    locs = loc+0.5
    cond_mean=0
    cdmeanmax=loc+10
    cdmeanmin=loc-10

    maxit=0
    while maxit<100 and abs(locs-loc)>0.00001:

      cond_mean=(cdmeanmin+cdmeanmax)/2


      # Fix the boundaries to avoid degenerative distributions
      bnds_f = ((scale_down, scale_up), (skew_low , skew_high))
      res = minimize(mult_obj_distance, x0=x0_f,
                     bounds=bnds_f, method='SLSQP',
                     options={'maxiter':1000,  'ftol': 1e-04, 'eps': 1.5e-06})

      o_scale, o_skew  = res.x
      locs=cond_mean
      if locs>loc:
        cdmeanmax=cond_mean
      else:
        cdmeanmin=cond_mean
      maxit+=1

      ## Package the results into a dictionary
    fit_dict = {'loc': float("{:.4f}".format(cond_mean)),
                'df': int(o_df),
                'scale': float("{:.4f}".format(o_scale)),
                'skew': float("{:.4f}".format(o_skew))}

    return(fit_dict)

###############################################################################
#%% make pdf and cdf
###############################################################################

def gen_skewt(fitdate,fitparam,cond_quant,horizon,freq,olsmean):

    if fitparam['fittype']=='T-skew':
        loc=0
        loc=cond_quant [0.5]

        tsfit=tskew_fit(cond_quant,fitparam)

        loc=tsfit['loc']
        min_v = loc-8
        max_v = loc+8
        while tskew_cdf(min_v+1, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])>0.05:
            min_v-=1

        x_list = [x for x in np.arange(min_v,max_v,0.05)]
        yvals= [tskew_pdf(z, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew']) for z in x_list]
        ycdf = [tskew_cdf(z, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew']) for z in x_list]
        yzero=tskew_cdf(0, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])

        tmp_dic={'Tskew_PDF_x':x_list,'Tskew_PDF_y':yvals,'Tskew_CDF':ycdf}
        dfpdf=pd.DataFrame(tmp_dic)

        for i,y in enumerate(ycdf):
            q5loc=i
            if y>0.05:
                break
        for i,y in enumerate(ycdf):
            q10loc=i
            if y>0.1:
                break
        for i,x in enumerate(x_list):
            zerog=i
            if x<=0 and (i==len(x_list)-1 or x_list[i+1]>0):
                break

        xq5=tskew_ppf(0.05, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
        yq5= tskew_pdf(xq5, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
        ycq5= tskew_cdf(xq5, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])

        meanx=tskew_mean(df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])

        modx=tsfit['loc']
        mody=tskew_pdf(loc, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])

        medx=tskew_ppf(0.5, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
        medy=tskew_pdf(medx, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])
        meany=tskew_pdf(meanx, df=tsfit['df'], loc=tsfit['loc'], scale=tsfit['scale'], skew=tsfit['skew'])

        res=[]
        res.append(['Date of input',fitdate])
        res.append(['Horizon forward',horizon])
        res.append(['Conditional mode',float("{:.4f}".format(loc))])
        res.append(['Conditional median',float("{:.4f}".format(medx))])
        res.append(['Conditional mean',float("{:.4f}".format(meanx))])
        res.append(['GaR5%',float("{:.4f}".format(xq5))])
        res.append(['Growth below 0 probablity',float("{:.4f}".format(yzero))])
        res.append(['Skewness',tsfit['skew']])
        res.append(['Scale',tsfit['scale']])

        cqlist=[['Tau','Cond_quant']]

        qlist=list(cond_quant.keys())
        qlist.sort()
        for q in qlist:
            cqlist.append([q,cond_quant[q]])

        return res,cqlist,dfpdf
