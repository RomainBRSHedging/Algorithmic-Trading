# df handling and statistical analysis
import numpy as np
import yfinance as yf
import quantstats as qs
import datetime as dt

# df visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Optimization and allocation
from pypfopt.efficient_frontier import EfficientFrontier 
from pypfopt import risk_models
from pypfopt import expected_returns


####Market df#####
stocks_list = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'DIS', '^GSPC', '^FCHI', 'TSLA']
stocks = yf.download(stocks_list, start='2010-01-01', end=dt.datetime.now())
df = stocks.loc[:,'Close'].copy().dropna()
df2 = stocks.loc[:,'Adj Close'].copy().dropna()
#S&P500
sp500 = qs.utils.download_returns('^GSPC') 
sp500 = sp500.loc['2010-07-01':'2023-02-10'] 


####returns####
returns = df.pct_change(1).dropna()
returns2 = df2.pct_change(1).dropna()
sp500_returns = sp500.pct_change(1).dropna()
    #daily
daily_mean_returns = returns.mean()
daily_var_returns = returns.var()
daily_std_returns = returns.std()
    #annualy
annualy_mean_returns = daily_mean_returns*252
annualy_var_returns = daily_var_returns*252
annualy_std_returns = daily_std_returns*np.sqrt(252)
returns.plot(kind='hist', bins=200) #look like gaussian
plt.show()


##### Mean-Var analysis ######
mean_var = returns.describe().T.loc[:,['mean', 'std']]
mean_var['mean'] = mean_var['mean']*252
mean_var['std'] = mean_var['std']*np.sqrt(252)

    #Plot Risk/returns
mean_var.plot.scatter(x='std', y='mean')
for _ in mean_var.index:
    plt.annotate(_, xy= (mean_var.loc[_,'std']+0.002, mean_var.loc[_,'mean']+0.002) )
plt.show()

    ## Cov/Corr of the portfolio
cov_returns = returns.cov()
corr_returns = returns.corr()

    ## Heatmap correlations sotcks
sns.heatmap(corr_returns, cmap='Reds', annot=True)
plt.show()


#### Choose Stocks to invest in, with optimized weights ######

###### Portfolio allocation #######
µ = expected_returns.mean_historical_return(df2)
ß = risk_models.sample_cov(df2)

    # Markovic
ef = EfficientFrontier(µ,ß)
weights = ef.max_sharpe()
weights = ef.clean_weights()
weights_values = weights.values()
weights_list = list(weights_values)

# Portfolio returns
portfolio_returns = returns2.dot(weights_list)
portfolio_returns.plot()
plt.show()

### Plot Portfolio vs benchmark ### 
print(qs.reports.full(portfolio_returns, benchmark=sp500))
