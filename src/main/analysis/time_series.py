
def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = pn.rolling_mean(timeseries, window=12)
    rolstd = pn.rolling_std(timeseries, window=12)

    # Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pn.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print dfoutput





import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

results_tm = plt.hist(results_timeline, bins=24*7)
ts=results_tm[0]
ts_log = np.log(ts)
plt.plot(ts_log)

moving_avg = pn.rolling_mean(ts_log,12)
plt.plot(ts_log)
plt.plot(moving_avg, color='red')
plt.show()

ts_log_moving_avg_diff = ts_log - moving_avg

ts_log_moving_avg_diff = ts_log_moving_avg_diff[~np.isnan(ts_log_moving_avg_diff)]
test_stationarity(ts_log_moving_avg_diff)



expwighted_avg = pn.ewma(ts_log, halflife=12)
plt.plot(ts_log)
plt.plot(expwighted_avg, color='red')
plt.show()

ts_log_ewma_diff = ts_log - expwighted_avg
test_stationarity(ts_log_ewma_diff)

ts_log = pn.TimeSeries(ts_log)
ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)

ts_log_diff.dropna(inplace=True)
#test_stationarity(ts_log_diff)

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(pn.Series(ts_log[0], index=range(len(ts_log))))

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()



# hay que asignar a cada tweet un bin o una hora ya modificada para que se aplicamos el historama sobre la nueva
# hora tengamos una distribucion uniforme


