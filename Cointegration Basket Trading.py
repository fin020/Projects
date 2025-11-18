import os
import requests as r
import pandas as pd 
from pandas import Series, DataFrame
import numpy as np
from datetime import datetime, timedelta
import csv
from io import StringIO
from statsmodels.tsa.vector_ar.vecm import coint_johansen  #type: ignore
from statsmodels.tsa.stattools import adfuller #type: ignore
from typing import Any
from scipy.stats import linregress  #type: ignore
import matplotlib.pyplot as plt
import optuna

#Fetching data: 
CACHE_FILE = "Cached_data.csv"
EXPIRY_HOURS = 12
API: str|None = os.getenv('AV_API_KEY')

def is_cache_valid(filepath: str, expiry_hours: int=EXPIRY_HOURS): 
    if not os.path.exists(filepath): 
        return False
    modified_time = datetime.fromtimestamp(os.path.getmtime(filepath))
    return datetime.now() - modified_time < timedelta(hours=expiry_hours)

def fetch_data( url: str, params: dict[str,str|None]|None=None): 
    if is_cache_valid(CACHE_FILE,EXPIRY_HOURS): 
        print("Using cached data...")
        with open(CACHE_FILE, "r",encoding='utf-8') as f:
            reader = csv.reader(f)
            data = list(reader) 
            data = pd.DataFrame(data[1: ], columns=data[0])
            return data

    print('Fetching data...')
    response = r.get(url, params=params)
    response.raise_for_status()
    
    with open(CACHE_FILE, "w", encoding='utf-8') as f:
        f.write(response.text)
        data = pd.read_csv(StringIO(response.text), header = 0 ) # type: ignore
    
        return data

url = 'https://www.alphavantage.co/query'
params: dict[str,str|None] = {
          'function': 'TIME_SERIES_DAILY', 
          'symbol': 'QQQ',
          'outputsize': 'full',
          'datatype': 'csv', 
          'apikey': API}

def csv_opens(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        data = list(reader)
        df = pd.DataFrame(data[1: ], columns=data[0])
        return df

def annualised_returns(df: Series, year_periods: int=250):  
    mean_returns = df.mean() 
    return mean_returns * year_periods

def annualised_sharpe(returns: Series, periods_per_year: int=252):  
    returns = returns.dropna()
    if returns.empty: 
        return 0.0

    std = returns.std()
    if std < 1e-12:
        return 0.0 

    return np.sqrt(periods_per_year) * returns.mean() / std

def max_drawdown(returns: Series) -> float:  
    '''
    Args:
        returns: log returns of a stock over time
    Returns: 
    
    '''
    cum: Series = pd.Series(np.exp(returns.cumsum()),index=returns.index)
    peak: Series = cum.cummax() 
    dd: Series = (cum - peak) / peak
    return dd.min()

def stationary_test(data: Series, name: str="Series", quiet: bool=False) -> dict[str,Any]:  
    """
    Perform the Augmented Dickey-Fuller test on a time series.

    Args:  
        data (Series):   The time series to test.
        name (str):   Optional name for the series.
        quiet (bool):   If True, suppresses printed output.

    Returns:  
        Dict[str, Any]:   A dictionary with ADF statistic, p-value, and critical values.
    """
    data = data.dropna()
    data = data[np.isfinite(data)]
    test = adfuller(data) #type: ignore
    output: dict[str, Any] = { 
        'ADF Stat': test[0],
        'p-value': test[1],
        'critical values': test[4] #type: ignore
    }
    if not quiet:
        print(f'Test statistic:  {output["ADF Stat"]: .3f}')
        print(f'P-value:  {output['p-value']: .3f}')
        print([f'{key}:  {value: .3f}' for key, value in output['critical values'].items()])
        
    return output  
    
def johansen_weights(price_df: DataFrame, det_order: int=0, k_ar_diff: int=1):  
    """
    price_df: DataFrame with asset prices as columns
    returns: normalised cointegrating vector (weights) so weighted sum =  spread
    """
    arr: Any = price_df #type: ignore
    result = coint_johansen(arr, det_order, k_ar_diff)
    vec = result.evec[ : ,0]
    vec = vec/ np.sum(np.abs(vec))
    weights = pd.Series(vec, index = price_df.columns)
    return weights

def construct_spread(price_df: DataFrame, weights: Series):
    """
    args:
        weights: Series aligned with the price_df columns
    Results:
        spread
    """
    p = price_df.copy()
    spread = (p * weights).sum(axis=1)
    return spread

def half_life(spread: pd.Series) -> float|None: 
    s = spread.dropna()
    ds = s.diff().dropna()
    s_lag = s.shift(1).dropna()
    result = linregress(s_lag.values, ds.values)
    slope: float = result.slope #type: ignore
    if slope >= 0:
        return None
    return -np.log(2) / slope #type: ignore 

def backtest_spread_strategy(price_df: DataFrame, weights: Series,
                             entry_z: float=2.0, exit_z: float = 0.5,
                             lookback_z: int = 60,
                             max_pos: float = 1.0,
                             tc_per_trade: float = 0.0005,
                             slippage: float=0.0005,
                             use_log: bool=True
                             ): 
    '''
    Returns daily pnl series, positions and diagnostics
    positions:
        +1 means long spread 
        -1 short spread
    we assume weights define spread = w1log(p1)+...; position in individual assets proportional to weights'''
    p = price_df.copy()
    if use_log:
        p_use = np.log(p)
    else:
        p_use = p 
    spread = (p_use * weights).sum(axis=1) # type: ignore
    rolling_mean = spread.rolling(lookback_z).mean()
    rolling_std = spread.rolling(lookback_z).std()
    z = (spread - rolling_mean)/(rolling_std)
    pos = pd.Series(0.0, index=spread.index, dtype=float)
    position = 0.0
    
    for t in range(lookback_z, len(spread)):
        _ = spread.index[t]
        zt = z.iloc[t]
        # entry
        if position == 0:
            if zt > entry_z:
                position = -max_pos
            elif zt < -entry_z:
                position = max_pos
        elif position > 0:
            if zt >= -exit_z:
                position = 0
        elif position < 0:
            if zt <= exit_z:
                position = 0.0
        pos.iloc[t] = position
        
    norm_w = weights / weights.abs().sum()
    returns = p.pct_change().fillna(0) # type: ignore
    pos_lag = pos.shift(1).fillna(0)  # type: ignore
    asset_pos = (pos_lag.to_frame(name='pos') # type: ignore
                 .join(pd.DataFrame([norm_w]*len(pos), index=pos.index))
                 .drop(columns=['pos']))
    asset_pos = asset_pos.multiply(pos_lag, axis=0)
    
    pnl = (asset_pos * returns).sum(axis=1)
    turnover = pos.diff().abs().fillna(0) # type: ignore
    tc = turnover * tc_per_trade
    slippage_cost = turnover * slippage
    pnl_net = pnl - tc - slippage_cost
    diagnostics:  dict[str, Any] = {
        'spread':  spread,
        'zscore':  z,
        'position':  pos,
        'asset_pos':  asset_pos,
        'pnl_gross':  pnl,
        'pnl_net':  pnl_net,
        'transaction_costs':  tc,
        'slippage':  slippage_cost
    }
    return pnl_net, diagnostics

def plot_diagnostics(diagnostics: dict[str, Any]):
    spread = diagnostics['spread']
    zscore = diagnostics['zscore']
    position = diagnostics['position']
    pnl_net = diagnostics['pnl_net']

    plt.style.use('classic') 
    fig, axs = plt.subplots(2, 2, sharex=True) #type: ignore

    cumulative_pnl = pnl_net.cumsum()
    plots = [ #type: ignore
        (axs[0, 0], spread, 'Spread', 'blue', [('mean', spread.mean(), 'red', '--')]),
        (axs[0, 1], zscore, 'Z-score with Thresholds', 'purple',
         [('Entry', 2.0, 'green', '--'), ('Entry', -2.0, 'green', '--'),
          ('Exit', 0.5, 'orange', '--'), ('Exit', -0.5, 'orange', '--')]),
        (axs[1, 0], position, 'Position Over Time', 'black', []),
        (axs[1, 1], cumulative_pnl, 'Cumulative Net PnL', 'darkorange', [])
    ]

    for ax, series, title, color, hlines in plots: #type: ignore
        ax.plot(series, label=title, color=color, linewidth=0.8, alpha=0.8)#type: ignore
        for label, y, hcolor, style in hlines: #type: ignore
            ax.axhline(y, color=hcolor, linestyle=style, label=label if 'label' not in ax.get_legend_handles_labels()[1] else "") #type: ignore
        ax.set_title(title) #type: ignore
        ax.grid(True) #type: ignore
        ax.legend(loc='best') #type: ignore
        ax.set_facecolor('white') #type: ignore


    for ax in axs.flat: 
        ax.set_xlabel('Date')

    plt.grid(True) #type: ignore
    plt.tight_layout()
    plt.show() #type: ignore
    
def objective_weights_only(price_df: DataFrame,
                           lookback_z: int=60,metric: str='sharpe',
                           tc: float=0.0005, slippage: float=0.0005): 
    """
    Use inside Optuna study:  trial is an optuna.trial.Trial object.
    We'll suggest N raw weight parameters and normalize them.
    """
    def opt_fn(trial: optuna.trial.Trial): 
        n = price_df.shape[1]
        raw = np.array([trial.suggest_float(f'w_{i}', -2.0, 2.0) for i in range(n)]) 
        denom = np.sum(np.abs(raw))
        if denom == 0:
            norm = np.ones_like(raw) /len(raw)
        else:    
            norm = raw / denom
        weights = pd.Series(norm, index=price_df.columns) #type: ignore
        pnl, diag = backtest_spread_strategy(price_df, weights,
                                            lookback_z=lookback_z,
                                            tc_per_trade=tc,
                                            slippage=slippage)
        sr = annualised_sharpe(pnl.dropna())
        mdd = max_drawdown(pnl.dropna())
        if metric == 'sharpe':
            score = sr - 5.0 * abs(mdd)  
        
        elif metric == 'profit_factor':
            gross = diag['pnl_gross']
            wins = gross[gross>0].sum()
            losses = -gross[gross<0].sum()
            pf = wins/(losses)
            score = pf
        else:
            score = sr
        return score
    return opt_fn

def objective_weights_and_thresholds(price_df: DataFrame):
    def opt_fn(trial: optuna.trial.Trial):
        n = price_df.shape[1]
        raw = np.array([trial.suggest_float(f'w_{i}', -2.0, 2.0) for i in range(n)])
        norm = raw - raw.mean()
        norm = norm / (np.abs(norm).sum() + 1e-12)
        weights = pd.Series(norm, index=price_df.columns)
        entry_z = trial.suggest_float('entry_z', 1.0, 3.5)
        exit_z  = trial.suggest_float('exit_z', 0.1, 1.5)
        pnl, _ = backtest_spread_strategy(price_df, weights,
                                          entry_z=entry_z, exit_z=exit_z)
        sr = annualised_sharpe(pnl.dropna())
        return sr
    return opt_fn

def run_optuna(price_df: DataFrame, objective_fn: Any,
               n_trials: int=200, sampler: None|Any =optuna.samplers.TPESampler(seed=42),
               direction: str='maximize'):
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=True)
    return study



def rolling_evaluation(price_df: DataFrame, train_days: int=504, test_days: int=126, step_days: int=126,
                       bo_trials: int=80, optimize_thresholds: int=True):
    """
    train_days, test_days in trading days (approx 252 days = 1y)
    """
    dates = price_df.index
    results = []
    start_idx = 0
    while start_idx + train_days + test_days <= len(dates):
        train_idx = slice(start_idx, start_idx + train_days)
        test_idx = slice(start_idx + train_days, start_idx + train_days + test_days)
        train_prices = price_df.iloc[train_idx]
        test_prices = price_df.iloc[test_idx]
        try:
            base_w = johansen_weights(train_prices)
        except Exception:
            base_w = pd.Series(np.ones(train_prices.shape[1]) / train_prices.shape[1], index=train_prices.columns)
        if optimize_thresholds:
            obj = objective_weights_and_thresholds(train_prices)
        else:
            obj = objective_weights_only(price_df=train_prices)  # you can adapt
        study = optuna.create_study(direction='maximize')
        study.optimize(obj, n_trials=bo_trials, show_progress_bar=False)
        best_params = study.best_params
        # reconstruct weights & thresholds
        n = train_prices.shape[1]
        raw = np.array([best_params.get(f'w_{i}', 0.0) for i in range(n)])
        norm = raw - raw.mean()
        norm = norm / (np.abs(norm).sum() + 1e-12)
        opt_weights = pd.Series(norm, index=train_prices.columns)
        entry_z = best_params.get('entry_z', 2.0)
        exit_z  = best_params.get('exit_z', 0.5)
        # Evaluate on test
        pnl_test, _ = backtest_spread_strategy(test_prices, opt_weights, entry_z=entry_z, exit_z=exit_z)
        sr_test = annualised_sharpe(pnl_test.dropna())
        mdd_test = max_drawdown(pnl_test.dropna())
        halflife_test = half_life(construct_spread(test_prices, opt_weights))
        # baseline on test
        pnl_base_test, _ = backtest_spread_strategy(test_prices, base_w)
        sr_base = annualised_sharpe(pnl_base_test.dropna())
        results.append({ #type: ignore
            'train_start':  dates[train_idx.start],
            'train_end':  dates[start_idx + train_days - 1],
            'test_start':  dates[test_idx.start],
            'test_end':   dates[start_idx + train_days + test_days - 1],
            'opt_weights':  opt_weights,
            'entry_z':  entry_z, 'exit_z':  exit_z,
            'sr_test':  sr_test, 'mdd_test':  mdd_test, 'halflife':  halflife_test,
            'sr_base':  sr_base
        })
        start_idx += step_days
    return pd.DataFrame(results)


def plot_comparison(test_prices: DataFrame, opt_weights: Series, base_weights: Series, entry_z: float, exit_z: float):
    pnl_opt, _ = backtest_spread_strategy(test_prices, opt_weights, entry_z=entry_z, exit_z=exit_z)
    pnl_base, _ = backtest_spread_strategy(test_prices, base_weights)
    cum_opt = (1 + pnl_opt).cumprod()
    cum_base = (1 + pnl_base).cumprod()
    plt.figure(figsize=(10,6)) #type: ignore
    plt.plot(cum_opt.index, cum_opt.values, label='BO optimized') #type: ignore
    plt.plot(cum_base.index, cum_base.values, label='Johansen baseline', linestyle='--')#type: ignore
    plt.tick_params('x', size=250)
    plt.legend() #type: ignore
    plt.title('Cumulative return:  Optimized vs Baseline') #type: ignore
    plt.show() #type: ignore



np.set_printoptions(legacy='1.25')
spy = csv_opens('SPY.csv').dropna() #type: ignore
spy = spy.set_index('timestamp').astype(float).sort_index(ascending=True)
qqq = csv_opens('QQQ.csv').dropna() #type: ignore
qqq = qqq.set_index('timestamp').astype(float).sort_index(ascending=True)
spy['returns'] = np.log(spy['close']) - np.log(spy['close'].shift(1))
qqq['returns'] = np.log(qqq['close']) - np.log(qqq['close'].shift(1))
spy.dropna() #type: ignore
qqq.dropna() #type: ignore
spy_renamed = spy.rename(columns={'close':  'close_spy'})
qqq_renamed = qqq.rename(columns={'close':  'close_qqq'})

price_df: DataFrame = pd.concat([spy_renamed['close_spy'], qqq_renamed['close_qqq']], axis=1, join='inner')   
print(price_df)
print(annualised_returns(spy['returns']), annualised_returns(qqq['returns']))
print(annualised_sharpe(spy['returns']), annualised_sharpe(qqq['returns'])) 
print(max_drawdown(spy['returns']), max_drawdown(qqq['returns']))
print(stationary_test(qqq['returns']))
print(stationary_test(spy['returns']))
weights = johansen_weights(price_df)
spread = construct_spread(price_df, weights)
print(stationary_test(spread))
print(half_life(spread))
diagnostics = backtest_spread_strategy(price_df=price_df, weights=weights, lookback_z=60, entry_z=2.5, exit_z=1, tc_per_trade=0)[1]
# import time
# start = time.time()
# plot_diagnostics(diagnostics=diagnostics)
# print(f"Render time:  {time.time() - start: .2f} seconds")
opt_fn = objective_weights_only(price_df)
study = (run_optuna(price_df,opt_fn))
results = rolling_evaluation(price_df)
print(results)
row = results.iloc[38]
test_start = row['test_start']
test_end = row['test_end']
test_prices = price_df.loc[test_start:test_end]
opt_weights = row['opt_weights']   # optimized weights from BO
base_weights = johansen_weights(price_df.loc[row['train_start']:row['train_end']])
entry_z = row.get('entry_z', 2.0)
exit_z = row.get('exit_z', 0.5)
plot_comparison(test_prices, opt_weights, base_weights, entry_z, exit_z)
