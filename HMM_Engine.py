
import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd
#Baum-welch
def baum_welch(X, n_components=3, n_iter=1000):
    T, N = X.shape
    K = n_components

    pi = np.ones(K) / K
    A = np.ones((K, K)) / K
    A += np.random.rand(3, 3) * 0.1
    A = A / A.sum(axis=1, keepdims=True)

    means = X[np.random.choice(T, K, replace=False)]
    covars = np.array([np.cov(X.T)] * K)

    for iteration in range(n_iter):
        B = np.zeros((T, K))
        for k in range(K):
            B[:, k] = multivariate_normal.pdf(X, mean=means[k], cov=covars[k])
        
        alpha = np.zeros((T, K))
        c = np.zeros(T)
        alpha[0] = pi * B[0]
        c[0] = 1.0 / (np.sum(alpha[0]) + 1e-10)
        alpha[0] *= c[0]

        for t in range(1, T):
            alpha[t] = alpha[t-1].dot(A) * B[t]
            c[t] = 1.0 / (np.sum(alpha[t]) + 1e-10)
            alpha[t] *= c[t]
        
        beta = np.zeros((T, K))
        beta[-1] = 1.0 * c[-1]

        for t in range(T-2, -1, -1):
            beta[t] = A.dot(B[t+1] * beta[t+1]) * c[t]
        
        gamma = alpha * beta
        gamma = gamma / (gamma.sum(axis=1, keepdims=True) + 1e-10)

        xi = np.zeros((T-1, K, K))
        for t in range(T-1):
            denominator = np.dot(np.dot(alpha[t], A) * B[t+1], beta[t+1]) + 1e-10
            for i in range(K):
                numerator = alpha[t, i] * A[i] * B[t+1] * beta[t+1]
                xi[t, i] = numerator / denominator

        pi = gamma[0]
        A = np.sum(xi, axis=0) / (np.sum(gamma[:-1], axis=0).reshape(-1,1) + 1e-10)

        for k in range(K):
            gamma_k = gamma[:, k]
            sum_gamma_k = np.sum(gamma_k) + 1e-10

            means[k] = np.sum(gamma_k[:, np.newaxis] * X, axis=0) / sum_gamma_k

            diff = X - means[k]
            covars[k] = np.dot(gamma_k * diff.T, diff) / sum_gamma_k
            covars[k].flat[::N+1] += 1e-5
        
    return pi, A, means, covars

#viterbi
def viterbi(X, pi, A, means, covars):
    T, K = X.shape[0], len(pi)

    log_pi = np.log(pi + 1e-10)
    log_A = np.log(A + 1e-10)

    V = np.zeros((T, K))
    ptr = np.zeros((T, K), dtype=int)

    log_B = np.zeros((T, K))

    for k in range(K):
        log_B[:, k] = multivariate_normal.logpdf(X, mean=means[k], cov=covars[k])
    V[0] = log_pi + log_B[0]

    for t in range(1, T):
        for j in range(K):
            prob = V[t-1]
            ptr[t, j] = np.argmax(prob)
            V[t, j] = np.max(prob) + log_B[t, j]

    states = np.zeros(T, dtype=int)
    states[-1] = np.argmax(V[-1])
    for t in range(T-2, -1, -1):
        states[t] = ptr[t+1, states[t+1]]
    
    return states

def generate_report(df, asset_name):
    strategies = {
        "Buy & Hold": df['market_return'],
        "Pure HMM": df['pure_hmm_return'],
        "HMM + GARCH": df['strategy_return']
    }
    
    trades_market = 1 
    trades_hmm = (df['hmm_signal'].diff().fillna(0) != 0).sum()
    rounded_position = df['final_position'].round(2)
    trades_garch = (rounded_position.diff().fillna(0) != 0).sum()

    print("\n" + "="*80)
    print(f"REPORT: ADAPTIVE VOLATILITY MODELING ({asset_name})".center(80))
    print("="*80)
    print(f"{'Metric':<22} | {'Buy & Hold':<15} | {'Pure HMM':<12} | {'HMM + GARCH':<15}")
    print("-" * 80)
    
    for metric in ['Total Return', 'Annualized Return', 'Volatility (Risk)', 'Max Drawdown', 'Sharpe Ratio', 'Number of Trades']:
        row = f"{metric:<22} |"
        for name, returns in strategies.items():
            cum_ret = np.exp(returns.sum()) - 1
            ann_ret = np.exp(returns.mean() * 252) - 1
            ann_vol = returns.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
            
            curve = np.exp(returns.cumsum())
            peak = curve.cummax()
            mdd = ((curve - peak) / peak).min()
            
            if metric == 'Total Return': val = f"{cum_ret*100:>8.2f}%"
            elif metric == 'Annualized Return': val = f"{ann_ret*100:>8.2f}%"
            elif metric == 'Volatility (Risk)': val = f"{ann_vol*100:>8.2f}%"
            elif metric == 'Max Drawdown': val = f"{mdd*100:>8.2f}%"
            elif metric == 'Sharpe Ratio': val = f"{sharpe:>8.2f} "
            elif metric == 'Number of Trades':
                if name == 'Buy & Hold': val = f"{trades_market:>8}"
                elif name == 'Pure HMM': val = f"{trades_hmm:>8}"
                elif name == 'HMM + GARCH': val = f"{trades_garch:>8}"

            row += f" {val:<14}|" if name == 'Buy & Hold' else f" {val:<11}|" if name == 'Pure HMM' else f" {val:<15}"
        print(row)
    print("="*80 + "\n")