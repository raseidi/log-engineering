def meta_trace(keys, trace):
    import numpy as np
    from pandas import Series
    from scipy import stats
    # trace = []
    # n_events = 0
    # for trace in log:
    #     n_events += len(trace)
    #     trace.append(len(trace))
    trace = trace[1:]
    trace_min = np.min(trace)
    trace_max = np.max(trace)
    trace_mean = np.mean(trace)
    trace_median = np.median(trace)
    trace_mode = stats.mode(trace)[0][0]
    trace_std = np.std(trace)
    trace_variance = np.var(trace)
    trace_q1 = np.percentile(trace, 25)
    trace_q3 = np.percentile(trace, 75)
    trace_iqr = stats.iqr(trace)
    trace_geometric_mean = stats.gmean(trace+1)
    trace_geometric_std = stats.gstd(trace+1)
    trace_harmonic_mean = stats.hmean(trace)
    trace_skewness = stats.skew(trace)
    trace_kurtosis = stats.kurtosis(trace)
    trace_coefficient_variation = stats.variation(trace)
    trace_entropy = stats.entropy(trace)
    trace_hist, _ = np.histogram(trace, density=True)
    trace_skewness_hist = stats.skew(trace_hist)
    trace_kurtosis_hist = stats.kurtosis(trace_hist)

    return keys, [
        trace_min,
        trace_max,
        trace_mean,
        trace_median,
        trace_mode,
        trace_std,
        trace_variance,
        trace_q1,
        trace_q3,
        trace_iqr,
        trace_geometric_mean,
        trace_geometric_std,
        trace_harmonic_mean,
        trace_skewness,
        trace_kurtosis,
        trace_coefficient_variation,
        trace_entropy,
        *trace_hist,
        trace_skewness_hist,
        trace_kurtosis_hist,
    ]