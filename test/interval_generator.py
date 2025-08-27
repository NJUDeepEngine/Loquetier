from scipy.stats import truncnorm
import numpy as np

INTERVAL_GENERATORS = {}


def NormalInterval(length, total_time, mean_ratio=0.5, std_ratio=0.1, **kwargs):
    """
    Generate time intervals for a normal distribution under given arguments.
    
    Args:
    - length:       int, quantity of interval data to be generated in total
    - total_time:   float, duration of data
    - mean_ratio:   float, mean time ratio in the duration, default to be 0.5
    - std_ratio:    float, std time ratio in the duration, default to be 0.1
    
    Returns:
    - list, of time intervals
    """
    offset = kwargs.get('offest', 0)
    intervals = sorted(
        truncnorm.rvs(
            -mean_ratio / std_ratio,
            (1 - mean_ratio) / std_ratio,
            loc=total_time * mean_ratio,
            scale=total_time * std_ratio,
            size=length
        )
    )
    last_interval = -offset
    for interval in intervals:
        yield interval - last_interval
        last_interval = interval
    return

def StableInterval(length, mean_interval, random_factor=0.2, **kwargs):
    """
    Generate time intervals with restricted randomization under given arguments.
    
    Args:
    - length:           int, quantity of interval data to be generated in total
    - mean_interval:    float, mean interval
    - random_factor:    float, random disturbance ratio, default to be 0.2
    
    Returns:
    - list, of time intervals
    """
    offset = kwargs.get('offest', 0)
    if random_factor > 0.5:
        random_factor = 0.5
    elif random_factor < 0:
        random_factor = 0
    
    intervals = np.random.uniform(
        -mean_interval * random_factor,
        mean_interval * random_factor,
        size=length
    )
    yield mean_interval / 2 + intervals[0] + offset
    for interval in intervals[1:]:
        yield mean_interval + interval
    return

INTERVAL_GENERATORS['NormalInterval'] = NormalInterval
INTERVAL_GENERATORS['StableInterval'] = StableInterval
INTERVAL_GENERATORS['NoWait'] = None
