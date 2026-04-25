def simple_moving_average(values, window_size):
    n = len(values)
    
    if window_size > n:
        return []

    result = []

    for i in range(n - window_size + 1):
        window_sum = sum(values[i:i + window_size])
        result.append(window_sum / window_size)

    return result