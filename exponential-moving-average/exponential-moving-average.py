def exponential_moving_average(values, alpha):
    """
    Compute the exponential moving average of the given values.
    """
    # Write code here
    if not values:
        return []

    ema = [values[0]]

    for t in range(1, len(values)):
        ema1 = alpha* values[t] + (1 - alpha)* ema[-1]

        ema.append(ema1)

    return ema
    