

def calculate_activation(weights, data_point, bias):
    sum = bias

    for key,val in weights.items():
        sum += val*data_point[key]

    return sum