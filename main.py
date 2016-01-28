from data_import import data_reader
from calculate import calculate_activation
import sys
import random
import time





def main():
    if len(sys.argv)!= 4:
        print("Please add arguments")
        sys.exit(2)

    try:
        data = data_reader(sys.argv[1])
        test_data = data_reader(sys.argv[2])[0]
    except FileNotFoundError:
        print("File location in parameters are incorrect")
        sys.exit(2)

    ITERATIONS = 100

    training_data = data[0]
    class_name = data[1]
    weights = {}
    headers = list(training_data[0].keys())
    headers.remove(class_name)
    headers.sort()
    bias = 0

    for header in headers:
        weights[header] = 0

    cur_iterations = 0
    while cur_iterations <= ITERATIONS:
        all_correct = True

        for data_point in training_data:
            output = calculate_activation(weights, data_point, bias)

            if output*data_point[class_name] <= 0:
                all_correct = False
                for header in headers:
                    weights[header] += data_point[header]*data_point[class_name]
                bias += data_point[class_name]

        if (all_correct == True):
            break
        cur_iterations += 1

    for key, val in weights.items():
        print(key + " has weight: " + str(val))

    total = 0
    correct = 0
    for data_point in test_data:
        total += 1
        if data_point[class_name]*calculate_activation(weights, data_point, bias) == 1:
            correct += 1
    print(total)
    print(correct)
    print("Percentage correct: " + str((correct/total)*100) + "%")





if __name__ =="__main__":
    main()