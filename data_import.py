



def data_reader(loc):
    """

    :param loc: location where csv is stored
    :return: a tuple with an array of dictionaries which has the data, and the classification attribute of the data
    """
    import csv
    test_data = []
    with open(loc, 'rt') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            row_int = dict((k,(-1 if v == '0' else 1)) for k,v in row.items())
            test_data.append(row_int)
    return test_data, reader.fieldnames[len(reader.fieldnames)-1]
