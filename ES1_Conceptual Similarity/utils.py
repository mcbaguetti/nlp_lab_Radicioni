import csv
import numpy as np
from tabulate import tabulate

def load_data(filename):
    matrix = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)
        matrix.append(header)
        for row in csv_reader:
            matrix.append([row[0], row[1], float(row[2])])
        matrix = np.array(matrix, object)
    return matrix


def print_table(table, num_rows=None, show_indices=False):
    if num_rows:
        print(tabulate(table[:num_rows, :], headers='firstrow',
              showindex=show_indices, tablefmt='fancy_grid'))
    else:
        print(tabulate(table[:, :], headers='firstrow',
              showindex=show_indices, tablefmt='fancy_grid'))
