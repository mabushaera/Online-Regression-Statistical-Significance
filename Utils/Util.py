import numpy as np
import os


def get_dataset_path(file_name):
    """
    get the dataset path stored in the project directory.
    """
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    parent_folder_path = os.path.dirname(current_script_path)
    path = os.path.join(parent_folder_path, 'Datasets', 'Datasets_Generators_CSV', file_name)
    return path


def get_path_to_save_generated_dataset_file(directory):
    """
    returns the needed path to save the generated figure.
    """
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    parent_folder_path = os.path.dirname(current_script_path)
    path = os.path.join(parent_folder_path, 'Datasets', 'Datasets_Generators_CSV', directory)
    return path


def create_directory(path):
    """
    creates directory of the specified path if not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Folder created at {path}")
    else:
        print(f"Folder already exists at {path}")


def sum_lists_element_wise(array_a, array_b):
    """
    Summing two arrays element wise
    """
    if len(array_a) == 0:
        array_a = array_b.copy()
        return array_a

    sum_values = np.sum([array_a, array_b], axis=0)
    return sum_values


def combine_two_datasets(xs, ys, xs_new, ys_new):
    """
    Combines two data sets, used to generate adversarial scenarios
    """
    temp1 = list(zip(xs, ys))
    temp2 = list(zip(xs_new, ys_new))
    temp = temp1 + temp2
    res1, res2 = zip(*temp)
    xn, yn = list(res1), list(res2)
    xs_new = np.array(xn)
    ys_new = np.array(yn, dtype=object)
    return xs_new, ys_new


def calculate_no_of_base_model_points(no_of_data_points, base_model_percent=10):
    """
    compute the number of base model points
    which is usually a percent of the total points like 1%, 10%
    """
    calculate_start_points = int(no_of_data_points * base_model_percent / 100)
    return calculate_start_points


def print_header(header_text):
    """
    For formatted printing, header like.
    """
    print(header_text)
    print("=" * len(header_text))  # Print a line of '=' characters for underline effect


def sample_and_combine(Xj, yj, r_w_base):
    """
        generate sample data from the provided r_w_base which is the base model coefficients
        combines the generated data with the provided input feature matrix and output labels
        to generate the combined (current and sampled from the base model)

        Parameters:
        Xj (numpy.ndarray): Input feature matrix for the primary dataset.
        yj (numpy.ndarray): Output labels for the primary dataset.
        r_w_base (numpy.ndarray): The base model coefficients

        Returns:
        combinedXj (numpy.ndarray): Combined feature matrix
        combinedyj (numpy.ndarray): Combined output labels corresponding to the combined feature matrix.
        """
    d_b = r_w_base[-1]
    c_b = r_w_base[-2]

    base_yj = np.array([])
    for x in Xj:
        res = -1 * (np.dot(r_w_base[0:-2], x) + d_b) / c_b
        base_yj = np.append(base_yj, res)

    combinedXj, combinedyj = combine_two_datasets(Xj, yj, Xj, base_yj)
    return combinedXj, combinedyj


def print_average_of_maps(list_of_maps, model_name, num_of_first_batche=10):
    # Initialize a dictionary to store the averages
    averages = {}

    # Iterate through each key in the first map to initialize the averages dictionary
    for key in list_of_maps[0]:
        averages[key] = 0.0

    # Calculate the sum of values for each key
    for data_map in list_of_maps:
        for key, value in data_map.items():
            averages[key] += float(value)

    # Calculate the average for each key
    num_maps = len(list_of_maps)
    for key in averages:
        averages[key] /= num_maps

    # Print the resulting averages
    print(model_name, averages)

    # Calculate the sum of the values
    total_sum = sum(averages.values())

    # Calculate the count of the values
    count = len(averages)

    # Calculate the average
    average_value = total_sum / count

    # Print the result
    print('The average of all the values of ', model_name, ' is: ', average_value)

    # Extract the first 10 values from the dictionary
    first_10_values = list(averages.values())[:10]

    # Calculate the sum of the first 10 values
    total_sum_first_10 = sum(first_10_values)

    # Calculate the count of the first 10 values
    count_first_10 = len(first_10_values)

    # Calculate the average of the first 10 values
    average_first_10 = total_sum_first_10 / count_first_10

    # Print the result
    print('The average of the first 10 values is:', average_first_10)



