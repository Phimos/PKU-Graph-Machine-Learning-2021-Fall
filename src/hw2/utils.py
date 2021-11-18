import numpy as np


def load_edges_from_file(file_path: str):
    """
    Loads edges from file.

    :param file_path: path to file
    :return: list of edges
    """
    edges = []
    with open(file_path) as f:
        for line in f:
            edges.append(tuple(map(int, line.split())))
    return edges


def load_labels_from_file(file_path: str):
    labels = []
    with open(file_path) as f:
        for line in f:
            labels.append(tuple(map(int, line.split())))
    return labels


def load_nodes_from_file(file_path: str):
    """
    Loads nodes from file.

    :param file_path: path to file
    :return: list of nodes
    """
    nodes = []
    with open(file_path) as f:
        for line in f:
            nodes.append(int(line))
    return nodes


def load_attribute_from_file(file_path: str):
    """
    Loads attribute from file.

    :param file_path: path to file
    :return: list of attributes
    """
    attributes = []
    with open(file_path) as f:
        for line in f:
            attributes.append(list(map(float, line.split())))
    return attributes


if __name__ == '__main__':
    print(np.array(load_edges_from_file('data/cora/edges')).shape)
    print(np.array(load_nodes_from_file('data/cora/train_nodes')).shape)
    print(np.array(load_labels_from_file('data/cora/labels')).shape)
    print(np.array(load_attribute_from_file('data/cora/attributes')).shape)
