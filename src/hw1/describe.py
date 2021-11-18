from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx

from utils import load_edges_from_file


def draw_degree_distribution(G: nx.Graph,
                             save_path: str = 'degree_distribution.png',
                             title: Optional[str] = None) -> None:
    """Draws degree distribution.

    Args:
        G: networkx graph

    Returns:
        None
    """
    degrees_count = [d / G.number_of_nodes() for d in nx.degree_histogram(G)]
    plt.bar(range(len(degrees_count)), degrees_count)
    if title is None:
        title = 'Degree distribution'
    plt.title(title)
    plt.xlabel('Degree')
    plt.ylabel('Fraction of nodes')
    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == '__main__':
    G = nx.Graph()
    G.add_edges_from(load_edges_from_file('dataset/Cora/edges'))
    average_degree = 2 * G.number_of_edges() / G.number_of_nodes()
    print("Number of Nodes", G.number_of_nodes())
    print("Number of Edges", G.number_of_edges())
    print("Average Degree", average_degree)
    draw_degree_distribution(
        G, 'cora_degree_distribution.png', "Cora Degree Distribution")
    print("Average Clustering Coefficient", nx.average_clustering(G))

    G = nx.Graph()
    G.add_edges_from(load_edges_from_file('dataset/karateClub/edges'))
    average_degree = 2 * G.number_of_edges() / G.number_of_nodes()
    print("Number of Nodes", G.number_of_nodes())
    print("Number of Edges", G.number_of_edges())
    print("Average Degree", average_degree)
    draw_degree_distribution(
        G, 'karate_club_degree_distribution.png', 'KarateClub Degree Distribution')
    print("Average Clustering Coefficient", nx.average_clustering(G))
