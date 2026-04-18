import torch
from torch_geometric.utils import degree


def get_order(data, descending=False, **kwargs):
    """
    Returns a permutation of nodes sorted by node degree (connectivity)
    within each graph in the batch.
    """
    device = data.batch.device
    num_nodes = data.num_nodes

    # Compute node degrees
    # edge_index is shape [2, num_edges]. edge_index[0] is the source node index
    deg = degree(data.edge_index[0], num_nodes=num_nodes, dtype=torch.float64).to(
        device
    )

    # Add a small amount of random noise to break ties consistently
    noise = torch.rand(num_nodes, device=device, dtype=torch.float64) * 0.1
    scores = deg + noise

    if descending:
        scores = -scores

    # Normalize scores to be strictly between 0 and 1
    scores = scores - scores.min()
    max_score = scores.max()
    if max_score > 0:
        scores = scores / (max_score + 1.0)

    # Add batch index to group nodes by graph
    batch_scores = data.batch.to(torch.float64) + scores

    perm = torch.argsort(batch_scores)
    return perm
