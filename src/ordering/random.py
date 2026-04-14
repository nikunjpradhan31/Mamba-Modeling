import torch


def get_order(data, **kwargs):
    """
    Returns a random permutation of nodes within each graph in the batch.
    Nodes within the same graph are permuted randomly, while maintaining
    the overall batch structure.
    """
    device = data.batch.device
    num_nodes = data.num_nodes

    # Assign a random score to each node
    scores = torch.rand(num_nodes, device=device, dtype=torch.float64)

    # Add batch index to group nodes by graph
    # Since batch indices are integers (0, 1, 2,...) and scores are in [0, 1),
    # sorting by (batch + scores) will first group by batch, then sort randomly within.
    batch_scores = data.batch.to(torch.float64) + scores

    perm = torch.argsort(batch_scores)
    return perm
