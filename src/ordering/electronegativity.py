import torch
from mendeleev import element


def get_order(data, descending=False, **kwargs):
    """
    Returns a permutation of nodes sorted by electronegativity within each graph in the batch.
    """
    device = data.batch.device
    num_nodes = data.num_nodes

    # Extract atomic numbers (z)
    if hasattr(data, "z") and data.z is not None:
        z = data.z.long()
    else:
        z = data.x[:, 0].long()

    # Get Pauling electronegativity using mendeleev
    # Cache results to avoid repeated lookups
    unique_z = torch.unique(z).tolist()
    en_map = {}
    for atomic_num in unique_z:
        #try:
        en = element(int(atomic_num)).en_pauling
        en_map[atomic_num] = en if en is not None else 0.0
        # except Exception:
        #     en_map[atomic_num] = 0.0

    electronegativity = torch.tensor(
        [en_map[int(val.item())] for val in z],
        device=device,
        dtype=torch.float64
    )

    # Add small noise to break ties
    noise = torch.rand(num_nodes, device=device, dtype=torch.float64) * 0.01
    scores = electronegativity + noise

    if descending:
        scores = -scores

    # Normalize to (0, 1)
    scores = scores - scores.min()
    max_score = scores.max()
    if max_score > 0:
        scores = scores / (max_score + 1.0)

    # Group by batch
    batch_scores = data.batch.to(torch.float64) + scores

    perm = torch.argsort(batch_scores)
    return perm