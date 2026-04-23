import torch
from mendeleev import element

# Global cache for electronegativity values
_EN_CACHE = {}

def get_en_pauling(z_val):
    """
    Lookup electronegativity with caching.
    """
    z_int = int(z_val)
    if z_int not in _EN_CACHE:
        try:
            en = element(z_int).en_pauling
            _EN_CACHE[z_int] = float(en) if en is not None else 0.0
        except Exception:
            _EN_CACHE[z_int] = 0.0
    return _EN_CACHE[z_int]

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
        # Fallback to first feature if z is missing
        z = data.x[:, 0].long()

    # Vectorize electronegativity lookup via pre-populated cache
    # First, make sure all unique Z in the batch are in the cache
    unique_z = torch.unique(z).tolist()
    for atomic_num in unique_z:
        get_en_pauling(atomic_num)

    electronegativity = torch.tensor(
        [_EN_CACHE[int(val.item())] for val in z],
        device=device,
        dtype=torch.float32
    )

    # Add small noise to break ties
    noise = torch.rand(num_nodes, device=device, dtype=torch.float32) * 0.01
    scores = electronegativity + noise

    if descending:
        scores = -scores

    # Normalize to (0, 1) - adding epsilon to avoid div-by-zero
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    # Group by batch - ensure batch is on same device
    batch_indices = data.batch.to(device).to(torch.float32)
    # Combining batch index and score ensures sorting happens WITHIN each batch
    # Adding 1.0 to scores ensures they don't overlap with the next batch's base index
    batch_scores = batch_indices + (scores / 2.0)

    perm = torch.argsort(batch_scores)
    return perm