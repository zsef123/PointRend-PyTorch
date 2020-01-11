import torch


@torch.no_grad()
def sampling_points(mask, k=3, beta=0.75, training=True):
    """
    Args:
        mask(Tensor): [B, C, H, W]

    Return:
        selected_point(Tensor) : flattened indexing points [B, num_points]
    """
    assert mask.dim() == 4, "Dim must be N(Batch)CHW"
    device = mask.device
    v, _ = mask.softmax(1).sort(1, descending=True)
    # When Values bigger getting more Uncertainty
    uncertainty_map = -1 * (v[:, 0, :, :] - v[:, 1, :, :])

    B, H, W = uncertainty_map.shape
    # Num of points
    N = H * W

    over_generation = torch.randint(N, (B, k * N), dtype=torch.long, device=device)
    over_generation_map = torch.gather(uncertainty_map.view(B, N), 1, over_generation).view(B, -1)

    # most uncertain βN points
    _, idx = over_generation_map.topk(int(beta * N))
    importance = torch.gather(over_generation, 1, idx)

    coverage = torch.randint(N, (B, int((1 - beta) * N)), dtype=torch.long, device=device)
    return torch.cat([importance, coverage], dim=1).to(device=mask.device)
