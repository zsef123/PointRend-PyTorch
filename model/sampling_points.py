import torch


@torch.no_grad()
def sampling_points(mask, k=3, beta=0.75, training=True, N=-1):
    """
    Follows 3.1. Point Selection for Inference and Training

    In Train:, `The sampling strategy selects N points on a feature map to train on.`

    In Inference, `then selects the N most uncertain points`

    Args:
        mask(Tensor): [B, C, H, W]
        k(int): Over generation multiplier
        beta(float): ratio of importance points
        training(bool): flag
        N(int): In inference, num of selecetion points

    Return:
        selected_point(Tensor) : flattened indexing points [B, num_points]
    """
    assert mask.dim() == 4, "Dim must be N(Batch)CHW"
    assert not training and N == -1, "N must be positive int"

    device = mask.device
    B, _, H, W = mask.shape

    v, _ = mask.softmax(1).sort(1, descending=True)
    # When Values bigger getting more Uncertainty
    uncertainty_map = -1 * (v[:, 0, :, :] - v[:, 1, :, :])

    if not training:
        N = min(N, H * W)
        _, idx = uncertainty_map.view(B, -1).topk(N)
        return idx

    N = H * W
    over_generation = torch.randint(N, (B, k * N), dtype=torch.long, device=device)
    over_generation_map = torch.gather(uncertainty_map.view(B, N), 1, over_generation).view(B, -1)

    # most uncertain βN points
    _, idx = over_generation_map.topk(int(beta * N))
    importance = torch.gather(over_generation, 1, idx)

    coverage = torch.randint(N, (B, int((1 - beta) * N)), dtype=torch.long, device=device)
    return torch.cat([importance, coverage], dim=1).to(device=mask.device)
