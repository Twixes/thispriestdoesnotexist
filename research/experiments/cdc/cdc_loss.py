"""Independent CDC-only experiment; see README.md for paper and limitations.

This helper imports no Adobe implementation. Existing NVIDIA network code and
weights retain their upstream research/evaluation license.
"""
import torch
import torch.nn.functional as F


def collect_features(model, latents, blocks=(32, 64, 128), pool_size=8):
    """Capture corresponding synthesis features without changing mode or w_avg."""
    if not blocks or len(set(blocks)) != len(blocks):
        raise ValueError('Feature blocks must be a nonempty unique sequence')
    if pool_size < 1:
        raise ValueError('pool_size must be positive')
    modules = dict(model.named_modules())
    features = {}
    handles = []

    def capture(resolution):
        def hook(module, inputs, output):
            feature = output[0].float()
            features[resolution] = F.adaptive_avg_pool2d(feature, (pool_size, pool_size)).flatten(1)
        return hook

    try:
        for resolution in blocks:
            name = f'synthesis.b{resolution}'
            if name not in modules:
                raise ValueError(f'Generator lacks requested feature block {name}')
            handles.append(modules[name].register_forward_hook(capture(resolution)))
        # Unmixed z, no truncation, identical constant-noise buffers. Explicitly
        # skip w_avg updates even when the adapted generator is training.
        styles = model.mapping(latents, None, truncation_psi=1.0, skip_w_avg_update=True)
        model.synthesis(styles, noise_mode='const', force_fp32=True, fused_modconv=False)
    finally:
        for handle in handles:
            handle.remove()
    return features


def pairwise_log_probabilities(features, temperature=1.0):
    """For each sample, softmax similarities to every OTHER sample."""
    if features.ndim != 2 or features.shape[0] < 4:
        raise ValueError('CDC requires at least four feature vectors')
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    normalized = F.normalize(features, dim=1, eps=1e-8)
    similarities = normalized @ normalized.T / temperature
    batch = features.shape[0]
    off_diagonal = ~torch.eye(batch, device=features.device, dtype=torch.bool)
    return F.log_softmax(similarities[off_diagonal].reshape(batch, batch - 1), dim=1)


def cdc_loss(source, target, latents, blocks=(32, 64, 128), pool_size=8, temperature=1.0):
    """Mean KL(source || target), averaged over samples and feature blocks.

    Caller supplies a frozen source. This returns an unweighted differentiable
    loss; callers must calibrate their own coefficient for this reduction.
    """
    if latents.ndim != 2 or latents.shape[0] < 4:
        raise ValueError('CDC requires an unmixed latent batch of at least four')
    if any(parameter.requires_grad for parameter in source.parameters()):
        raise ValueError('Source generator must be frozen before computing CDC')
    with torch.no_grad():
        source_features = collect_features(source, latents, blocks, pool_size)
        source_log_probs = {block: pairwise_log_probabilities(feature, temperature)
                            for block, feature in source_features.items()}
    target_features = collect_features(target, latents, blocks, pool_size)
    losses = []
    for block in blocks:
        target_log_probs = pairwise_log_probabilities(target_features[block], temperature)
        source_log = source_log_probs[block]
        divergence = (source_log.exp() * (source_log - target_log_probs)).sum(dim=1).mean()
        losses.append(divergence)
    total = torch.stack(losses).mean()
    details = {'per_block_kl': {str(block): float(loss.detach()) for block, loss in zip(blocks, losses)},
               'feature_shapes': {str(block): list(value.shape) for block, value in target_features.items()},
               'batch': latents.shape[0], 'blocks': list(blocks), 'pool_size': pool_size,
               'temperature': temperature, 'reduction': 'mean over examples and blocks; sum over neighbors'}
    return total, details
