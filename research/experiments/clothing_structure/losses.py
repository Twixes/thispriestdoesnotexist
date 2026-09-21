"""Unused research loss components. No loaders, optimizer, hooks or output edits."""
import torch
from torch import nn
from torch.nn import functional as F


def _images(*images):
    first = images[0]
    if first.ndim != 4 or first.shape[1] not in (1, 3) or min(first.shape) < 1:
        raise ValueError('Expected nonempty NCHW grayscale or RGB images')
    for image in images:
        if image.shape != first.shape or image.device != first.device or image.dtype != torch.float32:
            raise ValueError('Images must share shape/device and use FP32')
        if not bool(torch.isfinite(image).all()):
            raise ValueError('Images must be finite')


def _gray(image):
    if image.shape[1] == 1:
        return image
    return (image * image.new_tensor([.299, .587, .114])[None, :, None, None]).sum(1, keepdim=True)


def _regions(image, clothing, collar):
    shape = (image.shape[0], 1, *image.shape[2:])
    for mask in (clothing, collar):
        if mask.shape != shape or mask.device != image.device:
            raise ValueError('Masks must match image batch/spatial shape and device')
        if not bool(torch.isfinite(mask).all()) or not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError('Input masks must be finite binary masks')
    clothing = clothing.detach().to(dtype=image.dtype)
    collar = collar.detach().to(dtype=image.dtype)
    rest = clothing - collar
    if bool((rest < 0).any()):
        raise ValueError('Collar must be contained in clothing')
    areas = [mask.flatten(1).sum(1) for mask in (collar, rest, 1-clothing)]
    if not all(bool((area > 0).all()) for area in areas):
        raise ValueError('Every image needs nonempty collar, rest clothing and protected regions')
    return clothing, collar, rest


def region_mean(values, mask):
    """Per-image channel/area normalization; supports fractional feature masks."""
    if values.ndim != 4 or mask.shape != (values.shape[0], 1, *values.shape[2:]):
        raise ValueError('Feature/mask shape mismatch')
    if values.device != mask.device or not bool(torch.isfinite(values).all()):
        raise ValueError('Features must be finite and share mask device')
    if not bool(torch.isfinite(mask).all()) or not bool(((mask >= 0) & (mask <= 1)).all()):
        raise ValueError('Feature mask must be finite in [0,1]')
    mask = mask.detach()
    mass = mask.flatten(1).sum(1) * values.shape[1]
    if not bool((mass > 0).all()):
        raise ValueError('Every feature loss region must have positive mass')
    return (values * mask).flatten(1).sum(1) / mass


class FrozenDPrefix(nn.Module):
    """NVIDIA D prefix; default taps b512/b256 yield spatial 256/128.

    Freezes the supplied D in place, retaining only prefix module references.
    The caller owns loading/provenance. Tiny tests use the same API at 64px.
    """
    def __init__(self, discriminator, input_resolution=1024, tap_blocks=(512, 256)):
        super().__init__()
        if discriminator.img_resolution != input_resolution or discriminator.img_channels != 3 or discriminator.c_dim != 0:
            raise ValueError('Expected matching unconditional RGB discriminator')
        resolutions = list(discriminator.block_resolutions)
        if not tap_blocks or len(set(tap_blocks)) != len(tap_blocks) or any(t not in resolutions for t in tap_blocks):
            raise ValueError('Taps must name distinct pre-epilogue discriminator blocks')
        last = max(resolutions.index(t) for t in tap_blocks)
        self.input_resolution = input_resolution
        self.tap_blocks = tuple(tap_blocks)
        self.block_resolutions = resolutions[:last+1]
        discriminator.eval().requires_grad_(False)
        self.blocks = nn.ModuleDict({str(r):discriminator._modules[f'b{r}'] for r in self.block_resolutions})
        self.eval().requires_grad_(False)

    def forward(self, image):
        if self.training or any(m.training for m in self.modules()) or any(p.requires_grad for p in self.parameters()):
            raise ValueError('Feature prefix must remain frozen and in eval mode')
        if image.ndim != 4 or image.shape[1:] != (3, self.input_resolution, self.input_resolution):
            raise ValueError('Feature input must be native-resolution RGB')
        x, img = None, image
        outputs = {}
        for resolution in self.block_resolutions:
            x, img = self.blocks[str(resolution)](x, img, force_fp32=True)
            if resolution in self.tap_blocks:
                outputs[resolution] = x
        return [outputs[r] for r in self.tap_blocks]


def feature_reconstruction_loss(prefix, generated, target, source, clothing, collar):
    """Unweighted frozen-D term plus detached diagnostics; no output compositing.

    Composites are temporary loss inputs. No target cache is retained by this
    minimal component, and target/source/masks receive no gradients.
    """
    _images(generated, target, source)
    clothing, collar, rest = _regions(generated, clothing, collar)
    y = _gray(generated)
    t = _gray(target.detach())
    s = _gray(source.detach())
    student_input = (clothing*y + (1-clothing)*s).expand(-1, 3, -1, -1)
    target_input = (clothing*t + (1-clothing)*s).expand(-1, 3, -1, -1)
    with torch.no_grad():
        target_features = [f.detach() for f in prefix(target_input)]
    student_features = prefix(student_input)
    if not student_features or len(student_features) != len(target_features):
        raise ValueError('Feature tap mismatch')
    terms, metrics = [], []
    for predicted, expected in zip(student_features, target_features):
        if predicted.shape != expected.shape:
            raise ValueError('Feature shape mismatch')
        c = F.interpolate(collar, size=predicted.shape[2:], mode='area')
        r = F.interpolate(rest, size=predicted.shape[2:], mode='area')
        # Per-image detached target RMS, floor1: never amplify tiny scales.
        scale = region_mean(expected.square(), c+r).sqrt().clamp_min(1).detach()
        difference = (predicted-expected).abs()
        tab = region_mean(difference, c)
        other = region_mean(difference, r)
        term = ((.5*tab + .5*other)/scale).mean()
        if not bool(torch.isfinite(term)):
            raise FloatingPointError('Nonfinite feature loss')
        terms.append(term)
        metrics.append({'shape':list(predicted.shape),'tab_l1':float(tab.mean().detach()),
                        'rest_l1':float(other.mean().detach()),'scale_mean':float(scale.mean()),
                        'normalized_loss':float(term.detach())})
    return torch.stack(terms).mean(), {'layers':metrics}


def collar_boundary_loss(generated, target, clothing, collar, radius=8):
    """Unweighted first-difference loss in collar band, plus diagnostics.

    Every compared edge has both endpoints in unchanged clothing. Full images
    are neither edited nor returned. Radius8 is the proposed 1024px setting.
    """
    _images(generated, target)
    clothing, collar, _ = _regions(generated, clothing, collar)
    if not isinstance(radius, int) or isinstance(radius, bool) or radius < 1:
        raise ValueError('Boundary radius must be a positive integer')
    kernel = 2*radius+1
    # Zero beyond image for both operations; avoid inventing outside-image edges.
    padded = F.pad(collar, (radius,)*4, value=0)
    dilation = F.max_pool2d(padded, kernel, stride=1)
    erosion = -F.max_pool2d(-padded, kernel, stride=1)
    band = (dilation-erosion)*clothing
    mx = clothing[:,:,:,1:]*clothing[:,:,:,:-1]*torch.maximum(band[:,:,:,1:], band[:,:,:,:-1])
    my = clothing[:,:,1:,:]*clothing[:,:,:-1,:]*torch.maximum(band[:,:,1:,:], band[:,:,:-1,:])
    y, t = _gray(generated), _gray(target.detach())
    dx = ((y[:,:,:,1:]-y[:,:,:,:-1])-(t[:,:,:,1:]-t[:,:,:,:-1])).abs()
    dy = ((y[:,:,1:,:]-y[:,:,:-1,:])-(t[:,:,1:,:]-t[:,:,:-1,:])).abs()
    count = mx.flatten(1).sum(1)+my.flatten(1).sum(1)
    if not bool((count > 0).all()):
        raise ValueError('Every image needs a nonempty valid collar-boundary edge set')
    per_image = ((dx*mx).flatten(1).sum(1)+(dy*my).flatten(1).sum(1))/count
    loss = per_image.mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite boundary loss')
    return loss, {'valid_edges_per_image':count.tolist(),'radius':radius,
                  'boundary_loss':float(loss.detach())}
