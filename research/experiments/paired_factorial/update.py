"""Original paired pixel objective plus optional reviewed frozen-D term, CPU only."""
import math
import torch
from research.experiments.clothing_structure.losses import feature_reconstruction_loss

FEATURE_WEIGHT=.05


def gradient_norms(loss,named_parameters):
    values=torch.autograd.grad(loss,[p for _,p in named_parameters],retain_graph=True,allow_unused=True)
    sums={'all':0.,'existing64plus':0.,'new_b32':0.}
    for (name,_),gradient in zip(named_parameters,values):
        if gradient is None:continue
        if not bool(torch.isfinite(gradient).all()):raise FloatingPointError('Nonfinite diagnostic gradient')
        square=float(gradient.detach().double().square().sum())
        sums['all']+=square;sums['new_b32' if name.startswith('synthesis.b32.') else 'existing64plus']+=square
    del values
    return {k:math.sqrt(v) for k,v in sums.items()}


def update(student,source,optimizer,pair,options,preservation,trainer,prefix=None,diagnostic=False):
    if pair['split']!='train':raise ValueError('Only six original training identities may update weights')
    optimizer.zero_grad(set_to_none=True)
    rgb=student.synthesis(pair['w'],noise_mode='const',force_fp32=True)
    generated=trainer.grayscale(rgb)
    clothing,tab,rest=trainer.partition_loss(generated,trainer.grayscale(pair['target']),pair['collar'],pair['rest'])
    protected=trainer.masked_l1(generated,trainer.grayscale(pair['original']),1-pair['mask'])
    paired=options['clothing_weight']*clothing+options['protected_weight']*protected
    metrics={'pair_ids':[pair['id']],'clothing_l1':float(clothing.detach()),'tab_l1':float(tab.detach()),
             'rest_l1':float(rest.detach()),'protected_l1':float(protected.detach()),'paired_pixel_loss':float(paired.detach()),
             'feature_loss':0.,'feature_weight':FEATURE_WEIGHT if prefix is not None else 0.,'fresh_preservation_l1':0.}
    feature=None
    if prefix is not None:
        feature,feature_metrics=feature_reconstruction_loss(prefix,rgb,pair['target'],pair['original'],pair['mask'],pair['collar'])
        metrics['feature_loss']=float(feature.detach());metrics['feature_diagnostics']=feature_metrics
    if diagnostic:
        if feature is None:raise ValueError('Gradient calibration requires a feature arm')
        params=[(n,p) for n,p in student.named_parameters() if p.requires_grad]
        pixel_norms=gradient_norms(paired,params);feature_norms=gradient_norms(FEATURE_WEIGHT*feature,params)
        metrics['gradient_calibration']={'pixel':'paired clothing plus protected pixel loss; fresh term excluded',
            'feature':'weighted 0.05 frozen-D term','pixel_l2':pixel_norms,'weighted_feature_l2':feature_norms,
            'raw_feature_l2':{k:v/FEATURE_WEIGHT for k,v in feature_norms.items()},
            'raw_norm_derivation':'weighted norm / 0.05; exact scalar homogeneity, no third autograd pass',
            'feature_over_pixel':{k:feature_norms[k]/pixel_norms[k] if pixel_norms[k]>0 else None for k in pixel_norms},
            'extra_autograd_passes':2,'optimizer_updates':1}
    loss=paired if feature is None else paired+FEATURE_WEIGHT*feature
    if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite paired/feature objective')
    loss.backward()
    del rgb,generated,clothing,tab,rest,protected,paired,feature,loss
    if options['fresh_weight']>0:
        z=torch.randn(1,source.z_dim,generator=preservation)
        with torch.no_grad():
            w=source.mapping(z,None,truncation_psi=1,skip_w_avg_update=True)
            original=trainer.grayscale(source.synthesis(w,noise_mode='const',force_fp32=True))
        predicted=trainer.grayscale(student.synthesis(w,noise_mode='const',force_fp32=True))
        height=max(1,round(source.img_resolution*options['upper_fraction']))
        fresh=(predicted[:,:,:height]-original[:,:,:height]).abs().mean()
        if not bool(torch.isfinite(fresh)):raise FloatingPointError('Nonfinite fresh preservation')
        (options['fresh_weight']*fresh).backward();metrics['fresh_preservation_l1']=float(fresh.detach())
    grads=[p.grad for p in student.parameters() if p.requires_grad and p.grad is not None]
    if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):raise FloatingPointError('Missing/nonfinite student gradient')
    if any(p.grad is not None for p in source.parameters()) or (prefix is not None and any(p.grad is not None for p in prefix.parameters())):raise AssertionError('Frozen source/D acquired gradients')
    optimizer.step()
    if not all(bool(torch.isfinite(p).all()) for p in student.parameters() if p.requires_grad):raise FloatingPointError('Nonfinite updated weights')
    return metrics
