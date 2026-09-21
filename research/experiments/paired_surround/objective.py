"""Isolated collar-surround objective; no loaders or active trainer mutations.

The update is adapted from the hash-pinned paired_regions update. Only the
clothing partition changes; protected/fresh terms, sampling and Adam step match.
"""
import torch
from torch.nn import functional as F

RADIUS=30


def partition(clothing,raw_tab,radius=RADIUS):
    if type(radius) is not int or radius<1:raise ValueError('Positive integer Chebyshev radius required')
    if clothing.shape!=raw_tab.shape or clothing.ndim!=4 or clothing.shape[1]!=1:raise ValueError('Matching N1HW masks required')
    for mask in [clothing,raw_tab]:
        if mask.requires_grad or not bool(torch.isfinite(mask).all()) or not bool(((mask==0)|(mask==1)).all()):raise ValueError('Masks must be detached finite binary tensors')
    # Separable square max equals radius-r Chebyshev dilation, with zero exterior.
    # This avoids a costly61x61 reduction at every native-resolution pixel.
    k=radius*2+1
    expanded=F.max_pool2d(raw_tab,(1,k),stride=1,padding=(0,radius))
    expanded=F.max_pool2d(expanded,(k,1),stride=1,padding=(radius,0))
    collar=clothing*raw_tab
    surround=expanded*(1-raw_tab)*clothing
    remainder=clothing-collar-surround
    regions=(collar.detach(),surround.detach(),remainder.detach())
    if not all(bool((m.flatten(1).sum(1)>0).all()) for m in regions):raise ValueError('Every image needs nonempty collar, surround and remainder')
    if not torch.equal(sum(regions),clothing) or any(bool((a*b).any()) for a,b in [(regions[0],regions[1]),(regions[0],regions[2]),(regions[1],regions[2])]):raise ValueError('Partition overlap or clothing leakage')
    if any(bool((m*(1-clothing)).any()) for m in regions):raise ValueError('Protected pixels entered clothing partition')
    return regions


def attach_regions(pairs,manifest,trainer):
    rows={row['id']:row for row in manifest['pairs']};counts=[]
    for pair in pairs:
        if pair['split']!='train':continue
        row=rows[pair['id']]
        raw=trainer.polygon_mask(row['collar_polygons'],pair['mask'].shape[-1])
        collar,surround,remainder=partition(pair['mask'],raw)
        if not torch.equal(collar,pair['collar']) or not torch.equal(surround+remainder,pair['rest']):raise ValueError('Existing collar/rest masks changed')
        pair['surround']=surround;pair['remainder']=remainder
        counts.append({'id':pair['id'],'radius':RADIUS,'raw_tab_pixels':int(raw.sum()),'collar_pixels':int(collar.sum()),
                       'surround_pixels':int(surround.sum()),'remainder_pixels':int(remainder.sum()),'clothing_pixels':int(pair['mask'].sum()),
                       'protected_leak_pixels':0,'partition_disjoint':True})
    return counts


def clothing_loss(generated,target,collar,surround,remainder,masked_l1):
    tab=masked_l1(generated,target,collar)
    ring=masked_l1(generated,target,surround)
    rest=masked_l1(generated,target,remainder)
    return (tab+ring+rest)/3,tab,ring,rest


def update(student,source,optimizer,pairs,options,sampling,preservation,trainer):
    if not pairs or any(pair['split']!='train' for pair in pairs):raise ValueError('Only training pairs may be passed to update')
    indices=torch.randint(len(pairs),(options['batch'],),generator=sampling).tolist()
    selected=[pairs[index] for index in indices]
    values={name:torch.cat([pair[name] for pair in selected]).to('cpu')
            for name in ['w','original','target','mask','collar','surround','remainder']}
    optimizer.zero_grad(set_to_none=True)
    generated=trainer.grayscale(student.synthesis(values['w'],noise_mode='const',force_fp32=True))
    clothing,collar,surround,remainder=clothing_loss(generated,trainer.grayscale(values['target']),values['collar'],values['surround'],values['remainder'],trainer.masked_l1)
    protected=trainer.masked_l1(generated,trainer.grayscale(values['original']),1-values['mask'])
    paired=options['clothing_weight']*clothing+options['protected_weight']*protected
    if not torch.isfinite(paired):raise FloatingPointError('Nonfinite paired loss')
    paired.backward()
    metrics={'pair_ids':[pair['id'] for pair in selected],'clothing_l1':float(clothing.detach()),
             'collar_l1':float(collar.detach()),'surround_l1':float(surround.detach()),'remainder_l1':float(remainder.detach()),
             'protected_l1':float(protected.detach()),'paired_loss':float(paired.detach()),'fresh_preservation_l1':0.0}
    del generated,clothing,collar,surround,remainder,protected,paired,values
    if options['fresh_weight']>0:
        z=torch.randn(1,source.z_dim,generator=preservation).to('cpu')
        with torch.no_grad():
            w=source.mapping(z,None,truncation_psi=1,skip_w_avg_update=True)
            original=trainer.grayscale(source.synthesis(w,noise_mode='const',force_fp32=True))
        predicted=trainer.grayscale(student.synthesis(w,noise_mode='const',force_fp32=True))
        height=max(1,round(source.img_resolution*options['upper_fraction']))
        fresh=(predicted[:,:,:height]-original[:,:,:height]).abs().mean()
        if not torch.isfinite(fresh):raise FloatingPointError('Nonfinite fresh-latent loss')
        (options['fresh_weight']*fresh).backward();metrics['fresh_preservation_l1']=float(fresh.detach())
    gradients=[p.grad for p in student.parameters() if p.requires_grad and p.grad is not None]
    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):raise FloatingPointError('Missing or nonfinite student gradients')
    if any(p.grad is not None for p in source.parameters()):raise AssertionError('Frozen source acquired a gradient')
    optimizer.step()
    if not all(bool(torch.isfinite(p).all()) for p in student.parameters() if p.requires_grad):raise FloatingPointError('Nonfinite updated student')
    return metrics
