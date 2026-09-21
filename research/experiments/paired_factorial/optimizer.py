"""Reviewed transfer logic copied from unused capacity draft; tested independently here."""
import copy
from research.experiments.paired_surround.runner import equal_tree

def fork_optimizer(student,parent_optimizer,variant,trainer,torch,expected_old_names=None):
    """Restore original64+ Adam by original order, then append empty-state b32 group."""
    old_names=trainer.freeze_student(student)
    if expected_old_names is not None and old_names!=expected_old_names:raise ValueError('Original parameter order differs')
    old_params=[p for p in student.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(old_params,lr=1e-4,betas=(.9,.999))
    optimizer.load_state_dict(copy.deepcopy(parent_optimizer))
    if not equal_tree(optimizer.state_dict(),parent_optimizer,torch):raise ValueError('Existing Adam state transfer changed')
    if len(optimizer.param_groups)!=1 or optimizer.param_groups[0]['lr']!=1e-4 or tuple(optimizer.param_groups[0]['betas'])!=(.9,.999):raise ValueError('Unexpected parent optimizer hyperparameters')
    added=[]
    if variant:
        for name,p in student.named_parameters():
            if name.startswith('synthesis.b32.') and not name.endswith('noise_strength'):
                p.requires_grad_(True);added.append((name,p))
        if not added:raise ValueError('Missing b32 trainable parameters')
        group={k:copy.deepcopy(v) for k,v in optimizer.param_groups[0].items() if k!='params'}
        group['params']=[p for _,p in added];optimizer.add_param_group(group)
        if any(p in optimizer.state for _,p in added):raise ValueError('New b32 state must start empty')
    expected=set(old_names)|{name for name,_ in added}
    if {n for n,p in student.named_parameters() if p.requires_grad}!=expected:raise ValueError('Unexpected freeze-policy change')
    old_after=optimizer.state_dict();old_only={'state':{k:v for k,v in old_after['state'].items() if k in parent_optimizer['state']},'param_groups':old_after['param_groups'][:1]}
    if not equal_tree(old_only,parent_optimizer,torch):raise ValueError('Appending group modified original Adam state')
    return optimizer,{'old_parameter_names':old_names,'new_parameter_names':[n for n,_ in added],
                      'old_optimizer_state_exact':True,'new_optimizer_state_empty':True}
