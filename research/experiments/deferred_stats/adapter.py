"""Experimental reporting transport; does not change NVIDIA's reductions.

Call flush() before any Collector update/read, ADA update or checkpoint capture.
Not installed in the active trainer. The sink must be the original CPU reporting
adapter, captured before this object replaces the reporting entry point.
"""
import torch


class DeferredStats:
    def __init__(self, sink):
        self.sink = sink
        self.pending = []
        self.last_flush = {'reports':0,'packs':0,'elements':0}

    def report(self, name, value):
        # Clone NOW: detach alone would allow later in-place tensor/NumPy/list
        # mutation to change the observation before it reaches the CPU sink.
        frozen = torch.as_tensor(value).detach().clone()
        self.pending.append((name,frozen))
        return value

    def flush(self):
        groups = {}
        records = []
        for name, frozen in self.pending:
            key = (frozen.device,frozen.dtype)
            if key not in groups:
                groups[key] = {'pieces':[],'elements':0}
            group = groups[key]
            size = frozen.numel()
            records.append((name,key,group['elements'],size))
            group['pieces'].append(frozen.reshape(-1))
            group['elements'] += size
        # Keep input dtypes; converting values/reducing on the device could alter
        # rounding versus the original CPU float32 reduction and float64 counters.
        host = {key:torch.cat(group['pieces']).cpu() for key,group in groups.items()}
        for name,key,start,size in records:
            self.sink(name,host[key][start:start+size])
        self.last_flush = {'reports':len(records),'packs':len(groups),
                           'elements':sum(group['elements'] for group in groups.values())}
        self.pending.clear()
        return self.last_flush

    def assert_empty(self):
        if self.pending:
            raise RuntimeError('Flush deferred reports before statistics/ADA/checkpoint access')
