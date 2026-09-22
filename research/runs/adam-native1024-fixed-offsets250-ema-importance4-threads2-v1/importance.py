"""Auditable empirical squared-gradient mechanics; no model or selection policy.

Each add_example call MUST contain gradients of ONE example's scalar objective.
The helper cannot infer whether a caller supplied a batch-averaged gradient.
For D, one example is a real/fake PAIR: differentiate the summed real+fake
loss first, then square. Separate squared real/fake gradients lose cross terms.
For source AdAM G/D losses this estimates empirical loss-gradient second moments,
not a claim of an exact probabilistic Fisher matrix. Source reference:
AdAM 6428e99cfb36bc8bda3506350824f0c7dac9a5ad,
AdAM_importance_probing.py:193-260 and AdAM_main_adaptation.py:709-811.

Sums are CPU float64 by default (explicit numerical deviation from source FP32);
float32 is available. Normalize once by actual successful example count, never
by parameter size. Missing gradients fail closed unless explicitly treated as
zero, in which case their examples remain in the denominator. State dictionaries
contain only primitives and detached tensors for torch.save/weights_only reload.
This is accumulator continuation, not model/RNG checkpoint or training resume.
"""
from collections.abc import Mapping

import torch


class EmpiricalSquaredGradients:
    def __init__(self, named_tensors, *, missing="error", accumulator_dtype=torch.float64):
        if missing not in ("error", "zero"):
            raise ValueError("missing must be 'error' or explicit 'zero'")
        if accumulator_dtype not in (torch.float32, torch.float64):
            raise ValueError("accumulator_dtype must be float32 or float64")
        items = list(named_tensors.items() if isinstance(named_tensors, Mapping) else named_tensors)
        names = [name for name, _ in items]
        if not names or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
            raise ValueError("Require nonempty, unique string names")
        self.missing = missing
        self.dtype = accumulator_dtype
        self.sample_count = 0
        self.gradient_sums = {}
        self.squared_gradient_sums = {}
        self.observed_counts = {}
        for name, tensor in sorted(items):
            if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point() or tensor.numel() == 0:
                raise ValueError(f"{name}: require nonempty real floating tensor")
            self.gradient_sums[name] = torch.zeros(tensor.shape, dtype=self.dtype, device="cpu")
            self.squared_gradient_sums[name] = torch.zeros_like(self.gradient_sums[name])
            self.observed_counts[name] = 0

    def add_example(self, named_gradients):
        """Atomically add one example. Missing names must be supplied as None."""
        if set(named_gradients) != set(self.gradient_sums):
            raise ValueError("Gradient names must exactly match declared names; use explicit None")
        prepared = {}
        for name, total in self.gradient_sums.items():
            grad = named_gradients[name]
            if grad is None:
                if self.missing == "error":
                    raise ValueError(f"{name}: missing gradient")
                continue
            if (not isinstance(grad, torch.Tensor) or not grad.is_floating_point()
                    or grad.layout != torch.strided or grad.shape != total.shape):
                raise ValueError(f"{name}: gradient type/layout/shape mismatch")
            grad = grad.detach().to(device="cpu", dtype=self.dtype)
            first = total + grad
            second = self.squared_gradient_sums[name] + grad.square()
            if not bool(torch.isfinite(first).all() and torch.isfinite(second).all()):
                raise ValueError(f"{name}: nonfinite gradient or accumulator overflow")
            prepared[name] = (first, second)
        # No mutation occurs until every supplied tensor passes validation.
        for name, (first, second) in prepared.items():
            self.gradient_sums[name] = first
            self.squared_gradient_sums[name] = second
            self.observed_counts[name] += 1
        self.sample_count += 1

    def mean_gradients(self):
        if self.sample_count == 0:
            raise ValueError("No examples accumulated")
        return {n: t / self.sample_count for n, t in self.gradient_sums.items()}

    def mean_squared_gradients(self):
        if self.sample_count == 0:
            raise ValueError("No examples accumulated")
        return {n: t / self.sample_count for n, t in self.squared_gradient_sums.items()}

    def state_dict(self):
        return {"version": 1, "missing": self.missing, "dtype": str(self.dtype),
                "sample_count": self.sample_count, "observed_counts": dict(self.observed_counts),
                "gradient_sums": {n: t.clone() for n, t in self.gradient_sums.items()},
                "squared_gradient_sums": {n: t.clone() for n, t in self.squared_gradient_sums.items()}}

    @classmethod
    def from_state_dict(cls, state):
        expected = {"version", "missing", "dtype", "sample_count", "observed_counts",
                    "gradient_sums", "squared_gradient_sums"}
        if set(state) != expected or state["version"] != 1:
            raise ValueError("Unsupported state schema")
        dtypes = {"torch.float32": torch.float32, "torch.float64": torch.float64}
        if state["dtype"] not in dtypes:
            raise ValueError("Unsupported state dtype")
        count = state["sample_count"]
        if type(count) is not int or count < 0:
            raise ValueError("Invalid sample count")
        obj = cls(state["gradient_sums"], missing=state["missing"], accumulator_dtype=dtypes[state["dtype"]])
        names = set(obj.gradient_sums)
        if set(state["squared_gradient_sums"]) != names or set(state["observed_counts"]) != names:
            raise ValueError("State names differ")
        for name in obj.gradient_sums:
            observed = state["observed_counts"][name]
            if type(observed) is not int or not 0 <= observed <= count:
                raise ValueError(f"{name}: invalid observed count")
            if obj.missing == "error" and observed != count:
                raise ValueError(f"{name}: missing observations under error policy")
            for key in ("gradient_sums", "squared_gradient_sums"):
                value = state[key][name]
                if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                        or value.device.type != "cpu" or value.dtype != obj.dtype
                        or value.shape != obj.gradient_sums[name].shape or not bool(torch.isfinite(value).all())):
                    raise ValueError(f"{name}: invalid {key}")
                if observed == 0 and bool(torch.count_nonzero(value)):
                    raise ValueError(f"{name}: nonzero sums without observations")
            if bool((state["squared_gradient_sums"][name] < 0).any()):
                raise ValueError(f"{name}: negative squared-gradient sum")
            obj.gradient_sums[name] = state["gradient_sums"][name].detach().clone()
            obj.squared_gradient_sums[name] = state["squared_gradient_sums"][name].detach().clone()
            obj.observed_counts[name] = observed
        obj.sample_count = count
        return obj


def source_uvb_score(u_fisher, v_fisher, b_fisher=None):
    """Reduce already sample-normalized factor scores, preserving source formulas.

    Without bias: mean(Fu)+Fv. With bias: (mean(Fu)+Fv+Fb)/2, including
    the source's divisor TWO for three terms. Callers explicitly supply relevant
    bias; G synthesis-conv noise/activation offsets are not automatically added.
    Shapes are vectors, biases must match v exactly. No min/max normalization,
    pooling, thresholds or architectural channel/selection mapping is performed.
    """
    tensors = [u_fisher, v_fisher] + ([] if b_fisher is None else [b_fisher])
    for t in tensors:
        if (not isinstance(t, torch.Tensor) or not t.is_floating_point() or t.ndim != 1
                or t.numel() == 0 or not bool(torch.isfinite(t).all()) or bool((t < 0).any())):
            raise ValueError("Require finite nonnegative nonempty vector second moments")
    if any(t.dtype != v_fisher.dtype or t.device != v_fisher.device for t in tensors):
        raise ValueError("Score tensors must share dtype and device")
    if b_fisher is not None and b_fisher.shape != v_fisher.shape:
        raise ValueError("Bias shape must match v; broadcasting is forbidden")
    score = u_fisher.detach().mean() + v_fisher.detach()
    if b_fisher is not None:
        score = (score + b_fisher.detach()) / 2
    if not bool(torch.isfinite(score).all()):
        raise ValueError("Score overflow")
    return score
