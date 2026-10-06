"""Torch model used by the AB-JChess V11 trainer.

The module has a dedicated name and fixed tensor shapes that mirror
the fixed native evaluator: a 2,048-wide feature transformer, sixteen
layer-stack heads, and SFNN squared/clipped 32/32 heads with concatenation and a skip used by the
runtime.
"""

from __future__ import annotations

import copy
import math
import numbers
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import nn
import torch.nn.functional as F

try:
    import pytorch_lightning as pl
    _LightningBase = pl.LightningModule
except Exception:  # pragma: no cover - doctor/static tools can omit Lightning
    pl = None
    _LightningBase = nn.Module

try:
    import ranger21  # type: ignore
except Exception:  # pragma: no cover - the fallback is sufficient for basic use
    ranger21 = None

from feature_transformer import DoubleFeatureTransformerSlice
from architecture_v11 import ARCHITECTURE_SHA256
import features_v11
from head_balance_v11 import compute_head_weights


DEFAULT_L1 = features_v11.L1_DIM
DEFAULT_L2 = features_v11.L2_DIM
DEFAULT_L3 = features_v11.L3_DIM
L1, L2, L3 = DEFAULT_L1, DEFAULT_L2, DEFAULT_L3

_UINT64_MASK = (1 << 64) - 1
_HEAD_SEED_STEP = 0x9E3779B1
_OUTPUT_WEIGHT_RANGE = 127 / 128
CONTEXT_INPUTS = 16
CONTEXT_HIDDEN = 16
HEAD_INPUTS = 2048 + CONTEXT_HIDDEN
_POOL_MAXIMA = (2.0, 2.0, 2.0, 5.0, 2.0, 2.0)


def _normalized_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise TypeError("seed must be an integer, not a boolean")
    return int(seed) & _UINT64_MASK


def _cpu_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def dims_for_feature_set(feature_set: Any) -> tuple[int, int, int]:
    return (
        int(getattr(feature_set, "l1", None) or DEFAULT_L1),
        int(getattr(feature_set, "l2", None) or DEFAULT_L2),
        int(getattr(feature_set, "l3", None) or DEFAULT_L3),
    )


def coalesce_ft_weights(model: Any, layer: nn.Module) -> torch.Tensor:
    """Fold factor rows into each of the 31,776 real feature rows."""

    weight = layer.weight.data
    gather = model.feature_set.get_virtual_to_real_features_gather_indices()
    result = weight.new_zeros((model.feature_set.num_real_features, weight.shape[1]))
    for real_index, factors in enumerate(gather):
        result[real_index, :] = sum((weight[index, :] for index in factors), weight.new_zeros(weight.shape[1]))
    return result


def get_parameters(layers: Iterable[nn.Module]) -> list[nn.Parameter]:
    return [parameter for layer in layers for parameter in layer.parameters()]


def inventory_context_from_sparse(white_indices, black_indices, white_to_move):
    """Recover exact pool and uncertainty counts from the public sparse rows."""
    device = white_indices.device
    batch = white_indices.shape[0]

    def for_perspective(indices):
        indices = torch.as_tensor(indices, device=device).long()
        valid = (indices >= 0) & (indices < features_v11.REAL_INPUTS)
        local = torch.remainder(indices.clamp_min(0), features_v11.PS_NB)

        def count(offset):
            return ((local == offset) & valid).sum(dim=1).float()

        own_pool = torch.stack([
            count(features_v11.BASE_PS_NB + features_v11.REST_OWN_OFFSET + ptype)
            for ptype in range(6)
        ], dim=1) / torch.tensor(_POOL_MAXIMA, device=device)
        enemy_pool = torch.stack([
            count(features_v11.BASE_PS_NB + features_v11.REST_THEM_OFFSET + ptype)
            for ptype in range(6)
        ], dim=1) / torch.tensor(_POOL_MAXIMA, device=device)

        dark_us = ((local >= features_v11.PS_DARK_US)
                   & (local < features_v11.PS_DARK_US + features_v11.NUM_SQ)
                   & valid).sum(dim=1, keepdim=True).float() / 16.0
        dark_them = ((local >= features_v11.PS_DARK_THEM)
                     & (local < features_v11.PS_DARK_THEM + features_v11.NUM_SQ)
                     & valid).sum(dim=1, keepdim=True).float() / 16.0
        unknown_us = torch.stack([
            count(features_v11.BASE_PS_NB + 40 + value) * value
            for value in range(16)
        ], dim=1).sum(dim=1, keepdim=True) / 15.0
        unknown_them = torch.stack([
            count(features_v11.BASE_PS_NB + 56 + value) * value
            for value in range(16)
        ], dim=1).sum(dim=1, keepdim=True) / 15.0
        context = torch.cat((own_pool, enemy_pool, dark_us, dark_them,
                             unknown_us, unknown_them), dim=1)
        # The native context tower consumes the same unsigned Q0.7 inputs.
        context = torch.clamp(context, 0.0, 1.0)
        return torch.floor(context * 127.0 + 0.5) / 127.0

    white_context = for_perspective(white_indices)
    black_context = for_perspective(black_indices)
    side = torch.as_tensor(white_to_move, device=device, dtype=white_context.dtype).reshape(batch, 1)
    return side * white_context + (1.0 - side) * black_context


@dataclass(frozen=True)
class ForwardComponents:
    accumulator: torch.Tensor
    head: torch.Tensor
    prediction: torch.Tensor

    def __iter__(self):
        return iter((self.accumulator, self.head, self.prediction))


class LayerStacks(nn.Module):
    """Sixteen independent Stockfish-style SFNN heads."""

    def __init__(self, count: int, l1: int, l2: int, l3: int) -> None:
        super().__init__()
        self.count = int(count)
        self.l1_dim, self.l2_dim, self.l3_dim = int(l1), int(l2), int(l3)
        if (self.l1_dim, self.l2_dim, self.l3_dim) != (1024, 32, 32):
            raise ValueError("V11 head dimensions are fixed at 1024/32/32")
        self.l1 = nn.Linear(HEAD_INPUTS, self.l2_dim * self.count)
        self.l2 = nn.Linear(self.l2_dim * 2, self.l3_dim * self.count)
        self.output = nn.Linear(self.l2_dim * 2 + self.l3_dim * 2, self.count)
        self._init_layers()

    def _init_layers(self) -> None:
        with torch.no_grad():
            self.output.bias.zero_()

    def reset_independent_heads(self, seed: int) -> None:
        base_seed = _normalized_seed(seed)
        with torch.no_grad():
            for index in range(self.count):
                generator = _cpu_generator((base_seed + index * _HEAD_SEED_STEP) & _UINT64_MASK)
                l1_weight = torch.empty((self.l2_dim, self.l1.in_features), dtype=torch.float32)
                l2_weight = torch.empty((self.l3_dim, self.l2.in_features), dtype=torch.float32)
                output_weight = torch.empty((1, self.output.in_features), dtype=torch.float32)
                nn.init.kaiming_uniform_(l1_weight, a=math.sqrt(5), generator=generator)
                nn.init.kaiming_uniform_(l2_weight, a=math.sqrt(5), generator=generator)
                nn.init.orthogonal_(output_weight, generator=generator)
                output_weight.mul_(_OUTPUT_WEIGHT_RANGE)
                a, b = index * self.l2_dim, (index + 1) * self.l2_dim
                self.l1.weight[a:b].copy_(l1_weight.to(self.l1.weight))
                self.l1.bias[a:b].zero_()
                self.l2.weight[a:b].copy_(l2_weight.to(self.l2.weight))
                self.l2.bias[a:b].zero_()
                self.output.weight[index:index + 1].copy_(output_weight.to(self.output.weight))
                self.output.bias[index:index + 1].zero_()

    @staticmethod
    def _selection(indices, blend, count, batch_size, device):
        indices = torch.as_tensor(indices, device=device)
        if indices.ndim == 2 and tuple(indices.shape) == (batch_size, 1):
            indices = indices[:, 0]
        if indices.ndim != 1 or indices.shape[0] != batch_size or indices.dtype not in {
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}:
            raise ValueError("V11 layer-stack indices must be integral [batch]")
        indices = indices.long()
        if bool(((indices < 0) | (indices >= count)).any().detach().item()):
            raise ValueError("V11 layer-stack index is outside [0, 16)")
        if blend is None:
            alpha = torch.zeros(batch_size, device=device)
        else:
            alpha = torch.as_tensor(blend, device=device).reshape(-1).float()
            if alpha.shape[0] != batch_size or not bool(torch.isfinite(alpha).all().detach().item()):
                raise ValueError("V11 layer-stack blend shape or finiteness is invalid")
            if bool(((alpha < 0) | (alpha > 1)).any().detach().item()):
                raise ValueError("V11 layer-stack blend is outside [0, 1]")
            alpha = torch.where(indices == count - 1, torch.zeros_like(alpha), alpha)
        return indices, alpha

    def _all_heads(self, x: torch.Tensor):
        batch = x.shape[0]
        l1 = self.l1(x).reshape(batch, self.count, self.l2_dim)
        sq0 = torch.clamp(l1.square(), 0.0, 127.0 / 128.0)
        cl0 = torch.clamp(l1, 0.0, 127.0 / 128.0)
        l2_in = torch.cat([sq0, cl0], dim=2)
        l2 = torch.einsum("bki,koi->bko", l2_in, self.l2.weight.reshape(self.count, self.l3_dim, -1))
        l2 = l2 + self.l2.bias.reshape(self.count, self.l3_dim).unsqueeze(0)
        sq1 = torch.clamp(l2.square(), 0.0, 127.0 / 128.0)
        cl1 = torch.clamp(l2, 0.0, 127.0 / 128.0)
        final_in = torch.cat([sq0, cl0, sq1, cl1], dim=2)
        output = torch.einsum("bki,ki->bk", final_in, self.output.weight)
        output = output + self.output.bias.unsqueeze(0)
        return output + l1[..., 30] - l1[..., 31], l1

    def forward_interpolated(self, x, ls_indices, blend=None):
        if x.ndim != 2 or x.shape[1] != HEAD_INPUTS:
            raise ValueError(f"V11 layer-stack input must have shape [batch, {HEAD_INPUTS}]")
        values, _ = self._all_heads(x)
        indices, alpha = self._selection(ls_indices, blend, self.count, x.shape[0], x.device)
        next_indices = torch.minimum(indices + 1, indices.new_tensor(self.count - 1))
        result = values.gather(1, indices[:, None])
        next_value = values.gather(1, next_indices[:, None])
        return result + (next_value - result) * alpha[:, None]

    def forward(self, x, ls_indices, blend=None):
        return self.forward_interpolated(x, ls_indices, blend)

    def get_coalesced_layer_stacks(self):
        for index in range(self.count):
            l1 = nn.Linear(HEAD_INPUTS, 32)
            l2 = nn.Linear(64, 32)
            output = nn.Linear(128, 1)
            with torch.no_grad():
                a, b = index * 32, (index + 1) * 32
                l1.weight.copy_(self.l1.weight[a:b]); l1.bias.copy_(self.l1.bias[a:b])
                l2.weight.copy_(self.l2.weight[a:b]); l2.bias.copy_(self.l2.bias[a:b])
                output.weight.copy_(self.output.weight[index:index + 1]); output.bias.copy_(self.output.bias[index:index + 1])
            yield l1, l2, output


class NNUE(_LightningBase):
    """V11 training module with Stockfish SFNN heads and no PSQT side channel."""

    def __init__(self, feature_set: features_v11.FeatureSetV11, lambda_: float = 1.0,
                 lr: float = 1.5e-3, head_init_seed: int = 0) -> None:
        super().__init__()
        if feature_set.name not in features_v11.FEATURE_NAMES:
            raise ValueError("NNUE V11 accepts only V11 feature sets")
        self.num_ls_buckets = int(feature_set.num_ls_buckets)
        self.l1_dim, self.l2_dim, self.l3_dim = dims_for_feature_set(feature_set)
        self.ft_dim = int(getattr(feature_set, "ft_dim", DEFAULT_L1 * 2) or DEFAULT_L1 * 2)
        if (self.ft_dim, self.l1_dim, self.l2_dim, self.l3_dim) != (2048, 1024, 32, 32):
            raise ValueError("V11 architecture is fixed at FT=2048 L1=1024 L2=32 L3=32")
        self.pairwise_ft = self.ft_dim == self.l1_dim * 2
        self.use_jieqi_squared = True
        self.feature_set = feature_set
        self.input = DoubleFeatureTransformerSlice(feature_set.num_features, self.ft_dim)
        self.context_tower = nn.Linear(CONTEXT_INPUTS, CONTEXT_HIDDEN)
        self.layer_stacks = LayerStacks(self.num_ls_buckets, self.l1_dim, self.l2_dim, self.l3_dim)
        self.lambda_ = float(lambda_)
        self.lr = float(lr)
        self.ranger21_num_batches_per_epoch: int | None = None
        self.ranger21_num_epochs: int | None = None
        self.weight_clipping = [
            {"params": [self.context_tower.weight], "min_weight": -127 / 128, "max_weight": 127 / 128},
            {"params": [self.layer_stacks.l1.weight], "min_weight": -127 / 128, "max_weight": 127 / 128},
            {"params": [self.layer_stacks.l2.weight], "min_weight": -127 / 64, "max_weight": 127 / 64},
            {"params": [self.layer_stacks.output.weight], "min_weight": -127 / 128, "max_weight": 127 / 128},
        ]
        self._init_layers()
        object.__setattr__(self, "_v11_head_balance", (1.0,) * self.num_ls_buckets)
        object.__setattr__(self, "_v11_head_balance_configured", False)
        object.__setattr__(self, "_v11_head_init_seed", int(head_init_seed))
        self.reset_independent_heads(head_init_seed)

    def _zero_virtual_feature_weights(self) -> None:
        if not self.feature_set.num_virtual_features:
            return
        with torch.no_grad():
            for begin, end in self.feature_set.get_virtual_feature_ranges():
                self.input.weight[begin:end].zero_()

    def _init_layers(self) -> None:
        with torch.no_grad():
            self._zero_virtual_feature_weights()

    def reset_independent_heads(self, seed: int) -> None:
        """Deterministically initialise all phase heads independently."""

        self.layer_stacks.reset_independent_heads(seed)
        object.__setattr__(self, "_v11_head_init_seed", int(_normalized_seed(seed)))

    def set_head_balance(self, weights: Iterable[float]) -> None:
        """Install sixteen finite positive head weights, normalised to sum 16."""

        try:
            values = tuple(float(value) for value in weights)
        except (TypeError, ValueError) as exc:
            raise ValueError("head balance must contain 16 finite positive weights") from exc
        if len(values) != self.num_ls_buckets:
            raise ValueError(f"head balance must contain exactly {self.num_ls_buckets} weights")
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("head balance weights must be finite and positive")
        total = math.fsum(values)
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("head balance weights must have positive sum")
        normalized = tuple(value * self.num_ls_buckets / total for value in values)
        object.__setattr__(self, "_v11_head_balance", normalized)
        object.__setattr__(self, "_v11_head_balance_configured", True)

    def head_balance(self) -> tuple[float, ...]:
        return tuple(getattr(self, "_v11_head_balance", (1.0,) * self.num_ls_buckets))

    def _effective_head_weights(self, layer_stack_indices: torch.Tensor,
                                layer_stack_blend: torch.Tensor | None,
                                *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        indices = torch.as_tensor(layer_stack_indices, device=device)
        if indices.ndim == 2 and indices.shape[1] == 1:
            indices = indices[:, 0]
        if indices.ndim != 1:
            raise ValueError("layer-stack bucket tensor must have shape [batch, 1] or [batch]")
        if indices.dtype not in {
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
        }:
            raise ValueError("layer-stack bucket tensor must be integral")
        indices = indices.long()
        if bool(((indices < 0) | (indices >= self.num_ls_buckets)).any().detach().item()):
            raise ValueError("layer-stack bucket index is outside [0, 16)")
        weights = torch.as_tensor(self.head_balance(), device=device, dtype=dtype)
        current = weights[indices]
        if layer_stack_blend is None:
            return current.unsqueeze(1)
        alpha = torch.as_tensor(layer_stack_blend, device=device, dtype=dtype)
        if alpha.ndim == 2 and alpha.shape[1] == 1:
            alpha = alpha[:, 0]
        if alpha.ndim != 1 or alpha.shape[0] != indices.shape[0]:
            raise ValueError("layer-stack blend tensor must have shape [batch, 1] or [batch]")
        if not bool(torch.isfinite(alpha).all().detach().item()) or bool(((alpha < 0) | (alpha > 1)).any().detach().item()):
            raise ValueError("layer-stack blend must be finite and in [0, 1]")
        next_weights = weights[torch.minimum(indices + 1, indices.new_tensor(self.num_ls_buckets - 1))]
        alpha = torch.where(indices == self.num_ls_buckets - 1, torch.zeros_like(alpha), alpha)
        return ((1.0 - alpha) * current + alpha * next_weights).unsqueeze(1)

    def _clip_weights(self) -> None:
        with torch.no_grad():
            for group in self.weight_clipping:
                for parameter in group["params"]:
                    data = parameter.data
                    minimum, maximum = group.get("min_weight"), group.get("max_weight")
                    virtual = group.get("virtual_params")
                    if virtual is not None:
                        repeats = data.shape[0] // virtual.shape[0]
                        expanded = virtual.data.repeat(repeats, 1)
                        if minimum is not None:
                            data.copy_(torch.maximum(data, data.new_full(data.shape, minimum) - expanded))
                        if maximum is not None:
                            data.copy_(torch.minimum(data, data.new_full(data.shape, maximum) - expanded))
                    else:
                        data.clamp_(minimum, maximum)

    def set_feature_set(self, new_feature_set: features_v11.FeatureSetV11) -> None:
        if new_feature_set.name == self.feature_set.name:
            return
        if self.feature_set.name == "HalfKAv2_hm_jieqi_v11" and new_feature_set.name == "HalfKAv2_hm_jieqi_v11^":
            with torch.no_grad():
                padding = self.input.weight.new_zeros((features_v11.VIRTUAL_INPUTS, self.input.weight.shape[1]))
                self.input.weight = nn.Parameter(torch.cat([self.input.weight, padding], dim=0))
            self.feature_set = new_feature_set
            return
        raise ValueError(f"cannot change V11 feature set from {self.feature_set.name} to {new_feature_set.name}")

    def forward_components(self, us, them, white_indices, white_values,
                           black_indices, black_values,
                           layer_stack_indices, layer_stack_blend=None) -> ForwardComponents:
        wp, bp = self.input(white_indices, white_values, black_indices, black_values)
        w, b = wp, bp
        if self.pairwise_ft:
            w0, w1 = torch.split(torch.clamp(w, 0.0, 1.0), self.l1_dim, dim=1)
            b0, b1 = torch.split(torch.clamp(b, 0.0, 1.0), self.l1_dim, dim=1)
            w, b = w0 * w1 * (127.0 / 128.0)**2, b0 * b1 * (127.0 / 128.0)**2
        batch_size = int(w.shape[0])
        def column(value, name):
            value = torch.as_tensor(value, device=w.device, dtype=w.dtype)
            if value.ndim == 1 and value.shape[0] == batch_size:
                value = value.unsqueeze(1)
            if tuple(value.shape) != (batch_size, 1):
                raise ValueError(f"V11 {name} must have shape [batch, 1]")
            return value
        us = column(us, "us")
        them = column(them, "them")
        accumulator = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
        clipped = torch.clamp(accumulator, 0.0, 1.0)
        context = inventory_context_from_sparse(white_indices, black_indices, us)
        context_hidden = torch.clamp(self.context_tower(context), 0.0, 127.0 / 128.0)
        head_input = torch.cat((clipped, context_hidden), dim=1)
        head = self.layer_stacks.forward_interpolated(head_input, layer_stack_indices, layer_stack_blend)
        return ForwardComponents(accumulator=accumulator, head=head, prediction=head)

    def forward(self, us, them, white_indices, white_values, black_indices,
                black_values, layer_stack_indices,
                layer_stack_blend=None):
        return self.forward_components(
            us, them, white_indices, white_values, black_indices, black_values,
            layer_stack_indices, layer_stack_blend).prediction

    @staticmethod
    def _unpack_batch(batch):
        if isinstance(batch, dict):
            values = [batch[key] for key in ("us", "them", "white_indices", "white_values", "black_indices", "black_values", "outcome", "score", "layer_stack_indices", "layer_stack_blend")]
            return tuple(values) + (batch.get("eval_weight"),)
        if len(batch) == 10:
            return tuple(batch) + (None,)
        if len(batch) == 11:
            return tuple(batch)
        raise ValueError("V11 batch must contain 10 tensors plus optional eval_weight")

    def step_(self, batch, loss_type: str):
        self._clip_weights()
        us, them, white_indices, white_values, black_indices, black_values, outcome, score, layer_stack_indices, layer_stack_blend, eval_weight = self._unpack_batch(batch)
        prediction = self(us, them, white_indices, white_values, black_indices, black_values, layer_stack_indices, layer_stack_blend)
        if prediction.ndim == 1:
            prediction = prediction.unsqueeze(1)
        if prediction.ndim != 2 or prediction.shape[1] != 1:
            raise ValueError(
                f"V11 prediction must have shape [batch, 1], got {tuple(prediction.shape)}")
        batch_size = int(prediction.shape[0])

        def target_column(value, name: str) -> torch.Tensor:
            value = torch.as_tensor(value, device=prediction.device,
                                    dtype=prediction.dtype)
            if value.ndim == 1 and value.shape[0] == batch_size:
                value = value.unsqueeze(1)
            if tuple(value.shape) != tuple(prediction.shape):
                raise ValueError(
                    f"V11 {name} must have shape [batch, 1], got {tuple(value.shape)}")
            return value

        outcome = target_column(outcome, "outcome")
        score = target_column(score, "score")
        q = (prediction * 600.0 / 361.0).sigmoid()
        p = (score / 410.0).sigmoid()
        eval_loss = (p - q).square()
        result_loss = (q - outcome).square()
        loss = self.lambda_ * eval_loss + (1.0 - self.lambda_) * result_loss
        if eval_weight is not None or layer_stack_blend is not None or getattr(self, "_v11_head_balance_configured", False):
            if eval_weight is None:
                weight = torch.ones_like(loss)
            else:
                weight = eval_weight.to(loss.device, dtype=loss.dtype)
                if weight.ndim == 1 and weight.shape[0] == loss.shape[0]:
                    weight = weight.unsqueeze(1)
                if tuple(weight.shape) != tuple(loss.shape):
                    raise ValueError("V11 eval_weight must have shape [batch, 1]")
                if not bool(torch.isfinite(weight).all().detach().item()):
                    raise ValueError("V11 eval_weight must be finite")
                weight = weight.clamp_min(0.0)
            weight = weight * self._effective_head_weights(
                layer_stack_indices, layer_stack_blend,
                dtype=loss.dtype, device=loss.device)
            denominator = weight.sum()
            if bool((denominator > 0).detach().item()):
                loss = (loss * weight).sum() / denominator
            else:
                loss = loss.sum() * 0.0
        else:
            loss = loss.mean()
        if hasattr(self, "log"):
            self.log(loss_type, loss, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step_(batch, "train_loss")

    def validation_step(self, batch, batch_idx):
        return self.step_(batch, "val_loss")

    def test_step(self, batch, batch_idx):
        return self.step_(batch, "test_loss")

    def configure_optimizers(self):
        lr = self.lr
        parameters = [
            {"params": get_parameters([self.input]), "lr": lr},
            {"params": get_parameters([self.context_tower]), "lr": lr},
            {"params": [self.layer_stacks.l1.weight, self.layer_stacks.l1.bias], "lr": lr},
            {"params": [self.layer_stacks.l2.weight, self.layer_stacks.l2.bias], "lr": lr},
            {"params": [self.layer_stacks.output.weight, self.layer_stacks.output.bias], "lr": lr / 10.0},
        ]
        if ranger21 is not None and self.ranger21_num_batches_per_epoch and self.ranger21_num_epochs:
            optimizer = ranger21.Ranger21(parameters, lr=1.0, betas=(0.9, 0.999), eps=1e-7,
                                          num_batches_per_epoch=self.ranger21_num_batches_per_epoch,
                                          num_epochs=self.ranger21_num_epochs, use_warmup=False,
                                          warmdown_active=False, using_gc=False, using_normgc=False,
                                          use_adaptive_gradient_clipping=False, softplus=False,
                                          pnm_momentum_factor=0.0, weight_decay=0.0, logging_active=False)
        else:
            optimizer = torch.optim.AdamW(parameters, lr=1.0, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.987)
        return [optimizer], [scheduler]

    def architecture_contract(self) -> dict[str, object]:
        return {
            "architecture_sha256": ARCHITECTURE_SHA256,
            "ft_dim": self.ft_dim,
            "l1": self.l1_dim,
            "l2": self.l2_dim,
            "l3": self.l3_dim,
            "psqt_buckets": 0,
            "layer_stacks": self.num_ls_buckets,
            "real_features": self.feature_set.num_real_features,
            "training_features": self.feature_set.num_features,
            "metadata_layout": "pool-dark-owner-exact-loss-context-v1",
            "inventory_context": {"inputs": CONTEXT_INPUTS, "hidden": CONTEXT_HIDDEN,
                                   "normalization": "per-side-inventory-maxima-v1"},
            "layer_stack_selection": "continuous-q0.8-v1",
            "layer_stack_q_format": "Q0.8",
        }


def load_v11_checkpoint(path: str, feature_set: features_v11.FeatureSetV11, *, map_location: str = "cpu") -> NNUE:
    """Load a checkpoint only after the caller has validated its V11 sidecar."""

    if pl is not None and str(path).lower().endswith(".ckpt"):
        return NNUE.load_from_checkpoint(path, feature_set=feature_set, map_location=map_location)
    value = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(value, NNUE):
        raise TypeError("V11 checkpoint does not contain model_v11.NNUE")
    value.set_feature_set(feature_set)
    return value
