import torch
import torch.nn as nn
import torch.nn.functional as F


def tfp_softclip(
    actions,
    lower,
    upper,
    hinge_softness=1.0,
    outer_clip_cut_off=10.0,
):
    """PyTorch equivalent of the paper code's ``DHSoftClip`` path.

    The original implementation first normalizes each action interval to
    ``[0, 1]``, applies ``tfp.bijectors.SoftClip(0, 1)``, and rescales.  This
    function follows that ordering exactly rather than applying two softplus
    hinges directly in action units.
    """
    if float(hinge_softness) <= 0.0:
        raise ValueError("hinge_softness must be positive")
    if float(outer_clip_cut_off) < 1.0:
        raise ValueError("outer_clip_cut_off must be at least 1")

    actions = torch.as_tensor(actions)
    lower = torch.as_tensor(lower, dtype=actions.dtype, device=actions.device)
    upper = torch.as_tensor(upper, dtype=actions.dtype, device=actions.device)
    if torch.any(upper < lower):
        raise ValueError("upper action bounds must be at least lower bounds")
    if torch.any(upper < 0.0) or torch.any(lower > 0.0):
        raise ValueError("action bounds must contain zero")

    clipped = torch.minimum(actions, upper * float(outer_clip_cut_off))
    clipped = torch.maximum(clipped, lower * float(outer_clip_cut_off))
    width = upper - lower
    safe_width = torch.where(width > 0.0, width, torch.ones_like(width))
    relative = (clipped - lower) / safe_width

    softness = torch.as_tensor(
        hinge_softness,
        dtype=actions.dtype,
        device=actions.device,
    )

    def softplus_hinge(value):
        return softness * F.softplus(value / softness)

    # TFP SoftClip for low=0 and high=1. The denominator is the correction
    # that keeps the lower endpoint inside the requested interval.
    relative = 1.0 - (
        softplus_hinge(1.0 - softplus_hinge(relative))
        / softplus_hinge(torch.ones_like(relative))
    )
    bounded = relative * width + lower
    return torch.where(width > 0.0, bounded, torch.zeros_like(bounded))


def legacy_proto_softclip(
    actions,
    lower,
    upper,
    hinge_softness=1.0,
    outer_clip_cut_off=10.0,
):
    """Preserve the historical PyTorch approximation for sensitivity tests."""
    actions = torch.as_tensor(actions)
    lower = torch.as_tensor(lower, dtype=actions.dtype, device=actions.device)
    upper = torch.as_tensor(upper, dtype=actions.dtype, device=actions.device)
    actions = torch.minimum(actions, upper * float(outer_clip_cut_off))
    actions = torch.maximum(actions, lower * float(outer_clip_cut_off))
    return upper - float(hinge_softness) * F.softplus(
        (
            upper
            - (
                lower
                + float(hinge_softness)
                * F.softplus((actions - lower) / float(hinge_softness))
            )
        )
        / float(hinge_softness)
    )


class SoftClip(nn.Module):
    """
    smooth clipping function used in Deep Hedging
    prevents extreme hedge ratios
    """

    def __init__(self, lower=-1.0, upper=1.0):
        super().__init__()

        self.lower = lower
        self.upper = upper

    def forward(self, x):

        return (
            self.lower
            + (self.upper - self.lower)
            * torch.sigmoid(x)
        )
