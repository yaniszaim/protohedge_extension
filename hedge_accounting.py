"""Shared hedge-return accounting conventions."""

HEDGE_ACCOUNTING_VERSION = "inventory-step-v1"
STEP_RETURN_MARKER = "hedges_are_step_returns"


def uses_step_returns(market):
    """Return whether ``market["hedges"]`` contains one-period returns."""
    return STEP_RETURN_MARKER in market
