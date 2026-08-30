"""Pi_05 model wrapper for the P1-gaze evaluation condition."""

from XPolicyLab.policy.Pi_05.model import Model as Pi05Model


class Model(Pi05Model):
    """Use the frozen Pi_05 model without changing its prompts or weights."""
