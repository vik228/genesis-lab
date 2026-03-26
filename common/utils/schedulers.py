"""Learning-rate schedulers including Noam schedule."""
import math

def noam_lr(step, d_model, warmup_steps):
    scale = d_model ** -0.5
    return scale * min(step ** -0.5, step * (warmup_steps ** -1.5))
