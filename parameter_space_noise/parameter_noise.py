import numpy as np
import torch


# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class AdaptiveParamNoiseSpec(object):
    def __init__(
        self,
        initial_stddev=0.2,
        desired_action_stddev=0.2,
        adaptation_coefficient=1.01,
    ):
        """
        Note that initial_stddev and current_stddev refer to std of parameter noise,
        but desired_action_stddev refers to (as name notes) desired std in action space
        """
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adaptation_coefficient = adaptation_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adaptation_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adaptation_coefficient

    def get_stats(self):
        stats = {
            "param_noise_stddev": self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = "AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adaptation_coefficient={})"
        return fmt.format(
            self.initial_stddev, self.desired_action_stddev, self.adaptation_coefficient
        )


def ddpg_distance(actions1, actions2):
    """
    Compute "distance" between actions taken by two policies at the same states
    Expects numpy arrays
    """
    diff = actions1 - actions2
    mean_diff = np.mean(np.square(diff), axis=0)
    dist = np.sqrt(np.mean(mean_diff))
    return dist


def perturb_model(
    perturbed_model: torch.nn.Module, param_noise: AdaptiveParamNoiseSpec
):
    for name, param in perturbed_model.named_parameters():
        if "ln" in name:
            continue
        noise = torch.randn_like(param) * param_noise.current_stddev
        param.data.add_(noise)
