from typing import Callable, Iterable, Tuple
import math
import numpy as np

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                # alpha is step size
                alpha = group["lr"]

                # Complete the implementation of AdamW here, reading and saving
                # your state in the `state` dictionary above.
                # The hyperparameters can be read from the `group` dictionary
                # (they are lr, betas, eps, weight_decay, as saved in the constructor).
                #
                # 1- Update first and second moments of the gradients
                # 2- Apply bias correction
                #    (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                #     also given in the pseudo-code in the project description).
                # 3- Update parameters (p.data).
                # 4- After that main gradient-based update, update again using weight decay
                #    (incorporating the learning rate again).

                ### TODO
                # Initializing the state
                if len(state) == 0:
                    # Moving average of gradient values m_t
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Moving average of the squared gradient values v_t
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # Time step
                    state['step'] = 0


                m_t, v_t, step = state['exp_avg'], state['exp_avg_sq'], state['step']
                beta1, beta2 = group['betas']


                # We take one step for every theta
                step += 1

                # Calculating the moving averages
                # Here, m_{t-1} and v_{t-1} will always be the previous calculation before the update for the first step, they're 0!
                # We also don't need to account for W and b separately
                m_t = beta1 * m_t + (1 - beta1) * grad
                v_t = beta2 * v_t + (1 - beta2) * (grad*grad)

                # Applying bias correction
                alpha = alpha * (np.sqrt(1 - beta2)/(1 - beta1))

                # Updating params less efficiently
                # m_t_corrected = m_t/(1 - beta1)
                # v_t_corrected = v_t/(1 - beta2)
                # p.data -= alpha * m_t_corrected/(np.sqrt(v_t_corrected) + group['eps'] + group['lr'] * p.data)

                # Updating params
                p_data_t = p.data
                p.data -= alpha * m_t/(np.sqrt(v_t) + group['eps']) + group['lr'] * p.data

                # Updating params using weight decay
                # According to Ilya Loshchilov et al., Adam with decouple weight decay without a scheduled lr is simply adding lambda * theta_{t-1}
                # p.data -= alpha * m_t/(np.sqrt(v_t) + group['eps']) + group['lr'] * p.data
                # p.data += group['weight_decay'] * group['lr']
                # p.data += group['weight_decay'] * p_data_t
                # p.data -= group['lr'] * (p.data + group['weight_decay'])
                # p.data -= group['weight_decay'] * (alpha * m_t/(np.sqrt(v_t) + group['eps']) + group['lr'] * p.data)

                return p.data
                #raise NotImplementedError
        return loss


    # when p.data -= alpha * m_t/(np.sqrt(v_t) + group['eps']) + group['lr'] * p.data for first update
    # tensor([[ 0.5548,  0.8667,  0.0729],
    #         [-0.4472, -0.2951, -0.2717]])
    # tensor([[ 0.5549,  0.6704,  0.3818],
    #         [-0.3780, -0.3042, -0.1660]])

    # when p.data -= alpha * m_t/(np.sqrt(v_t) + group['eps']) + group['lr'] * p_data_t
    # tensor([[ 0.5548,  0.8667,  0.0729],
    #         [-0.4472, -0.2951, -0.2717]])
    # tensor([[ 0.6213,  0.7386,  0.4485],
    #         [-0.4267, -0.3530, -0.2152]])

    # when p.data -= group['lr'] * (p.data + group['weight_decay'])
    # tensor([[ 0.5548,  0.8667,  0.0729],
    #         [-0.4472, -0.2951, -0.2717]])
    # tensor([[ 0.5543,  0.6698,  0.3811],
    #         [-0.3779, -0.3041, -0.1659]])

