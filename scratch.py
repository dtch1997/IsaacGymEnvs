""" Example on calculating Lie derivatives """

import torch
from torch.autograd.functional import jvp
from typing import Callable

TensorFunction = Callable[[torch.Tensor], torch.Tensor]

def f(x: torch.Tensor) -> torch.Tensor:
    """ 
    f: R^n --> R^n 

    Defines a vector field on R^n    
    
    args: 
        x: [..., n]-dim batched tensor
    
    return:
        [..., n]-dim batched tensor
    """
    return x

def V(x: torch.Tensor) -> torch.Tensor:
    """
    V: R^n --> R

    Defines a scalar on R^n

    args: 
        x: [..., n]-dim batched tensor

    return:
        [...]-dim batched tensor
    """
    return torch.norm(x, dim=-1)

def approximate_lie_derivative(
    V: TensorFunction,
    f: TensorFunction,
    x: torch.Tensor 
) -> torch.Tensor:
    """ 
    Approximate the Lie derivative of V w.r.t f, evaluated at x

    args:
        V: TensorCallable, [..., n] -> [...]
        f: TensorCallable, [..., n] -> [..., n]
        x: Tensor, [..., n]
    
    Reference: https://en.wikipedia.org/wiki/Lie_derivative
    'The Lie Derivative of a Function 
    """
    dt = 1e-5
    def P(t, x):
        """ Calculate x evolving according to xdot = f(x) for time duration t
        
        Note: we are using a linear approximation
        could try actually integrating it
        """
        return x + f(x) * t

    lie_deriv = (V(P(dt, x)) - V(x)) / dt
    return lie_deriv

from functorch import vmap, grad

def lie_derivative(
    V: TensorFunction,
    f: TensorFunction,
    x: torch.Tensor 
) -> torch.Tensor:
    """ Calculate the Lie derivative of V w.r.t f, evaluated at x 
    
    args: 
        V: TensorFunction [M, n] -> [M]
        f: TensorFunction [M, n] -> [M, n]
    """
    x.requires_grad_(True)
    dVdx = vmap(grad(V))(x)
    lie_deriv = torch.sum(f(x) * dVdx, dim=-1)
    x.detach()
    return lie_deriv

if __name__ == "__main__":
    # Sanity check the calculated Lie derivative against approximate
    # for N=1000 random samples 
    x_shape = (1000, 2)
    x = torch.normal(mean = torch.zeros(x_shape), std = torch.ones(x_shape))
    
    approx_LfVx = approximate_lie_derivative(V, f, x)
    actual_LfVx = lie_derivative(V, f, x)
    print(approx_LfVx.shape)
    print(actual_LfVx.shape)
    print("Approximate lie deriv: ", approx_LfVx[:10])
    print("Actual lie deriv: ", actual_LfVx[:10])
    assert torch.all(approx_LfVx.isclose(actual_LfVx, rtol=1e-2, atol=1e-2))