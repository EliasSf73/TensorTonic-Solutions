import numpy as np

def get_alpha_bar(betas: list[float]) -> list[float]:
    """
    Returns the cumulative alpha-bar values rounded to six decimals.
    """
    alphas= 1-np.array(betas)
    alpha_bar= np.cumprod(alphas)
    
    return np.round(alpha_bar,6).tolist()

def forward_diffusion(x_0: list, t: int, betas: list[float], epsilon: list) -> list:
    """
    Returns x_t with the same nested shape as x_0.
    """
    alpha_bar_t= get_alpha_bar(betas)[t-1]
    x_t= np.sqrt(alpha_bar_t)*np.array(x_0) + np.sqrt(1-alpha_bar_t)*np.array(epsilon)
    return np.round(x_t,4).tolist()