import numpy as np

def reverse_step(x_t: list, t: int, epsilon_pred: list, betas: list[float], z: list = None) -> list:
    """
    Returns x at timestep t - 1, rounded to four decimals.
    """
    x_t= np.array(x_t,dtype=np.float64)
    eps=np.array(epsilon_pred, dtype=np.float64)
    betas= np.array(betas, dtype=np.float64)
    
    alphas=1-betas
    alpha_bar= np.cumprod(alphas)
                # t is one based
    beta_t= betas[t-1]
    alpha_t= alphas[t-1]
    alpha_bar_t= alpha_bar[t-1]
    # prediction
    mu_t= (x_t-beta_t/np.sqrt(1-alpha_bar_t)*eps)/np.sqrt(alpha_t)
    # when to add randomness
    if t>1:
        x_prev= mu_t + np.sqrt(beta_t)*np.array(z, dtype=np.float64)
    else:
        x_prev= mu_t # last step: no noise added
    return np.round(x_prev,4).tolist()
        
    
    




    
   