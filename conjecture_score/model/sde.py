import math
import torch
import torch.nn as nn

from scipy import integrate

from model.module import sde_lib
from model.util import to_flattened_numpy, from_flattened_numpy
from datasets.dataset import unnormalize_img


import abc

class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__()
    self.sde = sde
    # Compute the reverse SDE/ODE
    self.rsde = sde.reverse(score_fn, probability_flow)
    self.score_fn = score_fn

  @abc.abstractmethod
  def update_fn(self, x, t):
    """One update of the predictor.

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass

class ReverseDiffusionPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, t, cond_frames, cond):
    f, G = self.rsde.discretize(x, t, cond_frames, cond)
    x_mean = x - f
    z = torch.randn_like(x)
    x = x_mean + G[:, None, None, None, None] * z
    return x, x_mean

class SDEs(nn.Module):
    def __init__(self,
                 score_model,
                 sde_cfg,
                 sampling_cfg,
                 noise_cfg):
        
        super().__init__()
        self.score_model = score_model
        self.sde_cfg = sde_cfg
        self.sampling_cfg = sampling_cfg
        self.noise_cfg = noise_cfg
        
        # define sde
        if sde_cfg.type.lower() == 'vpsde':
            self.sde = sde_lib.VPSDE(beta_min=sde_cfg.beta_min, beta_max=sde_cfg.beta_max, N=sde_cfg.num_scales)
            self.sampling_eps = 1e-3
        elif sde_cfg.type.lower() == 'subvpsde':
            self.sde = sde_lib.subVPSDE(beta_min=sde_cfg.beta_min, beta_max=sde_cfg.beta_max, N=sde_cfg.num_scales)
            self.sampling_eps = 1e-3
        elif sde_cfg.type.lower() == 'vesde':
            self.sde = sde_lib.VESDE(sigma_min=sde_cfg.sigma_min, sigma_max=sde_cfg.sigma_max, N=sde_cfg.num_scales)
            self.sampling_eps = 1e-5

        self.score_func = self.get_score_fn(continuous=sde_cfg.continuous)
    
    
    def noise_sampling(self, shape, device):
        b, c, f, h, w = shape
        
        if self.noise_cfg.noise_sampling_method == 'vanilla':
            noise = torch.randn(shape, device=device)
        elif self.noise_cfg.noise_sampling_method == 'pyoco_mixed':
            noise_alpha_squared = float(self.noise_cfg.noise_alpha) ** 2
            shared_noise = torch.randn((b, c, 1, h, w), device=device) * math.sqrt((noise_alpha_squared) / (1 + noise_alpha_squared))
            ind_noise = torch.randn(shape, device=device) * math.sqrt(1 / (1 + noise_alpha_squared))
            noise = shared_noise + ind_noise
        elif self.noise_cfg.noise_sampling_method == 'pyoco_progressive':
            noise_alpha_squared = float(self.noise_cfg.noise_alpha) ** 2
            noise = torch.randn(shape, device=device)
            ind_noise = torch.randn(shape, device=device) * math.sqrt(1 / (1 + noise_alpha_squared))
            for i in range(1, noise.shape[2]):
                noise[:, :, i, :, :] = noise[:, :, i - 1, :, :] * math.sqrt((noise_alpha_squared) / (1 + noise_alpha_squared)) + ind_noise[:, :, i, :, :]
        else:
            raise ValueError(f"Unknown noise sampling method {self.noise_cfg.noise_sampling_method}")

        return noise

    def get_score_fn(self, continuous):
        if isinstance(self.sde, sde_lib.VPSDE) or isinstance(self.sde, sde_lib.subVPSDE):
            def score_fn(x, t, cond_frames, cond):
                # Scale neural network output by standard deviation and flip sign
                if continuous or isinstance(self.sde, sde_lib.subVPSDE):
                    # For VP-trained models, t=0 corresponds to the lowest noise level
                    # The maximum value of time embedding is assumed to 999 for
                    # continuously-trained models.
                    labels = t * 999
                    score = self.score_model(x, labels, cond_frames, cond)
                    std = self.sde.marginal_prob(torch.zeros_like(x), t)[1]
                else:
                    # For VP-trained models, t=0 corresponds to the lowest noise level
                    labels = t * (self.sde.N - 1)
                    score = self.score_model(x, labels)
                    std = self.sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

                score = -score / std[:, None, None, None, None]
                return score

        elif isinstance(self.sde, sde_lib.VESDE):
            def score_fn(x, t, cond_frames, cond):
                if continuous:
                    labels = self.sde.marginal_prob(torch.zeros_like(x), t)[1]
                else:
                    # For VE-trained models, t=0 corresponds to the highest noise level
                    labels = self.sde.T - t
                    labels *= self.sde.N - 1
                    labels = torch.round(labels).long()

                score = self.score_model(x, labels, cond_frames, cond)
                return score

        else:
            raise NotImplementedError(f"SDE class {self.sde.__class__.__name__} not yet supported.")

        return score_fn
    
    def sde_loss_fn(self, x_cond, x_pred, cond):
        b, c, f, h, w, device = *x_pred.shape, x_pred.device
        reduce_op = torch.mean if self.sde_cfg.reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

        t = torch.rand(x_pred.shape[0], device=x_pred.device) * (self.sde.T - self.sampling_eps) + self.sampling_eps
        z = self.noise_sampling(shape = (b, c, f, h ,w), device=device)
        mean, std = self.sde.marginal_prob(x_pred, t)
        perturbed_data = mean + std[:, None, None, None, None] * z

        score = self.score_func(perturbed_data, t, cond_frames=x_cond, cond=cond)

        if not self.sde_cfg.likelihood_weighting:
            losses = torch.square(score * std[:, None, None, None, None] + z)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = self.sde.sde(torch.zeros_like(x_pred), t)[1] ** 2
            losses = torch.square(score + z / std[:, None, None, None, None])
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

        return torch.mean(losses)
    
    def smld_loss_fn(self, x_cond, x_pred, cond):
        """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
        assert isinstance(self.sde, sde_lib.VESDE), "SMLD training only works for VESDEs."
        b, c, f, h, w, device = *x_pred.shape, x_pred.device

        # Previous SMLD models assume descending sigmas
        smld_sigma_array = torch.flip(self.sde.discrete_sigmas, dims=(0,))
        reduce_op = torch.mean if self.sde_cfg.reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

        labels = torch.randint(0, self.sde.N, (x_pred.shape[0],), device=x_pred.device)
        sigmas = smld_sigma_array.to(x_pred.device)[labels]
        
        z = self.noise_sampling(shape = (b, c, f, h ,w), device=device)
        noise = z * sigmas[:, None, None, None, None]
        perturbed_data = noise + x_pred
        
        score = self.score_func(perturbed_data, labels, cond_frames=x_cond, cond=cond)
        
        target = -noise / (sigmas ** 2)[:, None, None, None, None]
        
        losses = torch.square(score - target)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas ** 2
        return torch.mean(losses)
    
    def ddpm_loss_fn(self, x_cond, x_pred, cond):
        """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
        assert isinstance(self.sde, sde_lib.VPSDE), "DDPM training only works for VPSDEs."
        b, c, f, h, w, device = *x_pred.shape, x_pred.device

        reduce_op = torch.mean if self.sde_cfg.reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

        labels = torch.randint(0, self.sde.N, (x_pred.shape[0],), device=x_pred.device)
        sqrt_alphas_cumprod = self.sde.sqrt_alphas_cumprod.to(x_pred.device)
        sqrt_1m_alphas_cumprod = self.sde.sqrt_1m_alphas_cumprod.to(x_pred.device)
        
        noise = self.noise_sampling(shape = (b, c, f, h ,w), device=device)
        perturbed_data = sqrt_alphas_cumprod[labels, None, None, None, None] * x_pred + \
                        sqrt_1m_alphas_cumprod[labels, None, None, None, None] * noise
                        
        score = self.score_func(perturbed_data, labels, cond_frames=x_cond, cond=cond)
        
        losses = torch.square(score - noise)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        return torch.mean(losses)

    def forward(self, x_cond, x_pred, cond=None):
        # define loss func
        if self.sde_cfg.continuous:
            loss_fn = self.sde_loss_fn
        else:
            assert not self.sde_cfg.likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
            if isinstance(self.sde, sde_lib.VESDE):
                loss_fn = self.smld_loss_fn     
            elif isinstance(self.sde, sde_lib.VPSDE):
                loss_fn = self.ddpm_loss_fn   
            else:
                raise ValueError(f"Discrete training for {self.sde.__class__.__name__} is not recommended.")
        
        return loss_fn(x_cond, x_pred, cond)

    def ode_sampler(self, z, cond_frames, cond):
        z_shape, z_device= z.shape, z.device

        def denoise_update_fn(x, cond_frames, cond):
            # Reverse diffusion predictor for denoising
            predictor_obj = ReverseDiffusionPredictor(self.sde, self.score_func, probability_flow=False)
            vec_eps = torch.ones(x.shape[0], device=x.device) * self.sampling_eps
            _, x = predictor_obj.update_fn(x, vec_eps, cond_frames, cond)
            return x        
        
        def ode_func(t, x, cond_frames, cond):
            x = from_flattened_numpy(x, z_shape).to(z_device).type(torch.float32)
            vec_t = torch.ones(z_shape[0], device=x.device) * t
                    
            # reverse ode : drift func
            rsde = self.sde.reverse(self.score_func, probability_flow=True)
            return to_flattened_numpy(rsde.sde(x, vec_t, cond_frames, cond)[0])
        
        with torch.no_grad():            
            solution = integrate.solve_ivp(ode_func,
                                        (self.sde.T, self.sampling_eps), 
                                        to_flattened_numpy(z),
                                        args = (cond_frames, cond),
                                        rtol=self.sampling_cfg.rtol, atol=self.sampling_cfg.atol, 
                                        method=self.sampling_cfg.method)
            
            x = torch.tensor(solution.y[:, -1]).reshape(z_shape).to(z_device).type(torch.float32)        
            if self.sampling_cfg.noise_removal:
                x = denoise_update_fn(x, cond_frames, cond)
            
            x = unnormalize_img(x)
            return x, solution.nfev
    
    @torch.inference_mode()
    def sample(self, cond_frames, gt_frames, cond=None):
        z = self.noise_sampling(shape = (gt_frames.shape), device=gt_frames.device)
        # z = self.sde.prior_sampling(gt_frames.shape).to(gt_frames.device)
    
        if self.sampling_cfg.method.lower() != "lmm":
            sampler = self.ode_sampler
        else:
            print("error")

        return sampler(z, cond_frames, cond)

