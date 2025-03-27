import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange
from tqdm import tqdm

from datasets.dataset import normalize_img, unnormalize_img
from model.util import exists, default, noise_sampling, freq_mix_3d, get_freq_filter


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        num_frames,
        text_use_bert_cls = False,
        channels = 3,
        timesteps = 1000,
        sampling_timesteps=250,
        ddim_sampling_eta=1.,
        loss_type = 'l1',
        use_dynamic_thres = False, # from the Imagen paper
        dynamic_thres_percentile = 0.9,
        null_cond_prob=0.1,
        sigma_zero=False,
        noise_cfg=None,
    ):
        super().__init__()
        self.null_cond_prob = null_cond_prob
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn
        self.sigma_zero = sigma_zero
        self.noise_cfg = noise_cfg

        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        if self.is_ddim_sampling:
            print("using ddim samping with %d steps" % sampling_timesteps)
        self.ddim_sampling_eta = ddim_sampling_eta
        # register buffer helper function that casts float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # text conditioning parameters

        self.text_use_bert_cls = text_use_bert_cls

        # dynamic thresholding when sampling

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile
    
    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, cond = None, cond_scale = 1.):
        x_recon = self.predict_start_from_noise(x, t=t, noise = self.denoise_fn.forward_with_cond_scale(x, t, cond = cond, cond_scale = cond_scale))

        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim = -1
                )

                s.clamp_(min = 1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
        # for the subsequent add_noise calls
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    @torch.inference_mode()
    def p_sample(self, x, t, motion_cond = None, cond_scale = 1., clip_denoised = True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x = x, t = t, clip_denoised = clip_denoised, motion_cond = motion_cond, cond_scale = cond_scale)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.inference_mode()
    def p_sample_loop(self, cond_frames, shape, motion_cond=None, cond_scale = 1.):
        device = self.betas.device

        b = shape[0]
        pred_frames = torch.randn(shape, device=device)
        
        pred_frames = torch.cat([cond_frames, pred_frames], dim=2)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            pred_frames = self.p_sample(pred_frames, torch.full((b,), i, device=device, dtype=torch.long), motion_cond = motion_cond, cond_scale = cond_scale)

        return unnormalize_img(pred_frames)

    @torch.inference_mode()
    def sample(self, gt_frames, cond_frames, cond=None, cond_scale = 1.):
        noise = noise_sampling(shape = (gt_frames.shape), device=gt_frames.device, noise_cfg=self.noise_cfg)
        
        batch_size = cond_frames.shape[0]
        image_size = self.image_size
        channels = self.channels
        num_frames = self.num_frames
        cf = cond_frames.shape[2]
        
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sampling
        return sample_fn(noise, cond_frames,(batch_size, channels, num_frames - cf, image_size, image_size),
                         cond=cond, cond_scale = cond_scale)

    @torch.no_grad()
    def ddim_sampling(self, x, cond_frames, shape, cond=None, cond_scale = 1., clip_denoised=True):
        
        batch, device, total_timesteps, sampling_timesteps, eta = \
        shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = torch.linspace(0., total_timesteps, steps=sampling_timesteps + 2)[:-1]
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        
        if self.sigma_zero == False:
            x = torch.randn(shape, device=device)

        for time, time_next in time_pairs:
            alpha = self.alphas_cumprod_prev[time]
            alpha_next = self.alphas_cumprod_prev[time_next]

            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            
            pred_noise, hidden_f = self.denoise_fn(x, time_cond, cond_frames=cond_frames, cond=cond)
            x_start = self.predict_start_from_noise(x, t=time_cond, noise=pred_noise)
            
            if clip_denoised:
                s = 1.
                if self.use_dynamic_thres:
                    s = torch.quantile(
                        rearrange(x_start, 'b ... -> b (...)').abs(),
                        self.dynamic_thres_percentile,
                        dim=-1
                    )

                    s.clamp_(min=1.)
                    s = s.view(-1, *((1,) * (x_start.ndim - 1)))

                # clip by threshold, depending on whether static or dynamic
                x_start = x_start.clamp(-s, s) / s
            if self.sigma_zero:
                sigma = 0.
            else:
                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = ((1 - alpha_next) - sigma ** 2).sqrt()

            noise = torch.randn_like(x) if time_next > 0 else 0.

            x = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        # img = unnormalize_to_zero_to_one(img)
        return unnormalize_img(x), hidden_f
    
    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start_cond, x_start_pred, t, cond = None, noise = None, clip_denoised=True, **kwargs):
        b, c, f, h, w, device = *x_start_pred.shape, x_start_pred.device
        
        # noise = default(noise, lambda: torch.randn_like(x_start_pred))
        noise = noise_sampling(shape = (b, c, f, h ,w), device=device, noise_cfg=self.noise_cfg)
        x_noisy = self.q_sample(x_start=x_start_pred, t=t, noise=noise)

        pred_noise, _ = self.denoise_fn(x_noisy, t, cond_frames=x_start_cond, cond=cond)
        
        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, pred_noise)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, pred_noise)
        else:
            raise NotImplementedError()
        
        pred_x0 = self.predict_start_from_noise(x_noisy, t, pred_noise)        
        if clip_denoised:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(pred_x0, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                )

                s.clamp_(min=1.)
                s = s.view(-1, *((1,) * (pred_x0.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            pred_x0 = pred_x0.clamp(-s, s) / s

        return loss, pred_x0

    def forward(self, x_cond, x_pred, cond=None, *args, **kwargs):
        b, device, img_size, = x_cond.shape[0], x_cond.device, self.image_size
        
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x_cond, x_pred, t, cond=cond, *args, **kwargs)