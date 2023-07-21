import numpy as np
import matplotlib.pyplot as plt
import tqdm

import torch
import torch.nn as nn
import torchvision
import scipy

import feature_projection
import blocks


class ScoreMatchingModel(nn.Module):
    
    def __init__(
        self,
        batch_size,
        channels,
        image_size,
        dimensions,
        embedding_size,
        groups_number,
        epsilon,
        sigma,
        scale,
        T,
        device
    ):
        super().__init__()
        
        self.device = device
        self.batch_size = batch_size
        self.channels = channels
        self.image_size = image_size
        self.dimensions = dimensions
        self.embedding_size = embedding_size
        self.groups_number = groups_number
        self.epsilon = epsilon
        self.sigma = sigma
        self.scale = scale
        self.T = T
        
        # POSITIONAL EMBEDDING
        self.embed = nn.Sequential(
            feature_projection.FeatureProjection(embedding_size=self.embedding_size, scale=self.scale), 
            nn.Linear(self.embedding_size, self.embedding_size)
        )
        
        # ENCODING
        self.convolution_1 = nn.Conv2d(self.channels, self.dimensions[0], kernel_size=3, stride=1, bias=False)
        self.dense_layer_1 = blocks.DenseLayer(self.embedding_size, self.dimensions[0])
        self.group_normalisation_1 = nn.GroupNorm(4, num_channels=self.dimensions[0])
        
        self.convolution_2 = nn.Conv2d(self.dimensions[0], self.dimensions[1], kernel_size=3, stride=2, bias=False)
        self.dense_layer_2 = blocks.DenseLayer(self.embedding_size, self.dimensions[1])
        self.group_normalisation_2 = nn.GroupNorm(self.groups_number, num_channels=self.dimensions[1])
        
        self.convolution_3 = nn.Conv2d(self.dimensions[1], self.dimensions[2], kernel_size=3, stride=2, bias=False)
        self.dense_layer_3 = blocks.DenseLayer(self.embedding_size, self.dimensions[2])
        self.group_normalisation_3 = nn.GroupNorm(self.groups_number, num_channels=self.dimensions[2])
        
        self.convolution_4 = nn.Conv2d(self.dimensions[2], self.dimensions[3], kernel_size=3, stride=2, bias=False)
        self.dense_layer_4 = blocks.DenseLayer(self.embedding_size, self.dimensions[3])
        self.group_normalisation_4 = nn.GroupNorm(self.groups_number, num_channels=self.dimensions[3])
        
        # DECODING
        self.decoding_convolution_4 = nn.ConvTranspose2d(self.dimensions[3], self.dimensions[2], kernel_size=3, stride=2, bias=False, output_padding=1)
        self.decoding_dense_layer_4 = blocks.DenseLayer(self.embedding_size, self.dimensions[2])
        self.decoding_group_normalisation_4 = nn.GroupNorm(self.groups_number, num_channels=self.dimensions[2])
        
        self.decoding_convolution_3 = nn.ConvTranspose2d(self.dimensions[2] + self.dimensions[2], self.dimensions[1], kernel_size=3, stride=2, bias=False, output_padding=1)
        self.decoding_dense_layer_3 = blocks.DenseLayer(self.embedding_size, self.dimensions[1])
        self.decoding_group_normalisation_3 = nn.GroupNorm(self.groups_number, num_channels=self.dimensions[1])
        
        self.decoding_convolution_2 = nn.ConvTranspose2d(self.dimensions[1] + self.dimensions[1], self.dimensions[0], kernel_size=3, stride=2, bias=False, output_padding=1)
        self.decoding_dense_layer_2 = blocks.DenseLayer(self.embedding_size, self.dimensions[0]) 
        self.decoding_group_normalisation_2 = nn.GroupNorm(self.groups_number, num_channels=self.dimensions[0])
        
        self.final_convolution = nn.ConvTranspose2d(self.dimensions[0] + self.dimensions[0], self.channels, kernel_size=3, stride=1)
     
    
    # Sampling with reverse-time Stochastic Differential Equations    
    def sample_with_SDE(
        self,
        sde_sampling_mode,
        signal_to_noise_ratio,
        
    ):
        t = torch.ones(self.batch_size, device=self.device)
        x_0 = torch.randn(self.batch_size, self.channels, self.image_size, self.image_size, device=self.device) * self.marginal_probability_std(t)[:, None, None, None]
        time_steps = torch.linspace(1.0, self.epsilon, self.T, device=self.device)
        step_size = time_steps[0] - time_steps[1]

        if sde_sampling_mode == "euler_maruyama_only":
            with torch.no_grad():
                for time_step in tqdm(time_steps, desc=f"Denoising timesteps"):
                    t = torch.ones(self.batch_size, device=self.device) * time_step
                    g = self.diffusion_coefficient(t)
                    x_mean = x_0 + (g**2)[:, None, None, None] * self.forward(x=x_0, t=t)
                    x_0 = x_mean + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x_0)
                return x_mean
        
        elif sde_sampling_mode == "langevin_mcmc_and_euler_maruyama":
            with torch.no_grad():
                for time_step in tqdm(time_steps, desc="Denoising timesteps"):
                    t = torch.ones(self.batch_size, device=self.device) * time_step
                    
                    # Langevin MCMC
                    grad = self.forward(x=x_0, t=t)
                    grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                    noise_norm = np.sqrt(np.prod(x_0.shape[1:]))
                    langevin_step_size = 2 * (signal_to_noise_ratio * noise_norm / grad_norm)**2
                    x_0 = x_0 + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x_0)
                    
                    # Euler-Maruyama
                    g = self.diffusion_coefficient(t)
                    x_mean = x_0 + (g**2)[:, None, None, None] * self.forward(x=x_0, t=t) * step_size
                    x_0 = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x_0)
                return x_mean

        else:
            raise ValueError('Invalid sde_sampling_mode. Please choose between "euler_maruyama_only" and "langevin_mcmc_and_euler_maruyama"')
        
        
    # Sampling with Ordinary Differential Equations
    def sample_with_ODE(
        self,
        ode_error_tolerance,
        z
    ):
        t = torch.ones(self.batch_size, device=self.device)
            
        if z is None:
            x_0 = torch.randn(self.batch_size, self.channels, self.size, self.size, device=self.device) * self.marginal_probability_std(t)[:, None, None, None]
        else:
            x_0 = z
            
        def score_evaluation(
            sample, 
            time_steps
        ):
            sample = torch.tensor(sample, device=self.device, dtype=torch.float32).reshape(x_0.shape)
            time_steps = torch.tensor(time_steps, device=self.device, dtype=torch.float32).reshape((sample.shape[0], ))    
            with torch.no_grad():    
                score = self.forward(sample, time_steps)
            return score.cpu().numpy().reshape((-1, )).astype(np.float64)

        def ode_function(
            t, 
            x_0
        ):        
            time_steps = np.ones((x_0.shape[0],)) * t    
            g = self.diffusion_coefficient(torch.tensor(t)).cpu().numpy()
            return  -0.5 * (g**2) * score_evaluation(x_0, time_steps)

        res = scipy.integrate.solve_ivp(
            ode_function, 
            (1.0, self.epsilon), 
            x_0.reshape(-1).cpu().numpy(), 
            rtol=ode_error_tolerance, 
            atol=ode_error_tolerance, 
            method="RK45"
        )  
        
        x = torch.tensor(res.y[:, -1], device=self.device).reshape(x_0.shape)
        return x
   
    def marginal_probability_std(
        self,
        t
    ):
        t = t.clone().detach().to(self.device)
        return torch.sqrt((self.sigma**(2 * t) - 1.0) / 2.0 / np.log(self.sigma))
    
    def diffusion_coefficient(
        self,
        t
    ):
        return self.sigma ** t
    
    
    def swish_activation(
        self, 
        x
    ):
        return x * torch.sigmoid(x)
        
    def display_images(
        self,
        x
    ):
        x = torch.clamp(x, 0.0, 1.0)
        plt.rcParams["figure.dpi"] = 175
        # plt.figure(figsize=(6, 6))
        plt.imshow(torchvision.utils.make_grid(x).cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)       # this is from the Durham template
        plt.axis("off")
        plt.grid(False)
        plt.show()

    def forward(
        self,
        x,
        t
    ):
        # Feature embedding for t
        feature_embedding = self.swish_activation(self.embed(t))
        
        # ENCODING
        x_1 = self.convolution_1(x)
        x_1 += self.dense_layer_1(feature_embedding)
        x_1 = self.group_normalisation_1(x_1)
        x_1 = self.swish_activation(x_1)
        
        x_2 = self.convolution_2(x_1)
        x_2 += self.dense_layer_2(feature_embedding)
        x_2 = self.group_normalisation_2(x_2)
        x_2 = self.swish_activation(x_2)
        
        x_3 = self.convolution_3(x_2)
        x_3 += self.dense_layer_3(feature_embedding)
        x_3 = self.group_normalisation_3(x_3)
        x_3 = self.swish_activation(x_3)
        
        x_4 = self.convolution_4(x_3)
        x_4 += self.dense_layer_4(feature_embedding)
        x_4 = self.group_normalisation_4(x_4)
        x_4 = self.swish_activation(x_4)
        
        # DECODING
        x = self.decoding_convolution_4(x_4)
        x += self.decoding_dense_layer_4(feature_embedding)
        x = self.decoding_group_normalisation_4(x)
        x = self.swish_activation(x)
        
        x = self.decoding_convolution_3(torch.cat([x, x_3], dim=1))
        x += self.decoding_dense_layer_3(feature_embedding)
        x = self.decoding_group_normalisation_3(x)
        x = self.swish_activation(x)
        
        x = self.decoding_convolution_2(torch.cat([x, x_2], dim=1))
        x += self.decoding_dense_layer_2(feature_embedding)
        x = self.decoding_group_normalisation_2(x)
        x = self.swish_activation(x)
        
        x = self.final_convolution(torch.cat([x, x_1], dim=1))
        
        x = x / self.marginal_probability_std(t)[:, None, None, None]
        
        return x