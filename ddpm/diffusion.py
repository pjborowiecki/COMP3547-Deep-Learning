import torch
import torchvision
import matplotlib.pyplot as plt


class DenoisingDiffusion:
    
    def __init__(
        self, 
        epsilon_theta_model,
        beta_initial,
        beta_final,
        T, 
        device
    ):
        super().__init__()
        
        self.T = T
        self.epsilon_theta_model = epsilon_theta_model
        self.beta = torch.linspace(beta_initial, beta_final, T).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sigma2 = self.beta
        
    def forward_diffusion(
        self, 
        x_0, 
        t,
        epsilon=None
    ):
        if epsilon is None:
            epsilon = torch.randn_like(x_0)
            
        mean = self.gather(self.alpha_bar, t) ** 0.5 * x_0
        variance = 1 - self.gather(self.alpha_bar, t)
        return mean + (variance ** 0.5) * epsilon
    
    def reverse_diffusion(
        self, 
        x_t, 
        t
    ):
        epsilon_theta = self.epsilon_theta_model(x_t, t)
        alpha = self.gather(self.alpha, t)
        alpha_bar = self.gather(self.alpha_bar, t)
        epsilon_coefficient = (1 - alpha) / (1 - alpha_bar) ** .5
        
        mean = 1 / (alpha ** 0.5) * (x_t - epsilon_coefficient * epsilon_theta)
        variance = self.gather(self.sigma2, t)
        epsilon = torch.randn(x_t.shape, device=x_t.device)
        return mean + (variance ** .5) * epsilon

    
    def gather(
        self,
        consts, 
        t
    ):
        c = consts.gather(-1, t)
        return c.reshape(-1, 1, 1, 1)

    def display_images(
        self,
        x
    ):
        x = torch.clamp(x, 0.0, 1.0)
        plt.rcParams["figure.dpi"] = 175
        plt.imshow(torchvision.utils.make_grid(x).cpu().data.permute(0,2,1).contiguous().permute(2,1,0), cmap=plt.cm.binary)
        plt.axis("off")
        plt.grid(False)
        plt.show()