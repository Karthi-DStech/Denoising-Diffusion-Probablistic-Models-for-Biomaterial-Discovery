import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils.images_utils import sample_batch
from utils.images_utils import train_dataset



class DiffusionModel():
    """
    This class implements the diffusion model for image generation.
    
    Parameters
    ----------
    T : int
        Number of time steps for diffusion
        
    model : nn.Module
        Function approximator model for the diffusion model
        
    device : str
        Device to run the model on
        
    Attributes
    ----------
        
    beta : torch.Tensor
        Beta values for the diffusion model
        
    alpha : torch.Tensor
        Alpha values for the diffusion model
        
    alpha_bar : torch.Tensor
        Cumulative product of alpha values for the diffusion model
        
    Methods
    -------
    training_steps(batch_size, optimizer)
        Perform training steps for the diffusion model
        
    sampling(n_samples, image_channels, img_size, use_tqdm)
        Perform sampling for the diffusion model
        
    
    """


    def __init__(self, T: int, model: nn.Module, device: str):
        super(DiffusionModel, self).__init__()

        self.T = T
        self.function_approximator = model.to(device)
        self.device = device

        self.beta = torch.linspace(1e-4, 0.02, T).to(device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def training_steps(self, batch_size, optimizer):
        """
        This function performs training steps for the diffusion model
        
        Parameters
        ----------
        batch_size : int
            Batch size for training
            
        optimizer : torch.optim
            Optimizer for training the model
            
        Returns
        -------
        float
            Loss value for the training step
        """

        x0 = sample_batch(batch_size, train_dataset, self.device)
        t = torch.randint(1, self.T + 1, (batch_size,), device=self.device, dtype=torch.long)
        eps = torch.randn_like(x0)

        alpha_bar_t = self.alpha_bar[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        eps_predicted = self.function_approximator(torch.sqrt(
            alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps, t-1)
        loss = F.mse_loss( eps, eps_predicted)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()


    @torch.no_grad()
    def sampling(self, n_samples=1, image_channels=1, img_size=(32, 32), use_tqdm=True):
        
        """
        This function performs sampling for the diffusion model
        
        Parameters
        ----------
        n_samples : int
            Number of samples to generate
            
        image_channels : int
            Number of channels in the image
            
        img_size : tuple
            Size of the image
            
        use_tqdm : bool
            Whether to use tqdm for progress bar
            
        Returns
        -------
        torch.Tensor
            Generated samples
        """
        
        x = torch.randn((n_samples, image_channels, img_size[0], img_size[1]), 
                         device=self.device)
        
        progress_bar = tqdm if use_tqdm else lambda x : x
        for t in progress_bar(range(self.T, 0, -1)):
            z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
            
            t = torch.ones(n_samples, dtype=torch.long, device=self.device) * t 
            
            beta_t = self.beta[t-1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            alpha_t = self.alpha[t-1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            alpha_bar_t = self.alpha_bar[t-1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            
            mean = 1 / torch.sqrt(alpha_t) * (x - ((1 - alpha_t) / torch.sqrt(
                1 - alpha_bar_t)) * self.function_approximator(x, t-1))
            sigma = torch.sqrt(beta_t)
            x = mean + sigma * z
    
        return x