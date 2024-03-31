import torch
import torch.nn as nn

class DistanceL1Loss(nn.Module):
    def __init__(self):
        super(DistanceL1Loss, self).__init__()
        self.log_2 = torch.log(torch.tensor(2.0))

    def forward(self, mu, log_b, target):
        # Clipping log_b to avoid numerical instability
        log_b = torch.clamp(log_b, min=-3, max=3)  # You can adjust the range as needed
        # This loss function will push the output to be Laplace.
        b = log_b.exp()
        loss = (mu - target).abs() / b + log_b + self.log_2.to(mu.device)
        
        return loss.mean()
    
    



