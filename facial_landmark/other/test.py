import torch
a = torch.tensor([5.1,2], dtype=torch.float32)
b = torch.tensor([3.2,2], dtype=torch.float32)
import torch.nn as nn 
import torch.optim as opt
loss_fn = nn.MSELoss()
loss = loss_fn(a,b)
loss.requires_grad = True
loss.backward()
print(loss.item())