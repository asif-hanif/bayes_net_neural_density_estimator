import torch

# Use GPU if available
USE_GPU = True

dtype = torch.float32           # we will be using float throughout this notebook

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Using Device=', device, '\nData Type=',dtype)
