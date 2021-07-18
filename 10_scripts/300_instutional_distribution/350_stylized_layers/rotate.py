import torch
import torch.nn as nn

import numpy as np



class rotate(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.m = nn.Parameter(torch.rand(2,2))
        self.n = torch.rand(2,2)
#         self.m = nn.Linear(2, 2)
        
    def forward(self, v):
        
#         out = self.m.detach().numpy() @ v.detach().numpy()
#         return torch.tensor(out)
        out = self.m @ v
        out = self.n @ out
        return out
#         return self.m(v)
        
        
        
        
        
if __name__ == "__main__":
    
    model = rotate()
    
    print(list(model.parameters()))
#     loss_fn = nn.MSELoss()
#     learning_rate = 1e-1
#     optimizer = torch.optim.Adam(
#       model.parameters(), 1e-4, weight_decay=1e-5, amsgrad=True)
    
    v = torch.tensor([1.,0.])
    out = model(v)
    print(out.requires_grad)
#     y = torch.tensor([0.,1.])
    
#     model.train()
#     for it in range(30):
        
#         y_pred = model(v)
#         loss = loss_fn(y_pred, y)
#         model.zero_grad()
#         loss.backward()
#         with torch.no_grad():
#             for param in model.parameters():
#                 param -= learning_rate * param.grad
        

#         print(loss.detach().item())
        
    
#     model.eval() 
#     print(model(v))

    
    