import torch
import torch.nn as nn
import pdb

class PromptPool(nn.Module):
    def __init__(self, length=5, embed_dim=4096, embedding_key='mean', prompt_init='uniform',  
                 use_prompt_pool=False, 
                 prompt_key=False, pool_size=None, prompt_top_k=1, batchwise_prompt=False, prompt_key_init='uniform',
                 device="cuda",pull_constraint_coeff=1):
        super().__init__()

        self.length = length
        self.embed_dim = embed_dim
        self.use_prompt_pool = use_prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init

        self.pool_size = pool_size
        self.top_k = prompt_top_k
        self.batchwise_prompt = batchwise_prompt
        self.pull_constraint_coeff=pull_constraint_coeff


        if self.prompt_init == 'zero':
            
            self.register_parameter("prompt", nn.Parameter(torch.zeros(self.length, self.embed_dim)))

        elif self.prompt_init == 'uniform':
            
            self.register_parameter("prompt", nn.Parameter(torch.randn(self.length, self.embed_dim)))

            nn.init.uniform_(self.prompt)

        

        prompt_mean = torch.mean(self.prompt, dim=1)
        self.prompt_key = prompt_mean 
        self.prompt_key = self.prompt_key.to(device)
        
            

        
    
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, x_embed, prompt_mask=None, cls_features=None):

        out = dict()
        self.prompt_key = torch.mean(self.prompt, dim=1)


        batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)
        
       
        out['total_prompt_len'] = batched_prompt.shape[1]
        out['prompted_embedding'] = batched_prompt 

        return out




