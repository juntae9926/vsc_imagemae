import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask

class NTXentLoss(nn.Module):
    def __init__(
        self,
        eps=1e-6,
        temperature: float = 0.1,
        tau_plus: float = 0.1,
        debiased = True,
        pos_in_denominator = True
    ):
        super(NTXentLoss, self).__init__()
        self.eps = eps
        self.temperature = temperature
        self.debiased = debiased
        self.pos_in_denominator = pos_in_denominator
        self.tau_plus = tau_plus

    def forward(
            self,
            out_1: torch.Tensor,
            out_2: torch.Tensor,
        ):
        """
        out_1: [batch_size, dim]
        out_2: [batch_size, dim]
        """
        out_1 = F.normalize(out_1, dim=1)
        out_2 = F.normalize(out_2, dim=1)

        # Full similarity matrix
        out = torch.cat([out_1, out_2], dim=0)
        batch_size = out_1.shape[0]
        mask = get_negative_mask(batch_size).cuda()
        cov = torch.mm(out, out.t().contiguous())
        neg  = torch.exp(cov / self.temperature)
        neg = neg.masked_select(mask).view(2 * batch_size, -1)
    
        # Positive similarity
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)

        if self.debiased:
            N = batch_size * 2 - 2
            Ng = (-self.tau_plus * N * pos + neg.sum(dim = -1)) / (1 - self.tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min = N * np.e**(-1 / self.temperature))
            denominator = Ng + pos
        elif self.pos_in_denominator:
            Ng = neg.sum(dim=-1)
            denominator = Ng + pos
        else:
            Ng = neg.sum(dim=-1)
            denominator = Ng
            
        return -torch.log(pos / (denominator + self.eps)).mean()


# 검증 필요 https://github.com/Lightning-AI/lightning-bolts/blob/master/pl_bolts/losses/self_supervised_learning.py
def nt_xent_loss(out_1, out_2, temperature):
    """Loss used in SimCLR."""
    out = torch.cat([out_1, out_2], dim=0)
    n_samples = len(out)

    # Full similarity matrix
    cov = torch.mm(out, out.t().contiguous())
    sim = torch.exp(cov / temperature)

    # Negative similarity
    mask = ~torch.eye(n_samples, device=sim.device).bool()
    neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

    # Positive similarity :
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)
    loss = -torch.log(pos / neg).mean()

    return loss

import torch
import torch.nn.functional as F

class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, query, pos_samples):
        # Flatten the input tensors
        query = query.view(query.size(0), -1)
        pos_samples = pos_samples.view(pos_samples.size(0), -1)

        # Concatenate the query and positive samples
        # along the batch dimension
        pos_samples = torch.cat([query, pos_samples], dim=1)

        # Compute the dot product of the query and positive samples
        pos_dot_product = torch.bmm(pos_samples, query.unsqueeze(2)).squeeze()
        pos_dot_product /= self.temperature

        # Compute the dot product of the query and negative samples
        neg_samples = torch.cat([pos_samples[:, :i, :], pos_samples[:, i+1:, :]], dim=1)
        neg_samples = neg_samples.view(-1, neg_samples.size(2))
        neg_dot_product = torch.mm(neg_samples, query.t())
        neg_dot_product /= self.temperature

        # Compute the contrastive loss using the InfoNCE criterion
        contrastive_logits = torch.cat([pos_dot_product, neg_dot_product], dim=1)
        contrastive_labels = torch.zeros(query.size(0), dtype=torch.long).to(query.device)
        contrastive_loss = F.cross_entropy(contrastive_logits, contrastive_labels)

        return contrastive_loss