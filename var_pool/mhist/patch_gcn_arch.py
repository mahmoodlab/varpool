"""
Modules from https://github.com/mahmoodlab/Patch-GCN/
"""
import torch.nn as nn
import torch
import torch.nn.functional as F


class MIL_Attention_FC_surv(nn.Module):
    def __init__(self, size_arg="small", dropout=0.25, n_classes=4):
        r"""
        Attention MIL Implementation
        Args:
            size_arg (str): Size of NN architecture (Choices: small or large)
            dropout (float): Dropout rate
            n_classes (int): Output shape of NN
        """
        super(MIL_Attention_FC_surv, self).__init__()
        self.size_dict_path = {"small": [1024, 512, 256],
                               "big": [1024, 512, 384]}

        ### Deep Sets Architecture Construction
        size = self.size_dict_path[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(
            L=size[1], D=size[2], dropout=dropout, n_classes=1
        )
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.rho = nn.Sequential(
            *[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)]
        )

        self.classifier = nn.Linear(size[2], n_classes)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(
                self.attention_net, device_ids=device_ids
            ).to("cuda:0")

        self.rho = self.rho.to(device)
        self.classifier = self.classifier.to(device)

    def forward(self, bag):

        A, h_path = self.attention_net(bag)
        A = torch.transpose(A, 1, 0)
        A_raw = A
        A = F.softmax(A, dim=1)
        h_path = torch.mm(A, h_path)
        h_path = self.rho(h_path).squeeze()
        h = h_path  # [256] vector

        logits = self.classifier(h).unsqueeze(0)  # logits needs to be a [1 x 4] vector
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        return hazards, S, Y_hat, None, None


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)
        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x
