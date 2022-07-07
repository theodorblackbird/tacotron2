import torch
import torch.nn.functional as F

class EncConvLayer(torch.nn.Module):
    def __init__(self,
            char_embedding_size,
            filters,
            kernel_size,
            dropout_rate) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = torch.nn.Conv1d(
                char_embedding_size,
                filters,
                kernel_size,
                padding=padding)
        self.bn = torch.nn.BatchNorm1d(filters)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = F.relu(y)
        y = self.dropout(y)
        return y


