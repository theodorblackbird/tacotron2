import torch

class EncConvLayer(torch.nn.Module):
    def __init__(self,
            char_embedding_size,
            filters,
            kernel_size,
            dropout_rate) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = torch.nn.Conv1D(
                char_embedding_size,
                filters,
                kernel_size,
                padding=padding)
        self.bn = torch.nn.BatchNorm1d(filters)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
    def call(self, x, training=True):
        y = self.conv(x)
        y = self.bn(y)
        y = tf.nn.relu(y)
        y = self.dropout(y)
        return y


