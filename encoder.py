import copy

from torch import nn

from embedding import Embedder
from encoderLayer import EncoderLayer
from norm import Norm
from positionalEncoder import PositionalEncoder


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = self.get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)

    def get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)