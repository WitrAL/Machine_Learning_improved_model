from .Decoder import Decoder
from .Encoder import Encoder
import torch.nn as nn
from .utils import RevIN
class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.Encoder = Encoder(args)
        self.Decoder = Decoder(args)
        self.out = nn.Linear(args.d_model, 7)
        self.revin = RevIN(7)

    def forward(self, src, tgt, src_mask, tgt_mask): # what does the mask do?

        src = self.revin(src, 'norm') # Revin norm

        encoder_outputs = self.Encoder(src, src_mask)
        decoder_outputs = self.Decoder(encoder_outputs, tgt, src_mask, tgt_mask)
        outputs = self.out(decoder_outputs)

        outputs = self.revin(outputs, 'denorm') #Revin denorm

        return outputs
