import torch.nn as nn

from hw_tts.model.FastSpeech2.utils import Transpose


class ScalarPredictor(nn.Module):
    def __init__(self, embed_dim, 
                 predictor_filter_size,
                 predictor_kernel_size,
                 dropout, **kwargs):
        super().__init__()

        self.input_size = embed_dim
        self.filter_size = predictor_filter_size
        self.kernel = predictor_kernel_size
        self.conv_output_size = predictor_filter_size
        self.dropout = dropout

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                self.input_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                self.filter_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_output):
        encoder_output = self.conv_net(encoder_output)
            
        out = self.linear_layer(encoder_output)
        out = self.relu(out)
        out = out.squeeze(-1)
        if not self.training:
            out = out.unsqueeze(0)
        return out


# import torch
# dur_predictor = ScalarPredictor(256, 256, 3, 0.1)

# inp_tensor = torch.rand(
#     2, # batch_size
#     12, #seq_len
#     256,
#     dtype=torch.float32
# )
# dur_prediction = dur_predictor(inp_tensor)
# print(dur_prediction)