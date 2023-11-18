import torch
import torch.nn as nn
import torch.nn.functional as F


from hw_tts.model.FastSpeech2.scalar_predictor import ScalarPredictor


def create_alignment(expand_max_len, duration_predictor_output):
    base_mat = torch.zeros((
        duration_predictor_output.size(0),
        expand_max_len,
        duration_predictor_output.size(1)
    ))
    B, M, L = base_mat.shape
    arange_tensor = torch.arange(M).view(1, 1, M)

    # Создаем матрицу индексов для каждого примера в батче
    index_matrix = torch.cumsum(duration_predictor_output, dim=1) - 1
    index_matrix = torch.clamp(index_matrix.unsqueeze(2) - arange_tensor, 0, M - 1)

    # Обновляем значения в base_mat
    base_mat.scatter_(1, index_matrix.transpose(1, 2), 1)
    base_mat = base_mat - torch.concat((torch.zeros((B, M, 1)), base_mat[:, :, :-1]), dim=-1)

    return base_mat


class LengthRegulator(nn.Module):
    def __init__(self, encoder_dim, 
                 duration_predictor_filter_size,
                 duration_predictor_kernel_size,
                 dropout, **kwargs):
        super().__init__()
        self.duration_predictor = ScalarPredictor(
            encoder_dim, 
            duration_predictor_filter_size,
            duration_predictor_kernel_size,
            dropout
        )

    def LR(self, x, duration_predictor_output, mel_max_length=None):
        expand_max_len = torch.max(torch.sum(duration_predictor_output, -1), -1)[0].int().item()
        alignment = create_alignment(expand_max_len, duration_predictor_output)
        output = alignment @ x
        if mel_max_length:
            output = F.pad(output, (0, 0, 0, mel_max_length - output.size(1), 0, 0))
        return output

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
        duration_predictor_output = self.duration_predictor(x)
        if target is not None:
            output = self.LR(x, target, mel_max_length)
            return output, duration_predictor_output
        else:
            duration_predictor_output = (alpha * (duration_predictor_output + 0.5)).int()
            output = self.LR(x, duration_predictor_output, mel_max_length)
            mel_pos = torch.stack(
                [torch.Tensor([i+1  for i in range(output.size(1))])]
            ).long().to(x.device)
            return output, mel_pos
        

# import torch
# regulator = LengthRegulator(256, 256, 3)

# inp_tensor = torch.rand(
#     2, # batch_size
#     12, #seq_len
#     256,
#     dtype=torch.float32
# )
# regulator(inp_tensor)
# duration_predictor_output = torch.tensor([[2,1,2]])
# alignment = create_alignment(duration_predictor_output.sum(1).max().item(), duration_predictor_output)
# print(alignment)