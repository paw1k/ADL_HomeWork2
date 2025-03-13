import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
#         raise NotImplementedError()
        self.embedding = torch.nn.Embedding(n_tokens, d_latent)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_latent,
            nhead=8,
            dim_feedforward=4*d_latent,
            batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.output_layer = torch.nn.Linear(d_latent, n_tokens)
        self.n_tokens = n_tokens

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
#         raise NotImplementedError()
        x = x.squeeze()
        B, h, w = x.shape
        seq_len = h * w
        x_flat = x.view(B, seq_len)
        embeddings = self.embedding(x_flat)

        shifted_embeds = torch.zeros_like(embeddings)
        shifted_embeds[:, 1:] = embeddings[:, :-1].clone()

        mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        transformer_output = self.transformer_encoder(shifted_embeds, mask)

        logits = self.output_layer(transformer_output)
        logits = logits.view(B, h, w, self.n_tokens)
        return logits, {}

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
#         raise NotImplementedError()
        device = device or torch.device('cpu')
        output = torch.zeros((B, h * w), dtype=torch.long, device=device)
        for i in range(h * w):
            input_seq = torch.roll(output, shifts=1, dims=1)
            input_seq[:, 0] = 0
            input_reshaped = input_seq.view(B, h, w)
            with torch.no_grad():
                logits, _ = self.forward(input_reshaped)
            logits_flat = logits.view(B, h * w, -1)
            next_logits = logits_flat[:, i, :]
            next_token = torch.argmax(next_logits, dim=-1)
            output[:, i] = next_token
        return output.view(B, h, w)
