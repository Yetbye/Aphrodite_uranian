import torch
from diffusers import AutoencoderKLLTXVideo


class LtxVAE:
    def __init__(
        self,
        pretrained_model_type_or_path,
        dtype = torch.bfloat16,
        device = "cuda",
    ):
        self.model = AutoencoderKLLTXVideo.from_pretrained(pretrained_model_type_or_path)
        self.model = self.model.eval().requires_grad_(False).to(device).to(dtype)

    # torch.Size([1, 3, 33, 512, 512]) -> torch.Size([128, 5, 16, 16])
    def encode(self, video):
        latents = self.model.encode(video, return_dict=False)[0].sample()
        return latents[0]

    # torch.Size([128, 5, 16, 16]) -> torch.Size([1, 3, 33, 512, 512])
    def decode(self, zs):
        latents = zs.unsqueeze(0)
        image = self.model.decode(latents, return_dict=False)[0]
        return image
