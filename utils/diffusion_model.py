import torch
from omegaconf import OmegaConf
import importlib
from pytorch_lightning import seed_everything
from ldm.util import instantiate_from_config



def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpectModuleNotFoundError: No module named 'omegaconf'ed keys:")
        print(u)

    if device == torch.device("cuda"):
        model.cuda()
    elif device == torch.device("cpu"):
        model.cpu()
        model.cond_stage_model.device = "cpu"
    else:
        raise ValueError(f"Incorrect device name. Received: {device}")
    model.eval()
    return model


def load_pretrained_model(ckpt, config):
    seed_everything(42)
    config = OmegaConf.load(config)
    device = torch.device("cuda")
    model = load_model_from_config(config, ckpt, device)
    return model


# --------------------- Functionality --------------------- 
 
@torch.no_grad()
def to_img(model, latent, to_numpy=True):
    img = model.decode_first_stage(latent)
    img = torch.clamp((img + 1.0) / 2.0, min=0.0, max=1.0)
    img = img.permute(0, 2, 3, 1)
    if to_numpy:
        img = img.cpu().numpy()
    
    return img






