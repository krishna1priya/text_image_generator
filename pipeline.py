import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(
    prompt,
    uncond_prompt=None,     #negative prompt
    strength=0.8,           #(0-1)
    cfg=True,               #classifier free guidance
    cfg_scale=8,          #attention given to the prompt by the model(1-14)
    sampler_name="ddpm",
    num_inference_steps=50,
    models={},
    seed=None,
    device=None,
    # idle_device=None,
    tokenizer=None,
):
    with torch.no_grad():
        if device:
            to_device = lambda x: x.to(device)
        else:
            to_device = lambda x: x

        generator = torch.Generator(device=device)      #to generate the nois e
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)
        
        if cfg:
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)  # (batch_size, seq_len)
            cond_prompt = clip(cond_tokens)    # (batch_size, seq_len, vec_size)
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            uncond_prompt = clip(uncond_tokens)
            prompt = torch.cat([cond_prompt, uncond_prompt])         # (2*batch_size, seq_len)
        else:
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            prompt = clip(tokens)
        to_device(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(num_inference_steps)
        else:
            raise ValueError("Unknown sampler")
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)
            model_input = latents       #(batch_size, 4, latent height, latent width)

            if cfg:
                model_input = model_input.repeat(2, 1, 1, 1)         #(2*batch_size, 4, latent height, latent width)
            model_output = diffusion(model_input, prompt, time_embedding)       #predicted noise by unet

            if cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            latents = sampler.step(timestep, latents, model_output)

        to_device(diffusion)

        decoder = models["decoder"]
        decoder.to(device)    

        images = decoder(latents)        #(2*batch_size, 4, latent height, latent width) - (batch_size, 3, height, width)
        to_device(decoder)
        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]
    
def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None] 
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
