import numpy as np
import pickle as pickle
import os
import sys
import wget
import torch
import gradio
from huggingface_hub import hf_hub_download

os.system("git clone https://github.com/NVlabs/stylegan3")
sys.path.append('./stylegan3')

model_names = {
        'AFHQv2-512-R': 'stylegan3-r-afhqv2-512x512.pkl',
        'FFHQ-1024-R': 'stylegan3-r-ffhq-1024x1024.pkl',
        'FFHQ-U-256-R': 'stylegan3-r-ffhqu-256x256.pkl',
        'FFHQ-U-1024-R': 'stylegan3-r-ffhqu-1024x1024.pkl',
        'MetFaces-1024-R': 'stylegan3-r-metfaces-1024x1024.pkl',
        'MetFaces-U-1024-R': 'stylegan3-r-metfacesu-1024x1024.pkl',
        'AFHQv2-512-T': 'stylegan3-t-afhqv2-512x512.pkl',
        'FFHQ-1024-T': 'stylegan3-t-ffhq-1024x1024.pkl',
        'FFHQ-U-256-T': 'stylegan3-t-ffhqu-256x256.pkl',
        'FFHQ-U-1024-T': 'stylegan3-t-ffhqu-1024x1024.pkl',
        'MetFaces-1024-T': 'stylegan3-t-metfaces-1024x1024.pkl',
        'MetFaces-U-1024-T': 'stylegan3-t-metfacesu-1024x1024.pkl',
                }
model_dict = {
        name: file_name
        for name, file_name in model_names.items()
    }

def fetch_model(url_or_path):
    basename = os.path.basename(url_or_path)
    if os.path.exists(basename):
        return basename
    else:
        wget.download(url_or_path)
        print(basename)
        return basename
           
def load_model(file_name: str, device: torch.device):
    #path = torch.hub.download_url_to_file('https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/'+f'{file_name}',
    #                       f'{file_name}')
    base_url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/"
    network_url = base_url + f'{file_name}'
    #local_path = '/content/'f'{file_name}'
    with open(fetch_model(network_url), 'rb') as f:
        model = pickle.load(f)['G_ema']
    model.eval()
    model.to(device)
    with torch.inference_mode():
        z = torch.zeros((1, model.z_dim)).to(device)
        label = torch.zeros([1, model.c_dim], device=device)
        model(z, label)
    return model
    
def generate_image(model_name: str, seed: int, truncation_psi: float):
   device = 'cpu'
   model = model_dict[model_name]
   model = load_model(model, device)
   seed = int(np.clip(seed, 0, np.iinfo(np.uint32).max))
   z = torch.from_numpy(np.random.RandomState(seed).randn(1, model.z_dim)).to(device)
   label = torch.zeros([1, model.c_dim], device=device)

   out = model(z, label, truncation_psi=truncation_psi)
   out = (out.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
   return out[0].cpu().numpy()   
   
import gradio as gr
gr.Interface(
        generate_image,
        [
            gr.inputs.Radio(list(model_names.keys()),
                            type='value',
                            default='FFHQ-1024-R',
                            label='Model'),
            gr.inputs.Number(default=0, label='Seed'),
            gr.inputs.Slider(
                0, 2, step=0.05, default=0.7, label='Truncation psi')
        ],
        gr.outputs.Image(type='numpy', label='Output')
    ).launch(debug=True)
    
#os.system("git rm -r --cached stylegan3") 