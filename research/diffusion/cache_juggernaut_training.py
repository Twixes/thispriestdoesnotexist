"""Cache exact Juggernaut captions/posteriors; train split only, no UNet loaded."""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil
from compare import ROOT, sha, write
from juggernaut_pilot import PROTOCOL, MODEL, DATASET, preflight


def worker(output):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    import torch
    from PIL import Image, ImageOps
    from torchvision.transforms import functional as TF, InterpolationMode
    from diffusers import StableDiffusionXLPipeline, AutoencoderKL
    from safetensors.torch import save_file
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.mps.set_per_process_memory_fraction(.55)
    initial_swap = psutil.swap_memory().used
    def guard(phase):
        torch.mps.synchronize()
        vm = psutil.virtual_memory()
        r = {"phase":phase,"rss_bytes":psutil.Process().memory_info().rss,
            "driver_bytes":torch.mps.driver_allocated_memory(),"available_fraction":vm.available/vm.total,
            "swap_growth_bytes":psutil.swap_memory().used-initial_swap}
        with (output/"resources.jsonl").open("a") as stream:
            stream.write(json.dumps(r)+"\n")
        if r["driver_bytes"]>22*1024**3 or r["rss_bytes"]>24*1024**3 or r["available_fraction"]<.20 or r["swap_growth_bytes"]>512*1024**2:
            raise MemoryError(r)
    protocol, components = preflight()
    dataset = DATASET
    reviewed = json.loads((dataset/"manifest.json").read_text())
    entries = [e for e in reviewed["entries"] if e["split"]=="train"]
    if len(entries)!=20:
        raise ValueError("Expected reviewed20-image training split")
    model = MODEL
    encoder = StableDiffusionXLPipeline.from_pretrained(model,unet=None,vae=None,variant="fp16",
        torch_dtype=torch.float16,local_files_only=True).to("mps")
    guard("text_encoders_loaded")
    captions = []
    with torch.inference_mode():
        for entry in entries:
            prompt,_,pooled,_ = encoder.encode_prompt(entry["caption"],device="mps",
                do_classifier_free_guidance=False,num_images_per_prompt=1)
            if not torch.isfinite(prompt).all().item() or not torch.isfinite(pooled).all().item():
                raise FloatingPointError("Nonfinite CLIP embeddings")
            captions.append({"prompt_embeds":prompt.cpu().contiguous(),
                "pooled_prompt_embeds":pooled.cpu().contiguous()})
            guard("caption_encoded")
    del encoder,prompt,pooled
    gc.collect()
    torch.mps.empty_cache()
    vae = AutoencoderKL.from_pretrained(model,subfolder="vae",variant="fp16",
        torch_dtype=torch.float32,local_files_only=True).to("mps")
    guard("vae_loaded_encoders_released")
    manifest_entries=[]
    with torch.inference_mode():
        for index,entry in enumerate(entries):
            filename=Path(entry["destination"]).name
            path=dataset/"train"/filename
            if sha(path)!=entry["destination_sha256"]:
                raise ValueError("Reviewed image changed: "+filename)
            with Image.open(path) as image:
                image=ImageOps.exif_transpose(image).convert("RGB")
                original_size=[image.height,image.width]
                resized=TF.resize(image,512,interpolation=InterpolationMode.LANCZOS)
                crop_top_left=[int(round((resized.height-512)/2.0)),int(round((resized.width-512)/2.0))]
                cropped=TF.center_crop(resized,[512,512])
                pixels=TF.normalize(TF.to_tensor(cropped),[.5],[.5]).unsqueeze(0)
            posterior=vae.encode(pixels.to("mps")).latent_dist
            tensors={**captions[index],"latent_mean":posterior.mean.cpu().contiguous(),
                "latent_std":posterior.std.cpu().contiguous()}
            if not all(torch.isfinite(value).all().item() for value in tensors.values()):
                raise FloatingPointError("Nonfinite posterior: "+filename)
            target=output/(Path(filename).stem+".safetensors")
            save_file(tensors,target)
            manifest_entries.append({"file_name":filename,"caption":entry["caption"],"path":target.name,
                "sha256":sha(target),"image_sha256":sha(path),"original_size":original_size,
                "crop_top_left":crop_top_left,"tensor_shapes":{k:list(v.shape) for k,v in tensors.items()},
                "tensor_dtypes":{k:str(v.dtype) for k,v in tensors.items()}})
            guard("posterior_cached")
            print(f"Cached {index+1}/20",flush=True)
    write(output/"manifest.json",{"complete":True,"entries":manifest_entries,"resolution":512,
        "dataset_manifest_sha256":sha(dataset/"manifest.json"),
        "model_provenance_sha256":sha(model/"provenance.json"),"model":"RunDiffusion/Juggernaut-X-Hyper",
        "protocol_sha256":sha(PROTOCOL),"upstream_checkpoint_sha256":components["upstream_checkpoint_sha256"],
        "vae_scaling_factor":vae.config.scaling_factor,"vae_compute_dtype":"float32",
        "text_encoder_compute_dtype":"float16","source_sha256":sha(Path(__file__)),
        "transform":"Lanczos Resize512,CenterCrop512,normalize[-1,1],no flip",
        "latent_sampling":"Posterior mean/std cached; training must resample fresh Gaussian each step before scaling."})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--worker",action="store_true")
    args=parser.parse_args();output=args.output.resolve()
    if args.worker:
        return worker(output)
    preflight()
    if output.exists():
        raise ValueError("Never overwrite cache experiments")
    if psutil.virtual_memory().available/psutil.virtual_memory().total<.35:
        raise MemoryError("At least35% available memory required")
    output.mkdir(parents=True)
    (output/"source.py").write_bytes(Path(__file__).read_bytes())
    (output/"juggernaut_pilot.py").write_bytes((ROOT/"juggernaut_pilot.py").read_bytes())
    (output/"protocol.json").write_bytes(PROTOCOL.read_bytes())
    start=time.monotonic();peak=0;failure=None
    initial_swap=psutil.swap_memory().used
    with (output/"worker.log").open("w") as log:
        process=subprocess.Popen([sys.executable,str(Path(__file__)),"--output",str(output),"--worker"],
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while process.poll() is None:
                vm=psutil.virtual_memory()
                try:peak=max(peak,psutil.Process(process.pid).memory_info().rss)
                except psutil.NoSuchProcess:break
                if peak>24*1024**3:failure="RSS exceeds24GiB"
                elif vm.available/vm.total<.20:failure="Available memory below20%"
                elif psutil.swap_memory().used-initial_swap>512*1024**2:failure="Swap growth exceeds512MiB"
                elif time.monotonic()-start>1200:failure="20minute deadline"
                if failure:
                    os.killpg(process.pid,signal.SIGTERM);break
                time.sleep(.5)
        finally:
            if process.poll() is None:
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL)
            code=process.wait()
            write(output/"supervisor.json",{"exit_code":code,"failure":failure,
                "seconds":time.monotonic()-start,"peak_sampled_rss_bytes":peak})
    if failure or code:raise SystemExit(f"Cache failed:{failure or code}")


if __name__=="__main__":main()
