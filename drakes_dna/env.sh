mamba install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
uv pip install packaging
uv pip install ninja
uv pip install transformers
uv pip install datasets
uv pip install omegaconf
conda install ipykernel
python -m ipykernel install --user --name sedd --display-name "Python (sedd)"
uv pip install hydra-core --upgrade
uv pip install hydra-submitit-launcher

# for mdlm
uv pip install causal-conv1d
uv pip install lightning
uv pip install timm
uv pip install rich

uv pip install scipy
uv pip install wandb
