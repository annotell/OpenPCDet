### to train multi-gpu
python -m torch.distributed.launch --nproc_per_node=2 --rdzv_endpoint=localhost:64000 train.py --launcher pytorch --cfg_file cfgs/cosmos_models/pointpillar.yaml --workers 12

###Install OpenPCDet

1. git clone https://github.com/open-mmlab/OpenPCDet.git
2. pip install numpy==1.20
3. pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
4. pip install -r requirements.txt (MAKE sure numpy version is 1.20 otherwise numba will complain)
5. pip install spconv-cu114
6. python setup.py develop