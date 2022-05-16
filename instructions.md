### to train multi-gpu
python -m torch.distributed.launch --nproc_per_node=2 --rdzv_endpoint=localhost:64000 train.py --launcher pytorch --cfg_file cfgs/cosmos_models/pointpillar.yaml --workers 12

###Install OpenPCDet

1. git clone https://github.com/open-mmlab/OpenPCDet.git
2. pip install numpy==1.20
3. pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
4. pip install -r requirements.txt (MAKE sure numpy version is 1.20 otherwise numba will complain)
5. pip install spconv-cu114
6. python setup.py develop

###Using docker - recommended 
Assuming CUDA is version 11.3 and installs pytorch 1.10.
Before building the docker image make sure that /etc/docker/daemon.json allows
docker to access cuda at build time. The file should be as follows. If not,
update it and run: sudo systemctl restart docker

`{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}`

Steps:

1. Clone repo: git clone git@github.com:annotell/OpenPCDet.git
2. Build image with:  docker build -t openpcdet-docker -f OpenPCDet/docker/Dockerfile .
3. If building image for GCR repo, run instead: docker build -t eu.gcr.io/annotell-com/openpcdet:TAGNAME -f OpenPCDet/docker/Dockerfile .
4. To push to GCR repo, run: docker push eu.gcr.io/annotell-com/openpcdet:TAGNAME
5. docker run -p 8888:8888 --hostname localhost -it -d --gpus all  -v /path/to/dataset/:/data -v /path/to/output/folder/:/root/OpenPCDet/output/ openpcdet-docker:latest 
6. docker exec -it container_id bash


/path/to/output/folder/ is where OpenPCDet will save logs and models outside the docker image
