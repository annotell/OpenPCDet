import torch
import torch.distributed as dist

def main():
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    print(f"Process {rank} initialized.")
    dist.barrier()
    print(f"Process {rank} completed barrier.")

if __name__ == "__main__":
    main()