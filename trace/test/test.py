import os
import argparse
import torch
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def get_local_rank():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    local_rank = args.local_rank
    if local_rank != -1:
        return local_rank
    if "LOCAL_RANK" in os.environ:
        return int(os.getenv("LOCAL_RANK"))
    return -1

def main():
    local_rank = get_local_rank()
    if local_rank == -1:
        os._exit(-1)

    dist.init_process_group(
        backend="nccl",
        init_method="env://"
    )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    model = torch.nn.Linear(10, 2).to(device)
    ddp_model = DDP(model, device_ids=[local_rank])

    torch.manual_seed(dist.get_rank())
    input_tensor = torch.randn(2, 10).to(device)

    output = ddp_model(input_tensor)
    if dist.get_rank() == 0:
        print(f"rank 0: {output}")

    local_tensor = torch.tensor([dist.get_rank() + 1.0], device=device)
    print(f"rank {dist.get_rank()} local_tensor before allreduce: {local_tensor.item()}")

    dist.all_reduce(local_tensor, op=dist.ReduceOp.SUM)
    print(f"rank {dist.get_rank()} local_tensor after allreduce: {local_tensor.item()}")

    dist.all_reduce(local_tensor, op=dist.ReduceOp.SUM)
    print(f"rank {dist.get_rank()} local_tensor after allreduce: {local_tensor.item()}")

    #allgather
    allgather_tensor = torch.tensor([dist.get_rank() + 1.0], device=device)
    print(f"rank {dist.get_rank()} allgather_tensor before allgather: {allgather_tensor.item()}")

    allgather_tensor_list = [torch.zeros_like(local_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(allgather_tensor_list, local_tensor)
    allgather_tensor = allgather_tensor_list[dist.get_rank()]
    print(f"rank {dist.get_rank()} allgather_tensor after allgather: {allgather_tensor.item()}")

    allgather_tensor_list = [torch.zeros_like(local_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(allgather_tensor_list, local_tensor)
    allgather_tensor = allgather_tensor_list[dist.get_rank()]
    print(f"rank {dist.get_rank()} allgather_tensor after allgather: {allgather_tensor.item()}")

    #reducescatter
    reducescatter_tensor = torch.tensor([dist.get_rank() + 1.0], device=device)
    print(f"rank {dist.get_rank()} reducescatter_tensor before reducescatter: {reducescatter_tensor.item()}")

    scatter_input_list = [torch.tensor([dist.get_rank() + 1.0], device=device) for _ in range(dist.get_world_size())]
    dist.reduce_scatter(reducescatter_tensor, scatter_input_list, op=dist.ReduceOp.SUM)
    print(f"rank {dist.get_rank()} reducescatter_tensor after reducescatter: {reducescatter_tensor.item()}")

    scatter_input_list = [torch.tensor([dist.get_rank() + 1.0], device=device) for _ in range(dist.get_world_size())]
    dist.reduce_scatter(reducescatter_tensor, scatter_input_list, op=dist.ReduceOp.SUM)
    print(f"rank {dist.get_rank()} reducescatter_tensor after reducescatter: {reducescatter_tensor.item()}")

    #broadcast
    broadcast_tensor = torch.tensor([dist.get_rank() + 1.0], device=device)
    print(f"rank {dist.get_rank()} broadcast_tensor before broadcast: {broadcast_tensor.item()}")

    dist.broadcast(broadcast_tensor, src=0)
    print(f"rank {dist.get_rank()} broadcast_tensor after broadcast: {broadcast_tensor.item()}")

    dist.broadcast(broadcast_tensor, src=0)
    print(f"rank {dist.get_rank()} broadcast_tensor after broadcast: {broadcast_tensor.item()}")

    #sendrecv
    sendrecv_tensor = torch.tensor([dist.get_rank() + 1.0], device=device)
    print(f"rank {dist.get_rank()} sendrecv_tensor before sendrecv: {sendrecv_tensor.item()}")

    dist.broadcast(sendrecv_tensor, src=0)
    print(f"rank {dist.get_rank()} sendrecv_tensor after sendrecv: {sendrecv_tensor.item()}")

    dist.broadcast(sendrecv_tensor, src=0)
    print(f"rank {dist.get_rank()} sendrecv_tensor after sendrecv: {sendrecv_tensor.item()}")

    #send
    send_tensor = torch.tensor([dist.get_rank() + 200.0], device=device)
    if dist.get_rank() == 0:
        print(f"rank 0 send_tensor before send: {send_tensor.item()}")
        dist.send(tensor=send_tensor, dst=1)
        print(f"rank 0 sent to rank 1")

    #recv
    recv_tensor = torch.tensor([0.0], device=device)
    if dist.get_rank() == 1:
        print(f"rank 1 recv_tensor before recv: {recv_tensor.item()}")
        dist.recv(tensor=recv_tensor, src=0)
        print(f"rank 1 recv_tensor after recv: {recv_tensor.item()}")

    time.sleep(2)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()