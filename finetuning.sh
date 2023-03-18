NNODES=1 #Number of total node
NODE_RANK=0 #Rank of total node
NPROC_PER_NODE=2 #Process number of this node
MASTER_ADDR='localhost'
MASTER_PORT='45454'

python -m torch.distributed.launch --nnodes=${NNODES} --node_rank=${NODE_RANK} \
    --nproc_per_node=${NPROC_PER_NODE} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    ./main_finetune.py --batch_size 64 --blr 1e-3

