## Masked Autoencoders: A PyTorch Implementation

<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/146857310-f258c86c-fde6-48e8-9cee-badd2b21bd2c.png" width="480">
</p>

```
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 --nnodes=1 --master_addr="localhost"  main_finetune.py
```

This is a PyTorch/GPU re-implementation of the paper [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377):
```
@Article{MaskedAutoencoders2021,
  author  = {Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and Piotr Doll{\'a}r and Ross Girshick},
  journal = {arXiv:2111.06377},
  title   = {Masked Autoencoders Are Scalable Vision Learners},
  year    = {2021},
}
```


