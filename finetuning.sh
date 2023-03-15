PRETRAIN_CHKPT='./pre/mae_pretrain_vit_base.pth'
DATA_DIR='/vsc2022_data_frame/'

python submitit_finetune.py \
    --nodes 1 \
    --batch_size 32 \
    --model vit_base_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path ${DATA_DIR}