# train
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 /data/la/program1/transformer/main.py \
    --mode 'train' \
    --data_path '/data/la/program1/transformer/dataset' \
    --d_model 512 \
    --n_head 8 \
    --n_encoder 2 \
    --n_decoder 1 \
    --dropout 0.1 \
    --epochs 150 \
    --output_size 336 \
    --input_size 96 \
    --lr 5e-3 \
    --batch_size 64 \
    --print_every 100 \
    --log_file '/data/la/program1/transformer/logs/conv336_train.log' \
    --save_path '/data/la/program1/transformer/ckpts/CONV+REVIN_336' \
    --special_tokens \