# test
export CUDA_VISIBLE_DEVICES=0
python /data/la/program1/transformer/main.py \
    --mode 'test' \
    --gpu 0 \
    --data_path '/data/la/program1/transformer/dataset' \
    --output_size 96 \
    --input_size 96 \
    --d_model 512 \
    --n_head 8 \
    --n_encoder 2 \
    --n_decoder 1 \
    --dropout 0.1 \
    --batch_size 32 \
    --print_every 20 \
    --log_file '/data/la/program1/transformer/logs/CONVtest96.log' \
    --save_path '/data/la/program1/transformer/ckpts/CONV+REVIN_96/10000.pth' \
    --figure_path '/data/la/program1/transformer/figures/new'