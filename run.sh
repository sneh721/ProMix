# CUSTOM
# python Train_promix.py --validation --model_name convnext_nano --noise_type worst --cosine --dataset custom --num_class 100 --rho_range 0.5,0.5 --tau 0.99 --pretrain_ep 0 --warmup_ep 1 --num_epochs 1 --noise_mode custom
# Try pretrain epochs 0


python Train_promix.py --lr 0.05 --validation --model_name convnext_nano --cosine --dataset custom --num_class 100 --rho_range 0.7,0.7 --tau 0.95 --pretrain_ep 0 --warmup_ep 1 --num_epochs 1 --debias_output 0.5 --debias_pl 0.5  --noise_mode sym --noise_rate 0.2
