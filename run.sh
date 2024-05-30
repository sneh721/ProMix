# CUSTOM
python Train_promix.py --validation --model_name convnext_nano --noise_type worst --cosine --dataset custom --num_class 100 --rho_range 0.5,0.5 --tau 0.99 --pretrain_ep 1 --noise_mode custom
# Try pretrain epochs 0
