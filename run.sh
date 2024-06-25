# CUSTOM
python Train_promix.py --validation --data_path cifar100png --noise_path Images/ --model_name convnext_nano --noise_type worst --cosine --dataset custom --num_class 100 --rho_range 0.5,0.5 --tau 0.99 --num_epochs 2 --noise_mode custom
# Try pretrain epochs 0


# Cifar 100 dataset - same settings as the example by Promix developers
# python Train_promix.py --validation --lr 0.05 --model_name convnext_nano --cosine --dataset custom --num_class 100 --rho_range 0.7,0.7 --tau 0.95 --pretrain_ep 1 --warmup_ep 0 --num_epochs 1 --debias_output 0.5 --debias_pl 0.5  --noise_mode sym --noise_rate 0.2
# NOTE: naming is very confusing
# pretrain_ep = warmup epochs
# warmup_ep = parameter rampup
# num epochs = train_ep + pre train ep
# train epochs = num epochs - pre train ep