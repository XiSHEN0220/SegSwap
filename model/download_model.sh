wget https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar

python convert_torchvision.py moco_v2_800ep_pretrain.pth.tar moco_v2_800ep_pretrain_torchvision.pth.tar

rm moco_v2_800ep_pretrain.pth.tar

