# Turn on virtual environment
cd /home/hao/repo/Domain-Game
mamba activate DG

# Debug
CUDA_VISIBLE_DEVICES=0   python main.py   --task  brain      --file   debug   -m "debug"   --debug  --deviceid 0

# Train brain tumor segmentation
CUDA_VISIBLE_DEVICES=1   python main.py   --task  brain      --file   main   -m "standard_brain_train"  --deviceid 0

# Train abdominal segmentation
CUDA_VISIBLE_DEVICES=1   python main.py   --task  abdominal  --file   main  -m "standard_abdominal_train"  #  --resume   $pth 







