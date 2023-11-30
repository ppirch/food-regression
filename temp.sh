python3 training.py \
    --model_type classify \
    --backbone resnet50 \
    --batch_size 1 \
    --epochs 2

python3 training.py \
    --model_type regress \
    --backbone resnet50 \
    --batch_size 1 \
    --epochs 2

python3 training.py \
    --model_type shared \
    --backbone resnet50 \
    --batch_size 1 \
    --epochs 2

python3 training.py \
    --model_type shared \
    --backbone resnet50 \
    --batch_size 1 \
    --epochs 2 \
    --loss_type uncertainty

python3 training.py \
    --model_type shared \
    --backbone resnet50 \
    --batch_size 1 \
    --epochs 2 \
    --loss_type automatic

python3 training.py \
    --model_type concat \
    --backbone resnet50 \
    --batch_size 1 \
    --epochs 2

python3 training.py \
    --model_type concat \
    --backbone resnet50 \
    --batch_size 1 \
    --epochs 2 \
    --loss_type uncertainty

python3 training.py \
    --model_type concat \
    --backbone resnet50 \
    --batch_size 1 \
    --epochs 2 \
    --loss_type automatic

