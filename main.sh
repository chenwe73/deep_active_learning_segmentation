#!/bin/bash

# do NOT edit file while running!

CITYSCAPES="/home/wt6chen/workspace/cityscapes_crop/"
CITYSCAPES_IMG="${CITYSCAPES}leftImg8bit/train/"
CITYSCAPES_LABEL="${CITYSCAPES}gtFine/train/"
CITYSCAPES_VAL_IMG="${CITYSCAPES}leftImg8bit/val/"
CITYSCAPES_VAL_LABEL="${CITYSCAPES}gtFine/val/"

IMG_PATTERN="*_leftImg8bit.png"
LABEL_PATTERN="*_gtFine_labelTrainIds.png"
MAX_TRAIN_SIZE=2975
MAX_VAL_SIZE=500
N_CLASS=19

OUT="./out/"
POOL="${OUT}pool.csv"
VAL="${OUT}val.csv"
LABELD="${OUT}labeled.csv"
UNLABELD="${OUT}unlabeled.csv"
TEMP="./tmp/"

WEIGHTS="${OUT}weights/"
HISTORY="${OUT}history/"
MODEL="fcn8"
INPUT_WIDTH=448 #224, 2048
INPUT_HEIGHT=448 #224, 1024
BATCH_SIZE=1
SEED=123

DEVICES=0
TRAIN_SIZE=10 #$MAX_TRAIN_SIZE
VAL_SIZE=4 #$MAX_VAL_SIZE
QUERY_BATCH=2
ITERATION=1
EPOCH=1 # 40 for fcn8, 60 for segnet, 100 for segnet with dropout, 60 for fcn8 small LR
INIT_BATCH=2


get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}


create_list()
{
    LIST_IMG="./img.csv"
    LIST_LABEL="./label.csv"
    find $1 -type f -name $IMG_PATTERN > $LIST_IMG
    find $2 -type f -name $LABEL_PATTERN > $LIST_LABEL
    sort -o $LIST_IMG $LIST_IMG
    sort -o $LIST_LABEL $LIST_LABEL
    paste -d ',' $LIST_IMG $LIST_LABEL > $3
    rm $LIST_IMG $LIST_LABEL
}


init()
{
    mkdir -p $OUT
    mkdir -p $WEIGHTS
    mkdir -p $HISTORY
    mkdir -p $TEMP
    
    create_list $CITYSCAPES_IMG $CITYSCAPES_LABEL $POOL
    head -n $TRAIN_SIZE $POOL > temp && mv temp $POOL
    cp $POOL $UNLABELD
    shuf -o $UNLABELD $POOL --random-source=<(get_seeded_random $SEED)
    
    #cp "./cereals/cereals_random.csv" $POOL
    #cp $POOL $UNLABELD
    > $LABELD
    
    create_list $CITYSCAPES_VAL_IMG $CITYSCAPES_VAL_LABEL $VAL
    head -n $VAL_SIZE $VAL > temp && mv temp $VAL
}


train()
{
    WEIGHTS_ITER="${WEIGHTS}weights_${1}"
    HISTORY_ITER="${HISTORY}history_${1}"
    
    CUDA_VISIBLE_DEVICES=$DEVICES \
    python3 train.py \
        --train_list=$LABELD \
        --val_list=$VAL \
        --save_weights_path=$WEIGHTS_ITER \
        --save_history_path=$HISTORY_ITER \
        --model_name=$MODEL \
        --n_classes=$N_CLASS \
        --batch_size=$BATCH_SIZE \
        --epochs=$EPOCH \
        --input_height=$INPUT_HEIGHT \
        --input_width=$INPUT_WIDTH
}


query()
{
    WEIGHTS_ITER="${WEIGHTS}weights_$((${1}-1))"
    
    CUDA_VISIBLE_DEVICES=$DEVICES \
    python3 query.py \
        --save_weights_path=$WEIGHTS_ITER \
        --list_file=$UNLABELD \
        --model_name=$MODEL \
        --n_classes=$N_CLASS \
        --input_height=$INPUT_HEIGHT \
        --input_width=$INPUT_WIDTH \
        --seed=$SEED
}


main()
{
    set -e
    rm -rf ${OUT}
    init
    
    echo "labeling init .........."
    head -n $INIT_BATCH $UNLABELD >> $LABELD
    tail -n +$(($INIT_BATCH+1)) $UNLABELD > temp && mv temp $UNLABELD
    
    echo "training init .........."
    train 0
    
    for (( i=1; i<$(($ITERATION+1)); i++ ))
    do
        echo "querying $i .........."
        query $i
        
        echo "labeling $i .........."
        head -n $QUERY_BATCH $UNLABELD >> $LABELD
        tail -n +$(($QUERY_BATCH+1)) $UNLABELD > temp && mv temp $UNLABELD
        
        echo "training $i .........."
        train $i
    done
}


test()
{
    CUDA_VISIBLE_DEVICES=$DEVICES \
    python3 query.py \
        --save_weights_path="./tmp/weights_4" \
        --list_file="./tmp/val.csv" \
        --model_name=$MODEL \
        --n_classes=$N_CLASS \
        --input_height=$INPUT_HEIGHT \
        --input_width=$INPUT_WIDTH \
        --seed=$SEED
}


#main
test


