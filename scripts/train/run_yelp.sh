# ================== Yelp  ==================
LOGDIR=logs
CKPTDIR=outputs
DATADIR="../nete_format_data/yelp"
export CUDA_VISIBLE_DEVICES=1
# mkdir if not exist
for fold in 1 2 3 4 5; do

    fold_logdir=$LOGDIR/${fold}/yelp
    fold_ckptdir=$CKPTDIR/${fold}/yelp
    mkdir -p $fold_logdir
    mkdir -p $fold_ckptdir
    python3 -u main_predict.py \
    --data_path $DATADIR/reviews.pickle \
    --index_dir $DATADIR/${fold}/ \
    --cuda \
    --checkpoint $fold_ckptdir/yelp/ >> $fold_logdir/yelp.log

    python3 -u predict.py --data_path "../nete_format_data/yelp/reviews.pickle" \
        --index_dir "../nete_format_data/yelp/${fold}/" \
        --model_path "outputs/${fold}/yelp/yelp/model.pt" \
        --words 20 \
        --output_path "outputs/${fold}/yelp/yelp/generated.jsonl" > "logs/${fold}/yelp/yelp.log"
done