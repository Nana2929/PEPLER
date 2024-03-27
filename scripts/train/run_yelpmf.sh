# ================== Yelp  ==================
LOGDIR=logs
CKPTDIR=outputs
DATADIR="../nete_format_data/yelp"
export CUDA_VISIBLE_DEVICES=0
# mkdir if not exist
for fold in 1 2 3 4 5; do

    fold_logdir=$LOGDIR/${fold}/yelp
    fold_ckptdir=$CKPTDIR/${fold}/yelp
    mkdir -p $fold_logdir
    mkdir -p $fold_ckptdir

    echo "Start training fold $fold yelp (reg; mf)"
    # python3 -u reg_predict.py \
    # --data_path $DATADIR/reviews.pickle \
    # --index_dir $DATADIR/${fold}/ \
    # --use_mf \
    # --checkpoint $fold_ckptdir/yelpmf/ >> $fold_logdir/yelpmf.log
    echo "Start predicting fold $fold yelp (reg; mf)"
    python3 -u predict.py --data_path "../nete_format_data/yelp/reviews.pickle" \
        --index_dir "../nete_format_data/yelp/${fold}/" \
        --model_path "outputs/${fold}/yelp/yelpmf/model.pt" \
        --words 20 \
        --output_path "outputs/${fold}/yelp/yelpmf/generated.jsonl" > "logs/${fold}/yelp/select_yelpmf.log"
done