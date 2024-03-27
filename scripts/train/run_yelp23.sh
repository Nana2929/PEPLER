# ================== Yelp23  ==================
LOGDIR=logs
CKPTDIR=outputs
DATADIR="../nete_format_data/yelp23"
DEVICE="cuda:1"
# mkdir if not exist
for fold in 1 2 3 4 5;
do
    mkdir -p $LOGDIR/${fold}/yelp23
    mkdir -p $CKPTDIR/${fold}/yelp23
    fold_logdir=$LOGDIR/${fold}/yelp23
    fold_ckptdir=$CKPTDIR/${fold}/yelp23
    echo "fold ${fold}"
    echo "logdir: ${fold_logdir}"
    echo "ckptdir: ${fold_ckptdir}"
    python3 -u main_predict.py \
    --data_path $DATADIR/reviews.pickle \
    --index_dir $DATADIR/${fold}/ \
    --device $DEVICE \
    --checkpoint ${fold_ckptdir}/yelp23/ >> ${fold_logdir}/yelp23.log

    python3 -u discrete_predict.py \
    --data_path $DATADIR/reviews.pickle \
    --index_dir $DATADIR/${fold}/ \
    --device $DEVICE \
    --checkpoint ${fold_ckptdir}/yelp23d/ >> ${fold_logdir}/yelp23d.log

    python3 -u reg_predict.py \
    --data_path $DATADIR/reviews.pickle \
    --index_dir $DATADIR/${fold}/ \
    --device $DEVICE \
    --use_mf \
    --checkpoint ${fold_ckptdir}/yelp23mf/ >> ${fold_logdir}/yelp23mf.log

    python3 -u reg_predict.py \
    --data_path $DATADIR/reviews.pickle \
    --index_dir $DATADIR/${fold}/ \
    --device $DEVICE \
    --rating_reg 1 \
    --checkpoint ${fold_ckptdir}/yelp23mlp/ >> ${fold_logdir}/yelp23mlp.log
done