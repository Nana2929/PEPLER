# ================== gest  ==================
LOGDIR=logs
CKPTDIR=outputs
DATADIR="../nete_format_data/gest"
DEVICE="cuda:0"
# mkdir if not exist
for fold in 2 3 4 5 #1
do
    mkdir -p $LOGDIR/${fold}/gest
    mkdir -p $CKPTDIR/${fold}/gest
    fold_logdir=$LOGDIR/${fold}/gest
    fold_ckptdir=$CKPTDIR/${fold}/gest


    echo "fold ${fold}"

    echo "logdir: ${fold_logdir}"
    echo "ckptdir: ${fold_ckptdir}"

    python3 -u main_predict.py \
    --data_path $DATADIR/reviews.pickle \
    --index_dir $DATADIR/${fold}/ \
    --device $DEVICE \
    --checkpoint $fold_ckptdir/gest/ >> $fold_logdir/gest.log

    python3 -u discrete_predict.py \
    --data_path $DATADIR/reviews.pickle \
    --index_dir $DATADIR/${fold}/ \
    --device $DEVICE \
    --checkpoint $fold_ckptdir/gestd/ >> $fold_logdir/gestd.log

    # --data_path "../nete_format_data/gest/reviews.pickle" --index_dir "../nete_format_data/gest/1/" --device "cuda:1" --checkpoint "outputs/gestd/"

    python3 -u reg_predict.py \
    --data_path $DATADIR/reviews.pickle \
    --index_dir $DATADIR/${fold}/ \
    --device $DEVICE \
    --use_mf \
    --checkpoint $fold_ckptdir/gestmf/ >> $fold_logdir/gestmf.log

    python3 -u reg_predict.py \
    --data_path $DATADIR/reviews.pickle \
    --index_dir $DATADIR/${fold}/ \
    --device $DEVICE \
    --rating_reg 1 \
    --checkpoint $fold_ckptdir/gestmlp/ >> $fold_logdir/gestmlp.log
done