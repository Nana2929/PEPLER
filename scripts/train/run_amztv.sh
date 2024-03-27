# ================== Amazon/MoviesAndTV  ==================
export CUDA_VISIBLE_DEVICES=1 
logdir=logs
ckptdir=outputs
# mkdir if not exist
mkdir -p $logdir
mkdir -p $ckptdir
# python3 -u main_predict.py \
# --data_path data/Amazon/MoviesAndTV/reviews.pickle \
# --index_dir data/Amazon/MoviesAndTV/1/ \
# --cuda \
# --checkpoint $ckptdir/amzmovies/ >> $logdir/amzmovies.log

python3 -u discrete_predict.py \
--data_path data/Amazon/MoviesAndTV/reviews.pickle \
--index_dir data/Amazon/MoviesAndTV/1/ \
--cuda \
--checkpoint $ckptdir/amzmoviesd/ >> $logdir/amzmoviesd.log

python3 -u reg_predict.py \
--data_path data/Amazon/MoviesAndTV/reviews.pickle \
--index_dir data/Amazon/MoviesAndTV/1/ \
--cuda \
--use_mf \
--checkpoint $ckptdir/amzmoviesmf/ >> $logdir/amzmoviesmf.log

python3 -u reg_predict.py \
--data_path data/Amazon/MoviesAndTV/reviews.pickle \
--index_dir data/Amazon/MoviesAndTV/1/ \
--cuda \
--rating_reg 1 \
--checkpoint $ckptdir/amzmoviesmlp/ >> $logdir/amzmoviesmlp.log