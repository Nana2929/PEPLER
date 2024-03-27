# ================== Amazon/ClothingShoesAndJewelry  ==================
export CUDA_VISIBLE_DEVICES=1
logdir=logs
ckptdir=outputs
# mkdir if not exist
mkdir -p $logdir
mkdir -p $ckptdir
python3 -u main_predict.py \
--data_path data/Amazon/ClothingShoesAndJewelry/reviews.pickle \
--index_dir data/Amazon/ClothingShoesAndJewelry/1/ \
--cuda \
--checkpoint $ckptdir/amzclothing/ >> $logdir/amzclothing.log

python3 -u discrete_predict.py \
--data_path data/Amazon/ClothingShoesAndJewelry/reviews.pickle \
--index_dir data/Amazon/ClothingShoesAndJewelry/1/ \
--cuda \
--checkpoint $ckptdir/amzclothingd/ >> $logdir/amzclothingd.log

python3 -u reg_predict.py \
--data_path data/Amazon/ClothingShoesAndJewelry/reviews.pickle \
--index_dir data/Amazon/ClothingShoesAndJewelry/1/ \
--cuda \
--use_mf \
--checkpoint $ckptdir/amzclothingmf/ >> $logdir/amzclothingmf.log

python3 -u reg_predict.py \
--data_path data/Amazon/ClothingShoesAndJewelry/reviews.pickle \
--index_dir data/Amazon/ClothingShoesAndJewelry/1/ \
--cuda \
--rating_reg 1 \
--checkpoint $ckptdir/amzclothingmlp/ >> $logdir/amzclothingmlp.log