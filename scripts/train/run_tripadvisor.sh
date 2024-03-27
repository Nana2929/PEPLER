# the "_predict.py" files are my revised version for better display of generated text
# ================== TripAdvisor  ==================
# python3 -u main_predict.py \
# --data_path data/TripAdvisor/reviews.pickle \
# --index_dir data/TripAdvisor/1/ \
# --cuda \
# --checkpoint ./tripadvisor/ >> tripadvisor.log

python3 -u discrete_predict.py \
--data_path data/TripAdvisor/reviews.pickle \
--index_dir data/TripAdvisor/1/ \
--cuda \
--checkpoint outputs/tripadvisord/ # >> tripadvisord.log

# python3 -u reg_predict.py \
# --data_path data/TripAdvisor/reviews.pickle \
# --index_dir data/TripAdvisor/1/ \
# --cuda \
# --use_mf \
# --checkpoint ./tripadvisormf/ >> tripadvisormf.log

# python3 -u reg_predict.py \
# --data_path data/TripAdvisor/reviews.pickle \
# --index_dir data/TripAdvisor/1/ \
# --cuda \
# --rating_reg 1 \
# --checkpoint ./tripadvisormlp/ >> tripadvisormlp.log
