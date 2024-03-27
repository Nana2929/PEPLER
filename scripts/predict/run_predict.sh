set -e
for dset in "yelp" "yelp23" "gest";do
    for index in 1 2 3 4 5;do
        echo "inferencing on ${dset} ${index} for no-reg"
        python3 -u predict.py --data_path "../nete_format_data/${dset}/reviews.pickle" \
        --index_dir "../nete_format_data/${dset}/${index}/" \
        --model_path "outputs/${index}/${dset}/${dset}mf/model.pt" \
        --words 20 \
        --output_path "outputs/${index}/${dset}/${dset}mf/generated.jsonl"
        echo "inferencing on ${dset} ${index} for no-reg"
        python3 -u predict.py --data_path "../nete_format_data/${dset}/reviews.pickle" \
        --index_dir "../nete_format_data/${dset}/${index}/" \
        --model_path "outputs/${index}/${dset}/${dset}/model.pt" \
        --words 20 \
        --output_path "outputs/${index}/${dset}/${dset}/generated.jsonl"
    done
done