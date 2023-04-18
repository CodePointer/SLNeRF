DATA_DIR="C:/SLDataSet/SLNeRF/1_Collected0218"
OUT_DIR=${DATA_DIR}"-out"

for (( i=9; i<=9; i++ ))
do
    for ARGSET in ClassicGCC  # ClassicBFH ClassicBFN
    do
        NUM=$(echo $i | awk '{printf("%02d",$0)}')
        python ../main.py \
            --argset ${ARGSET} \
            --train_dir ${DATA_DIR}/scene_"${NUM}" \
            --out_dir ${OUT_DIR}/scene_"${NUM}" \
            --run_tag eval34
    done
done
