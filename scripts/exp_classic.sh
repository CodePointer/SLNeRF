DATA_DIR="C:/SLDataSet/SLNeRF/6_Dataset0523"
OUT_DIR=${DATA_DIR}"-out"

for (( i=0; i<=0; i++ ))  # scene
do
    SCENE_NUM=$(echo $i | awk '{printf("%02d",$0)}')
    for ARGSET in ClassicGrayOnly  # ClassicBFH ClassicBFN
    do
        for (( j=3; j<=11; j++))
        do
          python main.py \
              --argset ${ARGSET} \
              --train_dir ${DATA_DIR}/scene_"${SCENE_NUM}" \
              --out_dir ${OUT_DIR}/scene_"${SCENE_NUM}" \
              --gc_digit ${j} \
              --run_tag gcd"${j}"

          python main.py \
              --argset ${ARGSET} \
              --train_dir ${DATA_DIR}/scene_"${SCENE_NUM}" \
              --out_dir ${OUT_DIR}/scene_"${SCENE_NUM}" \
              --gc_digit ${j} \
              --interpolation \
              --run_tag gcd"${j}"inter
        done
    done
done
