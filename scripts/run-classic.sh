DATA_DIR="/media/qiao/Videos/SLDataSet/SLNeuS/9_Dataset0923" 
SCENE_IDX_LIST=( 00 )

for SCENE_IDX in ${SCENE_IDX_LIST[*]}
do
    TRAIN_DIR=${DATA_DIR}"/scene_"${SCENE_IDX}
    OUT_DIR=${DATA_DIR}"-out/scene_"${SCENE_IDX}

    python main.py \
        --argset ClassicBFH \
        --train_dir "${TRAIN_DIR}" \
        --out_dir "${OUT_DIR}" \
        --run_tag hpmp-dyn

    python main.py \
        --argset ClassicBFN \
        --train_dir "${TRAIN_DIR}" \
        --out_dir "${OUT_DIR}" \
        --run_tag npmp-dyn

    python main.py \
        --argset ClassicGCC \
        --train_dir "${TRAIN_DIR}" \
        --out_dir "${OUT_DIR}" \
        --gc_digit 3 \
        --run_tag cgc6-dyn

    python main.py \
        --argset ClassicGCC \
        --train_dir "${TRAIN_DIR}" \
        --out_dir "${OUT_DIR}" \
        --gc_digit 4 \
        --run_tag cgc7-dyn

   for (( i=3; i<10; i++ ))
   do
       python main.py \
           --argset ClassicGrayOnly \
           --train_dir "${TRAIN_DIR}" \
           --out_dir "${OUT_DIR}" \
           --gc_digit ${i} \
           --interpolation \
           --run_tag gcd"${i}"-dyn

       python main.py \
           --argset ClassicGrayOnly \
           --train_dir "${TRAIN_DIR}" \
           --out_dir "${OUT_DIR}" \
           --gc_digit ${i} \
           --interpolation \
           --invert_projection \
           --run_tag gcd"${i}"inv-dyn
   done
done

