


for IPC in 10 50 100; do
    echo "===== Running IPC=${IPC} ====="

    RUN_TAG="FairDD_DC_BFFHQ_ipc${IPC}"
    # TMP_LOG="result/run_${RUN_TAG}.log"
    # FINAL_LOG="result/run_${RUN_TAG}.log"
    FINAL_RES="result/${RUN_TAG}"

    mkdir -p "$FINAL_RES"

    # run
    python main_DC_FairDD.py \
        --dataset BFFHQ \
        --ipc "$IPC" \
        --Iteration 700 \
        --save_path "result/${RUN_TAG}"
        #  \
        # 2>&1 | tee "$TMP_LOG"



done
