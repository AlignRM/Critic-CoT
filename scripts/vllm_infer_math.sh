MODEL_PATH=$1
TRAIN_DATA=$2
BASE_MODEL=$3
TOTAL_BATCH_SIZE=$4
TASK=$5

mkdir -p res
mkdir -p res/${TASK}/prediction
mkdir -p res/${TASK}/critic
mkdir -p res/${TASK}/refine

# Find the test file
export TEST_FILENAMME=$(ls data/${TASK}/test_*.jsonl)
echo "TEST_FILENAMME: ${TEST_FILENAMME}"
export TEST_INPUT=${TEST_FILENAMME}


echo "Majority Vote"
export SELF_SOLUTION=res/${TASK}/prediction/test_prediction_512_${TRAIN_DATA}_${BASE_MODEL}.jsonl
export SELF_CRITIC=res/${TASK}/critic/test_self_critic_whole_process_512_${BASE_MODEL}_${TRAIN_DATA}.jsonl
export SELF_ITERATIVE_REFINE=res/${TASK}/refine/test_self_iterative_refine_512_${BASE_MODEL}_${TRAIN_DATA}.jsonl

python3 src/run_critic_${TASK}_whole_process_vllm.py --mode solve --model ${MODEL_PATH} --src ${TEST_INPUT} --tgt ${SELF_SOLUTION} --num_return_sequences 512 --temperature 0.7 --request_batch_size 4 --max_instance 66666 --max_tokens 2048
python3 src/run_critic_${TASK}_whole_process_vllm.py --mode critic --model ${MODEL_PATH} --src ${SELF_SOLUTION} --tgt ${SELF_CRITIC} --temperature 0.5 --num_return_sequences 1 --request_batch_size ${TOTAL_BATCH_SIZE}
echo ${TRAIN_DATA}  >> evaluate/${TASK}_critic.txt
echo -n "${BASE_MODEL}	self_critic_majority_vote_512	" >> evaluate/${TASK}_critic.txt
python3 evaluate/eval_${TASK}_critic.py --pred_file ${SELF_CRITIC} >> evaluate/${TASK}_critic.txt
python3 src/run_critic_${TASK}_whole_process_vllm.py --mode iterative_refine --model ${MODEL_PATH} --tgt ${SELF_ITERATIVE_REFINE}  --src ${SELF_CRITIC} --temperature 0.5 --num_return_sequences 1 --request_batch_size ${TOTAL_BATCH_SIZE}
echo ${TRAIN_DATA}  >> evaluate/${TASK}_critic.txt
echo -n "${BASE_MODEL}	self_critic_iterative	" >> evaluate/${TASK}_critic.txt
python3 evaluate/eval_${TASK}_critic.py --pred_file ${SELF_ITERATIVE_REFINE} >> evaluate/${TASK}_critic.txt
