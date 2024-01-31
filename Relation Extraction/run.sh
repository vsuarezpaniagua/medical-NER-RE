#! /bin/bash

mkdir submitir


sentence_paths_test="data/testing/scenario1-main/input_scenario1.txt"
entities_paths_test="data/submit/scenario1-main/output_a_scenario1.txt"
relations_paths_test="data/submit/scenario1-main/output_b_scenario1.txt"


./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/baseline_cnn/checkpoints/model-9006"
cp runs/baseline_cnn/prediction-9006.txt submitir/baseline_cnn_model-9006_testmain.txt
./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/baseline_sampling_cnn/checkpoints/model-11431"
cp runs/baseline_sampling_cnn/prediction-11431.txt submitir/baseline_sampling_cnn_model-11431_testmain.txt
./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/position_cnn/checkpoints/model-7110"
cp runs/position_cnn/prediction-7110.txt submitir/position_cnn_model-7110_testmain.txt
./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/position_sampling_cnn/checkpoints/model-8946"
cp runs/position_sampling_cnn/prediction-8946.txt submitir/position_sampling_cnn_model-8946_testmain.txt

./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/baseline_rnn/checkpoints/model-5688"
cp runs/baseline_cnn/prediction-5688.txt submitir/baseline_rnn_model-5688_testmain.txt
./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/baseline_sampling_rnn/checkpoints/model-4970"
cp runs/baseline_sampling_rnn/prediction-4970.txt submitir/baseline_sampling_rnn_model-4970_testmain.txt
./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/position_rnn/checkpoints/model-11376"
cp runs/position_rnn/prediction-11376.txt submitir/position_rnn_model-11376_testmain.txt
./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/position_sampling_rnn/checkpoints/model-3479"
cp runs/position_sampling_rnn/prediction-3479.txt submitir/position_sampling_rnn_model-3479_testmain.txt

./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/baseline_cnnrnn/checkpoints/model-9480"
cp runs/baseline_cnnrnn/prediction-9480.txt submitir/baseline_cnnrnn_model-9480_testmain.txt
./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/baseline_sampling_cnnrnn/checkpoints/model-3479"
cp runs/baseline_sampling_cnnrnn/prediction-3479.txt submitir/baseline_sampling_cnnrnn_model-3479_testmain.txt
./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/position_cnnrnn/checkpoints/model-9954"
cp runs/position_cnnrnn/prediction-9954.txt submitir/position_cnnrnn_model-9954_testmain.txt
./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/position_sampling_cnnrnn/checkpoints/model-2982"
cp runs/position_sampling_cnnrnn/prediction-2982.txt submitir/position_sampling_cnnrnn_model-2982_testmain.txt


sentence_paths_test="data/testing/scenario2-taskA/input_scenario2.txt"
entities_paths_test="data/submit/scenario2-taskA/output_a_scenario2.txt"
relations_paths_test="data/submit/scenario2-taskA/output_b_scenario2.txt"


./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/baseline_cnn/checkpoints/model-9006"
cp runs/baseline_cnn/prediction-9006.txt submitir/baseline_cnn_model-9006_testA.txt
./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/baseline_sampling_cnn/checkpoints/model-11431"
cp runs/baseline_sampling_cnn/prediction-11431.txt submitir/baseline_sampling_cnn_model-11431_testA.txt
./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/position_cnn/checkpoints/model-7110"
cp runs/position_cnn/prediction-7110.txt submitir/position_cnn_model-7110_testA.txt
./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/position_sampling_cnn/checkpoints/model-8946"
cp runs/position_sampling_cnn/prediction-8946.txt submitir/position_sampling_cnn_model-8946_testA.txt

./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/baseline_rnn/checkpoints/model-5688"
cp runs/baseline_cnn/prediction-5688.txt submitir/baseline_rnn_model-5688_testA.txt
./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/baseline_sampling_rnn/checkpoints/model-4970"
cp runs/baseline_sampling_rnn/prediction-4970.txt submitir/baseline_sampling_rnn_model-4970_testA.txt
./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/position_rnn/checkpoints/model-11376"
cp runs/position_rnn/prediction-11376.txt submitir/position_rnn_model-11376_testA.txt
./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/position_sampling_rnn/checkpoints/model-3479"
cp runs/position_sampling_rnn/prediction-3479.txt submitir/position_sampling_rnn_model-3479_testA.txt

./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/baseline_cnnrnn/checkpoints/model-9480"
cp runs/baseline_cnnrnn/prediction-9480.txt submitir/baseline_cnnrnn_model-9480_testA.txt
./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/baseline_sampling_cnnrnn/checkpoints/model-3479"
cp runs/baseline_sampling_cnnrnn/prediction-3479.txt submitir/baseline_sampling_cnnrnn_model-3479_testA.txt
./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/position_cnnrnn/checkpoints/model-9954"
cp runs/position_cnnrnn/prediction-9954.txt submitir/position_cnnrnn_model-9954_testA.txt
./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/position_sampling_cnnrnn/checkpoints/model-2982"
cp runs/position_sampling_cnnrnn/prediction-2982.txt submitir/position_sampling_cnnrnn_model-2982_testA.txt


sentence_paths_test="data/testing/scenario3-taskB/input_scenario3.txt"
entities_paths_test="data/testing/scenario3-taskB/output_a_scenario3.txt"
relations_paths_test="data/submit/scenario3-taskB/output_b_scenario3.txt"


./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/baseline_cnn/checkpoints/model-9006"
cp runs/baseline_cnn/prediction-9006.txt submitir/baseline_cnn_model-9006_testB.txt
./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/baseline_sampling_cnn/checkpoints/model-11431"
cp runs/baseline_sampling_cnn/prediction-11431.txt submitir/baseline_sampling_cnn_model-11431_testB.txt
./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/position_cnn/checkpoints/model-7110"
cp runs/position_cnn/prediction-7110.txt submitir/position_cnn_model-7110_testB.txt
./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/position_sampling_cnn/checkpoints/model-8946"
cp runs/position_sampling_cnn/prediction-8946.txt submitir/position_sampling_cnn_model-8946_testB.txt

./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/baseline_rnn/checkpoints/model-5688"
cp runs/baseline_cnn/prediction-5688.txt submitir/baseline_rnn_model-5688_testB.txt
./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/baseline_sampling_rnn/checkpoints/model-4970"
cp runs/baseline_sampling_rnn/prediction-4970.txt submitir/baseline_sampling_rnn_model-4970_testB.txt
./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/position_rnn/checkpoints/model-11376"
cp runs/position_rnn/prediction-11376.txt submitir/position_rnn_model-11376_testB.txt
./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/position_sampling_rnn/checkpoints/model-3479"
cp runs/position_sampling_rnn/prediction-3479.txt submitir/position_sampling_rnn_model-3479_testB.txt

./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/baseline_cnnrnn/checkpoints/model-9480"
cp runs/baseline_cnnrnn/prediction-9480.txt submitir/baseline_cnnrnn_model-9480_testB.txt
./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/baseline_sampling_cnnrnn/checkpoints/model-3479"
cp runs/baseline_sampling_cnnrnn/prediction-3479.txt submitir/baseline_sampling_cnnrnn_model-3479_testB.txt
./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/position_cnnrnn/checkpoints/model-9954"
cp runs/position_cnnrnn/prediction-9954.txt submitir/position_cnnrnn_model-9954_testB.txt
./test.py --test_path=$sentence_paths_test --test_entity_path=$entities_paths_test --test_relation_path=$relations_paths_test --checkpoint_file="./runs/position_sampling_cnnrnn/checkpoints/model-2982"
cp runs/position_sampling_cnnrnn/prediction-2982.txt submitir/position_sampling_cnnrnn_model-2982_testB.txt
