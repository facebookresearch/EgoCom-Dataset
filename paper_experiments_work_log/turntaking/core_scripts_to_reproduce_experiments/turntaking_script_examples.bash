# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

{ time CUDA_VISIBLE_DEVICES=0 python3 nn_turntaking_all_experiments.py --param-idx 0 --num-param-blocks 4 --epochs 30  --prediction-task binary; } &> out_0_4_false_false_binary.log &
tail -f out_0_4_false_false_binary.log;

{ time CUDA_VISIBLE_DEVICES=1 python3 nn_turntaking_all_experiments.py --param-idx 1 --num-param-blocks 4 --epochs 30  --prediction-task binary; } &> out_1_4_false_false_binary.log &
tail -f out_1_4_false_false_binary.log;

{ time CUDA_VISIBLE_DEVICES=2 python3 nn_turntaking_all_experiments.py --param-idx 2 --num-param-blocks 4 --epochs 30  --prediction-task binary; } &> out_2_4_false_false_binary.log &
tail -f out_2_4_false_false_binary.log;

{ time CUDA_VISIBLE_DEVICES=3 python3 nn_turntaking_all_experiments.py --param-idx 3 --num-param-blocks 4 --epochs 30  --prediction-task binary; } &> out_3_4_false_false_binary.log &
tail -f out_3_4_false_false_binary.log;


{ time CUDA_VISIBLE_DEVICES=0 python3 nn_turntaking_all_experiments.py --param-idx 0 --num-param-blocks 4 --epochs 30 --predict-only-host --prediction-task binary; } &> out_0_4_false_true_binary.log &
tail -f out_0_4_false_true_binary.log;

{ time CUDA_VISIBLE_DEVICES=1 python3 nn_turntaking_all_experiments.py --param-idx 1 --num-param-blocks 4 --epochs 30 --predict-only-host --prediction-task binary; } &> out_1_4_false_true_binary.log &
tail -f out_1_4_false_true_binary.log;

{ time CUDA_VISIBLE_DEVICES=2 python3 nn_turntaking_all_experiments.py --param-idx 2 --num-param-blocks 4 --epochs 30 --predict-only-host --prediction-task binary; } &> out_2_4_false_true_binary.log &
tail -f out_2_4_false_true_binary.log;

{ time CUDA_VISIBLE_DEVICES=3 python3 nn_turntaking_all_experiments.py --param-idx 3 --num-param-blocks 4 --epochs 30 --predict-only-host --prediction-task binary; } &> out_3_4_false_true_binary.log &
tail -f out_3_4_false_true_binary.log;



{ time CUDA_VISIBLE_DEVICES=0 python3 nn_turntaking_all_experiments.py --param-idx 0 --num-param-blocks 24 --epochs 30 --predict-only-host --prediction-task binary; } &> out_0_24_false_true_binary.log &
tail -f out_0_24_false_true_binary.log;

{ time CUDA_VISIBLE_DEVICES=1 python3 nn_turntaking_all_experiments.py --param-idx 1 --num-param-blocks 24 --epochs 30 --predict-only-host --prediction-task binary; } &> out_1_24_false_true_binary.log &
tail -f out_1_24_false_true_binary.log;

{ time CUDA_VISIBLE_DEVICES=2 python3 nn_turntaking_all_experiments.py --param-idx 2 --num-param-blocks 24 --epochs 30 --predict-only-host --prediction-task binary; } &> out_2_24_false_true_binary.log &
tail -f out_2_24_false_true_binary.log;

{ time CUDA_VISIBLE_DEVICES=3 python3 nn_turntaking_all_experiments.py --param-idx 3 --num-param-blocks 24 --epochs 30 --predict-only-host --prediction-task binary; } &> out_3_24_false_true_binary.log &
tail -f out_3_24_false_true_binary.log;

{ time CUDA_VISIBLE_DEVICES=0 python3 nn_turntaking_all_experiments.py --param-idx 4 --num-param-blocks 24 --epochs 30 --predict-only-host --prediction-task binary; } &> out_4_24_false_true_binary.log &
tail -f out_4_24_false_true_binary.log;

{ time CUDA_VISIBLE_DEVICES=1 python3 nn_turntaking_all_experiments.py --param-idx 5 --num-param-blocks 24 --epochs 30 --predict-only-host --prediction-task binary; } &> out_5_24_false_true_binary.log &
tail -f out_5_24_false_true_binary.log;

{ time CUDA_VISIBLE_DEVICES=2 python3 nn_turntaking_all_experiments.py --param-idx 6 --num-param-blocks 24 --epochs 30 --predict-only-host --prediction-task binary; } &> out_6_24_false_true_binary.log &
tail -f out_6_24_false_true_binary.log;

{ time CUDA_VISIBLE_DEVICES=3 python3 nn_turntaking_all_experiments.py --param-idx 7 --num-param-blocks 24 --epochs 30 --predict-only-host --prediction-task binary; } &> out_7_24_false_true_binary.log &
tail -f out_7_24_false_true_binary.log;

{ time CUDA_VISIBLE_DEVICES=0 python3 nn_turntaking_all_experiments.py --param-idx 8 --num-param-blocks 24 --epochs 30 --predict-only-host --prediction-task binary; } &> out_8_24_false_true_binary.log &
tail -f out_8_24_false_true_binary.log;

{ time CUDA_VISIBLE_DEVICES=1 python3 nn_turntaking_all_experiments.py --param-idx 9 --num-param-blocks 24 --epochs 30 --predict-only-host --prediction-task binary; } &> out_9_24_false_true_binary.log &
tail -f out_9_24_false_true_binary.log;

{ time CUDA_VISIBLE_DEVICES=2 python3 nn_turntaking_all_experiments.py --param-idx 10 --num-param-blocks 24 --epochs 30 --predict-only-host --prediction-task binary; } &> out_10_24_false_true_binary.log &
tail -f out_10_24_false_true_binary.log;

{ time CUDA_VISIBLE_DEVICES=3 python3 nn_turntaking_all_experiments.py --param-idx 11 --num-param-blocks 24 --epochs 30 --predict-only-host --prediction-task binary; } &> out_11_24_false_true_binary.log &
tail -f out_11_24_false_true_binary.log;




{ time CUDA_VISIBLE_DEVICES=0 python3 nn_turntaking_all_experiments.py --param-idx 12 --num-param-blocks 24 --epochs 30 --predict-only-host --prediction-task binary; } &> out_12_24_false_true_binary.log &
tail -f out_12_24_false_true_binary.log;

{ time CUDA_VISIBLE_DEVICES=1 python3 nn_turntaking_all_experiments.py --param-idx 13 --num-param-blocks 24 --epochs 30 --predict-only-host --prediction-task binary; } &> out_13_24_false_true_binary.log &
tail -f out_13_24_false_true_binary.log;

{ time CUDA_VISIBLE_DEVICES=2 python3 nn_turntaking_all_experiments.py --param-idx 14 --num-param-blocks 24 --epochs 30 --predict-only-host --prediction-task binary; } &> out_14_24_false_true_binary.log &
tail -f out_14_24_false_true_binary.log;

{ time CUDA_VISIBLE_DEVICES=3 python3 nn_turntaking_all_experiments.py --param-idx 15 --num-param-blocks 24 --epochs 30 --predict-only-host --prediction-task binary; } &> out_15_24_false_true_binary.log &
tail -f out_15_24_false_true_binary.log;

{ time CUDA_VISIBLE_DEVICES=0 python3 nn_turntaking_all_experiments.py --param-idx 16 --num-param-blocks 24 --epochs 30 --predict-only-host --prediction-task binary; } &> out_16_24_false_true_binary.log &
tail -f out_16_24_false_true_binary.log;

{ time CUDA_VISIBLE_DEVICES=1 python3 nn_turntaking_all_experiments.py --param-idx 17 --num-param-blocks 24 --epochs 30 --predict-only-host --prediction-task binary; } &> out_17_24_false_true_binary.log &
tail -f out_17_24_false_true_binary.log;

{ time CUDA_VISIBLE_DEVICES=2 python3 nn_turntaking_all_experiments.py --param-idx 18 --num-param-blocks 24 --epochs 30 --predict-only-host --prediction-task binary; } &> out_18_24_false_true_binary.log &
tail -f out_18_24_false_true_binary.log;

{ time CUDA_VISIBLE_DEVICES=3 python3 nn_turntaking_all_experiments.py --param-idx 19 --num-param-blocks 24 --epochs 30 --predict-only-host --prediction-task binary; } &> out_19_24_false_true_binary.log &
tail -f out_19_24_false_true_binary.log;

{ time CUDA_VISIBLE_DEVICES=0 python3 nn_turntaking_all_experiments.py --param-idx 20 --num-param-blocks 24 --epochs 30 --predict-only-host --prediction-task binary; } &> out_20_24_false_true_binary.log &
tail -f out_20_24_false_true_binary.log;

{ time CUDA_VISIBLE_DEVICES=1 python3 nn_turntaking_all_experiments.py --param-idx 21 --num-param-blocks 24 --epochs 30 --predict-only-host --prediction-task binary; } &> out_21_24_false_true_binary.log &
tail -f out_21_24_false_true_binary.log;

{ time CUDA_VISIBLE_DEVICES=2 python3 nn_turntaking_all_experiments.py --param-idx 22 --num-param-blocks 24 --epochs 30 --predict-only-host --prediction-task binary; } &> out_22_24_false_true_binary.log &
tail -f out_22_24_false_true_binary.log;

{ time CUDA_VISIBLE_DEVICES=3 python3 nn_turntaking_all_experiments.py --param-idx 23 --num-param-blocks 24 --epochs 30 --predict-only-host --prediction-task binary; } &> out_23_24_false_true_binary.log &
tail -f out_23_24_false_true_binary.log;








{ time CUDA_VISIBLE_DEVICES=0 python3 nn_turntaking_all_experiments.py --param-idx 0 --num-param-blocks 4 --epochs 30 --use-all-perspectives --predict-only-host --prediction-task binary; } &> out_0_4_true_true_binary.log &
tail -f out_0_4_true_true_binary.log;

{ time CUDA_VISIBLE_DEVICES=1 python3 nn_turntaking_all_experiments.py --param-idx 1 --num-param-blocks 4 --epochs 30 --use-all-perspectives --predict-only-host --prediction-task binary; } &> out_1_4_true_true_binary.log &
tail -f out_1_4_true_true_binary.log;

{ time CUDA_VISIBLE_DEVICES=2 python3 nn_turntaking_all_experiments.py --param-idx 2 --num-param-blocks 4 --epochs 30 --use-all-perspectives --predict-only-host --prediction-task binary; } &> out_2_4_true_true_binary.log &
tail -f out_2_4_true_true_binary.log;

{ time CUDA_VISIBLE_DEVICES=3 python3 nn_turntaking_all_experiments.py --param-idx 3 --num-param-blocks 4 --epochs 30 --use-all-perspectives --predict-only-host --prediction-task binary; } &> out_3_4_true_true_binary.log &
tail -f out_3_4_true_true_binary.log;



{ time CUDA_VISIBLE_DEVICES=0 python3 nn_turntaking_all_experiments.py --param-idx 0 --num-param-blocks 4 --epochs 30 --prediction-task multi ; } &> out_0_4_multi.log &
tail -f out_0_4_multi.log;

{ time CUDA_VISIBLE_DEVICES=1 python3 nn_turntaking_all_experiments.py --param-idx 1 --num-param-blocks 4 --epochs 30 --prediction-task multi ; } &> out_1_4_multi.log &
tail -f out_1_4_multi.log;

{ time CUDA_VISIBLE_DEVICES=2 python3 nn_turntaking_all_experiments.py --param-idx 2 --num-param-blocks 4 --epochs 30 --prediction-task multi ; } &> out_2_4_multi.log &
tail -f out_2_4_multi.log;

{ time CUDA_VISIBLE_DEVICES=3 python3 nn_turntaking_all_experiments.py --param-idx 3 --num-param-blocks 4 --epochs 30 --prediction-task multi ; } &> out_3_4_multi.log &
tail -f out_3_4_multi.log;



























# NO PRIOR

{ time CUDA_VISIBLE_DEVICES=0 python3 nn_turntaking_all_experiments.py --param-idx 0 --num-param-blocks 4 --prediction-task binary; } &> out_0_4_false_false_false_binary.log &
tail -f out_0_4_false_false_false_binary.log;

{ time CUDA_VISIBLE_DEVICES=1 python3 nn_turntaking_all_experiments.py --param-idx 1 --num-param-blocks 4 --prediction-task binary; } &> out_1_4_false_false_false_binary.log &
tail -f out_1_4_false_false_false_binary.log;

{ time CUDA_VISIBLE_DEVICES=2 python3 nn_turntaking_all_experiments.py --param-idx 2 --num-param-blocks 4 --prediction-task binary; } &> out_2_4_false_false_false_binary.log &
tail -f out_2_4_false_false_false_binary.log;

{ time CUDA_VISIBLE_DEVICES=3 python3 nn_turntaking_all_experiments.py --param-idx 3 --num-param-blocks 4 --prediction-task binary; } &> out_3_4_false_false_false_binary.log &
tail -f out_3_4_false_false_false_binary.log;


{ time CUDA_VISIBLE_DEVICES=0 python3 nn_turntaking_all_experiments.py --param-idx 0 --num-param-blocks 4 --predict-only-host --prediction-task binary; } &> out_0_4_false_true_false_binary.log &
tail -f out_0_4_false_true_false_binary.log;

{ time CUDA_VISIBLE_DEVICES=1 python3 nn_turntaking_all_experiments.py --param-idx 1 --num-param-blocks 4 --predict-only-host --prediction-task binary; } &> out_1_4_false_true_false_binary.log &
tail -f out_1_4_false_true_false_binary.log;

{ time CUDA_VISIBLE_DEVICES=2 python3 nn_turntaking_all_experiments.py --param-idx 2 --num-param-blocks 4 --predict-only-host --prediction-task binary; } &> out_2_4_false_true_false_binary.log &
tail -f out_2_4_false_true_false_binary.log;

{ time CUDA_VISIBLE_DEVICES=3 python3 nn_turntaking_all_experiments.py --param-idx 3 --num-param-blocks 4 --predict-only-host --prediction-task binary; } &> out_3_4_false_true_false_binary.log &
tail -f out_3_4_false_true_false_binary.log;



{ time CUDA_VISIBLE_DEVICES=0 python3 nn_turntaking_all_experiments.py --param-idx 0 --num-param-blocks 4 --use-all-perspectives --predict-only-host --prediction-task binary; } &> out_0_4_true_true_false_binary.log &
tail -f out_0_4_true_true_false_binary.log;

{ time CUDA_VISIBLE_DEVICES=1 python3 nn_turntaking_all_experiments.py --param-idx 1 --num-param-blocks 4 --use-all-perspectives --predict-only-host --prediction-task binary; } &> out_1_4_true_true_false_binary.log &
tail -f out_1_4_true_true_false_binary.log;

{ time CUDA_VISIBLE_DEVICES=2 python3 nn_turntaking_all_experiments.py --param-idx 2 --num-param-blocks 4 --use-all-perspectives --predict-only-host --prediction-task binary; } &> out_2_4_true_true_false_binary.log &
tail -f out_2_4_true_true_false_binary.log;

{ time CUDA_VISIBLE_DEVICES=3 python3 nn_turntaking_all_experiments.py --param-idx 3 --num-param-blocks 4 --use-all-perspectives --predict-only-host --prediction-task binary; } &> out_3_4_true_true_false_binary.log &
tail -f out_3_4_true_true_false_binary.log;



{ time CUDA_VISIBLE_DEVICES=0 python3 nn_turntaking_all_experiments.py --param-idx 0 --num-param-blocks 4 --prediction-task multi ; } &> out_0_4_multi_no_prior.log &
tail -f out_0_4_multi_no_prior.log;

{ time CUDA_VISIBLE_DEVICES=1 python3 nn_turntaking_all_experiments.py --param-idx 1 --num-param-blocks 4 --prediction-task multi ; } &> out_1_4_multi_no_prior.log &
tail -f out_1_4_multi_no_prior.log;

{ time CUDA_VISIBLE_DEVICES=2 python3 nn_turntaking_all_experiments.py --param-idx 2 --num-param-blocks 4 --prediction-task multi ; } &> out_2_4_multi_no_prior.log &
tail -f out_2_4_multi_no_prior.log;

{ time CUDA_VISIBLE_DEVICES=3 python3 nn_turntaking_all_experiments.py --param-idx 3 --num-param-blocks 4 --prediction-task multi ; } &> out_3_4_multi_no_prior.log &
tail -f out_3_4_multi_no_prior.log;




