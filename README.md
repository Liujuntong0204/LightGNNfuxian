# 1. Requirements

```
python == 3.9

pytorch == 1.12.1

torch-sparse == 0.6.15

torch-scatter == 2.0.9

scipy == 1.9.3
```

# 2.1 Train the original teacher model 
```
python pretrainTeacher.py --dataset=$dataset --epoch_tea=$epoch_tea  --lr=$learning_rate_tea --latdim=$dim --gnn_layer=$layer_tea --reg=$decay_weight_tea --model_save_path=$tea_save_path  --his_save_path=$tea_record_save_path  | tee ./logs/teacher_log
```

# 2.2 Train the intermediate KD layer (supervised by the original teacher) as the final teacher model
```
 python Main.py --dataset=$dataset --mask_epoch=$intermediate_train_epoch --lr=$intermediate_train_lr --reg=$intermediate_train_reg --latdim=$dim --gnn_layer=$intermediate_layer --adj_aug=True --adj_aug_layer=1 --use_adj_mask_aug=False --use_SM_edgeW2aug=False  --IMP_START=1 --IMP_NUM=1   --model_save_path=$intermediate_mod_path --his_save_path=$intermediate_record --teacher_model=$tea_save_path --distill_from_middle_model=False --distill_from_teacher_model=True --train_middle_model=True | tee ./logs/intermediate_log
```

# 2.3 Train the student model (supervised by the intermediate model)
```
 python Main.py --dataset=$dataset --mask_epoch=$stu_epoch --lr=$stu_lr --reg=$stu_reg --latdim=$dim --gnn_layer=$stu_layer --adj_aug=True --adj_aug_layer=1 --use_adj_mask_aug=True --use_SM_edgeW2aug=True --adj_mask_aug1=$adj_mask_aug1 --adj_mask_aug2=$adj_mask_aug2 --use_mTea2drop_edges=True --use_tea2drop_edges=False  --IMP_START=1 --IMP_NUM=$pruning_epoch --model_save_path=$student_model_save_path --his_save_path=$stu_record_path --middle_teacher_model=$intermediate_mod_path --distill_from_middle_model=True --distill_from_teacher_model=False --train_middle_model=False --pruning_percent_adj=$edge_pruning_ratio_per_step --pruning_percent_emb=$embedding_pruning_ratio_per_step | tee ./logs/student_log
```
# Supplementary Material
See "**Supplementary Material.pdf**" for __more test results__ of *additional state-of-the-art baseline models* on additional datasets
