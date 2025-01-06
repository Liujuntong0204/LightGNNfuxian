## [WSDM 2025] LightGNN: Simple Graph Neural Network for Recommendation

![image](https://github.com/user-attachments/assets/9ddc22ff-96f3-4fc2-874f-9af87a91a8a3)


<div align="center">
<img src=https://github.com/user-attachments/assets/8e6220a5-23c0-435e-a508-78e2045b5ac0 width=50% />
</div>
<center class="half">
    <img src=https://github.com/user-attachments/assets/8e6220a5-23c0-435e-a508-78e2045b5ac0 width=50% /><img src=https://github.com/user-attachments/assets/9ddc22ff-96f3-4fc2-874f-9af87a91a8a3 width=50% />
</center>

![image](https://github.com/user-attachments/assets/86c731fc-c44e-4b6b-ac4b-3f87748c476c)


## 1. Abstract
Graph neural networks (GNNs) have demonstrated superior performance in collaborative recommendation through their ability to conduct high-order representation smoothing, effectively capturing structural information within users' interaction patterns. However, existing GNN paradigms face significant challenges in scalability and robustness when handling large-scale, noisy, and real-world datasets. To address these challenges, we present LightGNN, a lightweight and distillation-based GNN pruning framework designed to substantially reduce model complexity while preserving essential collaboration modeling capabilities. Our LightGNN framework introduces a computationally efficient pruning module that adaptively identifies and removes redundant edges and embedding entries for model compression. The framework is guided by a resource-friendly hierarchical knowledge distillation objective, whose intermediate
layer augments the observed graph to maintain performance, particularly in high-rate compression scenarios. Extensive experiments on public datasets demonstrate LightGNN's effectiveness, significantly improving both computational efficiency and recommendation accuracy. Notably, LightGNN achieves an 80% reduction in edge count and 90% reduction in embedding entries while maintaining performance comparable to more complex state-of-the-art baselines. The implementation of our LightGNN model is available at the github repository: https://github.com/HKUDS/LightGNN.



## 2. Requirements

```
python == 3.9

pytorch == 1.12.1

torch-sparse == 0.6.15

torch-scatter == 2.0.9

scipy == 1.9.3
```
Please refer to *develop-environment.md* for more details on the development environment.

## Quick Start

After preparing the development environment, we provide trained checkpoints of the teacher model and intermediate model based on the Gowalla dataset as examples to facilitate your quick start. Once the development environment is set up, you can run the following commands to quickly obtain the final student model of pruned LightGNN.


```
python Main.py \
    --dataset='gowalla' \
    --mask_epoch=300 \ ## 300 is an example; increase it for better recommendation performance 
    --lr=0.001 \
    --reg=1e-8 \
    --latdim=64 \
    --gnn_layer=2 \
    --adj_aug=True \
    --adj_aug_layer=1 \ # Necessary for obtaining the final student
    --use_adj_mask_aug=True \ # Necessary for obtaining the final student
    --use_SM_edgeW2aug=True \ # Necessary for obtaining the final student
    --adj_mask_aug1=0.05 \
    --adj_mask_aug2=1.0 \
    --use_mTea2drop_edges=True \
    --use_tea2drop_edges=False  \
    --PRUNING_START=1 \
    --PRUNING_END=13 \
    --model_save_path=./outModels/gowalla/example1/ \
    --his_save_path=./outModels/gowalla/example1/ \
    --middle_teacher_model=./inModels/gowalla/intermediate_KD_model_layer2_dim64_gowalla.mod \
    --distill_from_middle_model=True \
    --distill_from_teacher_model=False \
    --train_middle_model=False \
    --pruning_percent_adj=0.15 \
    --pruning_percent_emb=0.02 | tee ./logs/student_gowalla_example1_log
```

## 2.1 Train the original teacher model 
```
python pretrainTeacher.py \
    --dataset=$dataset \
    --epoch_tea=$epoch_tea \ 
    --lr=$learning_rate_tea \
    --latdim=$dim \
    --gnn_layer=$layer_tea \
    --reg=$decay_weight_tea \
    --model_save_path=$tea_save_path  \
    --his_save_path=$tea_record_save_path  | tee ./logs/teacher_log
```

## 2.2 Train the intermediate KD layer (supervised by the original teacher) as the final teacher model
```
 python Main.py \
    --dataset=$dataset \
    --mask_epoch=$intermediate_train_epoch \
    --lr=$intermediate_train_lr \
    --reg=$intermediate_train_reg \
    --latdim=$dim \
    --gnn_layer=$intermediate_layer \
    --adj_aug=True \
    --adj_aug_layer=1 \
    --use_adj_mask_aug=False \
    --use_SM_edgeW2aug=False  \
    --PRUNING_START=1 \
    --PRUNING_END=1   \
    --model_save_path=$intermediate_mod_path \
    --his_save_path=$intermediate_record \
    --teacher_model=$tea_save_path \
    --distill_from_middle_model=False \
    --distill_from_teacher_model=True \
    --train_middle_model=True | tee ./logs/intermediate_log
```

## 2.3 Train the student model (supervised by the intermediate model)
```
python Main.py \
    --dataset=$dataset \
    --mask_epoch=$stu_epoch \
    --lr=$stu_lr \
    --reg=$stu_reg \
    --latdim=$dim \
    --gnn_layer=$stu_layer \
    --adj_aug=True \
    --adj_aug_layer=1 \
    --use_adj_mask_aug=True \
    --use_SM_edgeW2aug=True \
    --adj_mask_aug1=$adj_mask_aug1 \
    --adj_mask_aug2=$adj_mask_aug2 \
    --use_mTea2drop_edges=True \
    --use_tea2drop_edges=False  \
    --PRUNING_START=1 \
    --PRUNING_END=$pruning_epoch \
    --model_save_path=$student_model_save_path \
    --his_save_path=$stu_record_path \
    --middle_teacher_model=$intermediate_mod_path \
    --distill_from_middle_model=True \
    --distill_from_teacher_model=False \
    --train_middle_model=False \
    --pruning_percent_adj=$edge_pruning_ratio_per_step \
    --pruning_percent_emb=$embedding_pruning_ratio_per_step | tee ./logs/student_log
```
# Supplementary Material
See "**Supplementary Material.pdf**" for __more test results__ of *additional state-of-the-art baseline models* on additional datasets
