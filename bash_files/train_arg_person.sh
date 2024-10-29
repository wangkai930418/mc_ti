#!/bin/bash

CAT_NAME=$1
CLS_LABEL=$2
GPU_ID=0
N_SHOT=5
TRAIN_STEP=3100
GAMMA=1.0
SCALE=10.0
GAMMA_END=0.0001
SAVE_STEP=100
WARMUP_STEP=3000
INIT_WORD="person"
SAVE_NAME="person_cls"
SUBDIR=${N_SHOT}_${TRAIN_STEP}_${GAMMA}_${SCALE}_${INIT_WORD}
DATA_DIR="./data/person/images_class_n_shots/${N_SHOT}/${CAT_NAME}"
MODEL_NAME="runwayml/stable-diffusion-v1-5"

accelerate launch --gpu_ids $GPU_ID mcti.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<person-toy>" \
  --initializer_token="${INIT_WORD}" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=${TRAIN_STEP} \
  --learning_rate=5.0e-04 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --mixed_precision "fp16" \
  --output_dir="output/${SAVE_NAME}/${SUBDIR}/${CAT_NAME}" \
  --class_label ${CLS_LABEL} \
  --concept_category "person_${N_SHOT}_shots" \
  --dataloader_num_workers 2 \
  --project_name ${SUBDIR} \
  --run_name ${CAT_NAME} \
  --save_steps ${SAVE_STEP} \
  --cls_warm_up_step ${WARMUP_STEP} \
  --no-proto_feat \
  --logit_scale ${SCALE} \
  --gamma ${GAMMA} \
  --gamma_end ${GAMMA_END} \
  --dataset_name "person"