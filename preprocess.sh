#!/usr/bin/env bash

export INPUT_FILE=$1
export OUTPUT_DIR=$2

mkdir work_dir/${OUTPUT_DIR}

python paragraph_selection/select_paras.py \
    --input_path=${INPUT_FILE} \
    --output_path=work_dir/${OUTPUT_DIR}/selected_paras.json \
    --ckpt_path=work_dir/para_select_model.bin \
    --split=${OUTPUT_DIR}

python bert_ner/predict.py \
    --ckpt_path=work_dir/bert_ner.pt \
    --input_path=work_dir/${OUTPUT_DIR}/selected_paras.json \
    --output_path=work_dir/${OUTPUT_DIR}/entities.json

python bert_ner/predict.py \
    --use_query \
    --ckpt_path=work_dir/bert_ner.pt \
    --input_path=${INPUT_FILE} \
    --output_path=work_dir/${OUTPUT_DIR}/query_entities.json

python DFGN/text_to_tok_pack.py \
    --full_data=${INPUT_FILE} \
    --entity_path=work_dir/${OUTPUT_DIR}/entities.json \
    --para_path=work_dir/${OUTPUT_DIR}/selected_paras.json \
    --example_output=work_dir/${OUTPUT_DIR}/examples.pkl.gz \
    --feature_output=work_dir/${OUTPUT_DIR}/features.pkl.gz \

python DFGN/create_graph.py \
    --example_path=work_dir/${OUTPUT_DIR}/examples.pkl.gz \
    --feature_path=work_dir/${OUTPUT_DIR}/features.pkl.gz \
    --graph_path=work_dir/${OUTPUT_DIR}/graph.pkl.gz \
    --query_entity_path=work_dir/${OUTPUT_DIR}/query_entities.json
