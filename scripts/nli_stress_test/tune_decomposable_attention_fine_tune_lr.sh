#!/usr/bin/env bash

if [ -z "$1" ]
then
    echo "First command line argument is null"
    echo "Please call the script with: script [adversary_name]"
    exit 1
fi

ADVERSARY_NAME=$1
if [[ ${ADVERSARY_NAME} = *"mismatched"* ]]; then
    DA_PATH="https://s3-us-west-2.amazonaws.com/ai2-nelsonl/adversarial/models/decomposable_attention_original_mismatched/model.tar.gz"
    echo "USING DECOMPOSABLE ATTENTION ORIGINAL MISMATCHED MODEL at ${DA_PATH}"
else
    DA_PATH="https://s3-us-west-2.amazonaws.com/ai2-nelsonl/adversarial/models/decomposable_attention_original_matched/model.tar.gz"
    echo "USING DECOMPOSABLE ATTENTION ORIGINAL MATCHED MODEL at ${DA_PATH}"
fi

mkdir -p "models/nli_stress_test/decomposable_attention_original/${ADVERSARY_NAME}/"

for num_examples in 5 10 50 100 400 500 750 1000 all; do
    for learning_rate in 0.000001 0.00001 0.0001 0.0004 0.001 0.01; do
        echo "Running with ${num_examples} examples and lr ${learning_rate}"
        allennlp fine-tune \
                 -m "${DA_PATH}" \
                 -c "training_configs/nli_stress_test/decomposable_attention_original/${ADVERSARY_NAME}/fine_tune_decomposable_attention_${ADVERSARY_NAME}.${num_examples}.json" \
                 -s "models/nli_stress_test/decomposable_attention_original/${ADVERSARY_NAME}/fine_tune_decomposable_attention_${ADVERSARY_NAME}.${num_examples}" \
                 -o '{"trainer": {"optimizer": {"lr": '${learning_rate}'}, "num_serialized_models_to_keep": 1}}'
    done
done
