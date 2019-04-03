#!/usr/bin/env bash

if [ -z "$1" ]
then
    echo "First command line argument is null"
    echo "Please call the script with: script [adversary_name]"
    exit 1
fi

ADVERSARY_NAME=$1
if [[ ${ADVERSARY_NAME} = *"mismatched"* ]]; then
    ESIM_PATH="https://s3-us-west-2.amazonaws.com/ai2-nelsonl/adversarial/models/esim_character_level_mismatched/model.tar.gz"
    echo "USING MISMATCHED CHARACTER-LEVEL MODEL at ${ESIM_PATH}"
else
    ESIM_PATH="https://s3-us-west-2.amazonaws.com/ai2-nelsonl/adversarial/models/esim_character_level_matched/model.tar.gz"
    echo "USING MATCHED CHARACTER-LEVEL MODEL at ${ESIM_PATH}"
fi

mkdir -p "models/nli_stress_test/esim_character_level/${ADVERSARY_NAME}/"

for num_examples in 5 10 50 100 400 500 750 1000 all; do
    for learning_rate in 0.000001 0.00001 0.0001 0.0004 0.001 0.01; do
        echo "Running with ${num_examples} examples and lr ${learning_rate}"
        allennlp fine-tune \
                 -m "${ESIM_PATH}" \
                 -c "training_configs/nli_stress_test/esim_character_level/${ADVERSARY_NAME}/fine_tune_esim_${ADVERSARY_NAME}.${num_examples}.json" \
                 -s "models/nli_stress_test/esim_character_level/${ADVERSARY_NAME}/fine_tune_esim_${ADVERSARY_NAME}.${num_examples}" \
                 -o '{"trainer": {"optimizer": {"lr": '${learning_rate}'}, "num_serialized_models_to_keep": 1}}'
    done
done
