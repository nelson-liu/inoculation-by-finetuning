#!/usr/bin/env bash

mkdir -p "models/adversarial_squad/qanet/"

for num_examples in 5 10 50 100 400 500 750 1000 all; do
    for learning_rate in 0.00001 0.0001 0.001 0.01; do
        echo "Running with ${num_examples} examples and lr ${learning_rate}"
        allennlp fine-tune \
                 -m "https://s3-us-west-2.amazonaws.com/ai2-nelsonl/adversarial/models/qanet_original/model.tar.gz" \
                 -c "training_configs/adversarial_squad/qanet/fine_tune_qanet.${num_examples}.jsonnet" \
                 -s "models/adversarial_squad/qanet/fine_tune_qanet.${num_examples}" \
                 -o '{"trainer": {"optimizer": {"lr": '${learning_rate}'}, "num_serialized_models_to_keep": 1}}'
    done
done
