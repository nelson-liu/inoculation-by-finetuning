# Inoculation by Fine-Tuning

Code for [_Inoculation by Fine-Tuning: A Method for Analyzing Challenge Datasets_](http://nelsonliu.me/papers/liu+schwartz+smith.naacl2019.pdf), to appear at NAACL 2019.

## Table of Contents

- [Installation](#installation)
- [Training the models used for inoculation](#training-the-models-used-for-inoculation)
- [Re-running the NLI Challenge Dataset Experiments](#re-running-the-nli-challenge-dataset-experiments)
  * [Decomposable Attention](#decomposable-attention)
  * [ESIM](#esim)
  * [ESIM + Character-Level Component](#esim--character-level-component)
- [Re-running the Adversarial SQuAD Experiments](#re-running-the-adversarial-squad-experiments)
  * [BiDAF](#bidaf)
  * [QANet](#qanet)
- [References](#references)

## Installation

This project was developed in Python 3.6, with the AllenNLP framework.

[Conda](https://conda.io/) will set up a virtual environment with the exact
version of Python used for development along with all the dependencies needed to
run the code.

1.  [Download and install Conda](https://conda.io/docs/download.html).

2.  Change your directory to your clone of this repo.

    ```bash
    cd inoculation-by-finetuning
    ```

3.  Create a Conda environment with Python 3.6 .

    ```bash
    conda create -n inoculation python=3.6
    ```

4.  Now activate the Conda environment. You will need to activate the Conda
    environment in each terminal in which you want to run code from this repo.

    ```bash
    source activate inoculation
    ```

5.  Install the required dependencies.

    ```bash
    pip install -r requirements.txt
    ```

## Training the models used for inoculation

You can download the pretrained models we used in the paper with the links below:

- Decomposable Attention ([Matched](https://s3-us-west-2.amazonaws.com/ai2-nelsonl/adversarial/models/decomposable_attention_original_matched/model.tar.gz), [Mismatched](https://s3-us-west-2.amazonaws.com/ai2-nelsonl/adversarial/models/decomposable_attention_original_mismatched/model.tar.gz))
- ESIM ([Matched](https://s3-us-west-2.amazonaws.com/ai2-nelsonl/adversarial/models/esim_original_matched/model.tar.gz), [Mismatched](https://s3-us-west-2.amazonaws.com/ai2-nelsonl/adversarial/models/esim_original_mismatched/model.tar.gz))
- ESIM + character-level component ([Matched](https://s3-us-west-2.amazonaws.com/ai2-nelsonl/adversarial/models/esim_character_level_matched/model.tar.gz), [Mismatched](https://s3-us-west-2.amazonaws.com/ai2-nelsonl/adversarial/models/esim_character_level_mismatched/model.tar.gz))
- [BiDAF](https://s3-us-west-2.amazonaws.com/ai2-nelsonl/adversarial/models/bidaf_original/model.tar.gz)
- [QANet](https://s3-us-west-2.amazonaws.com/ai2-nelsonl/adversarial/models/qanet_original/model.tar.gz)

AllenNLP configuration files for training the NLI models used for inoculation
can be found
in
[`inoculation-by-finetuning/training_configs/nli_stress_test/`](inoculation-by-finetuning/training_configs/nli_stress_test/) .

In particular, the Decomposable Attention model was trained with
`decomposable_attention_original_matched.jsonnet` and
`decomposable_attention_original_mismatched.jsonnet`, the ESIM model was trained
with `esim_character_level_matched.json` and
`esim_character_level_mismatched.json`, and the ESIM + character-level component
model was trained with `esim_character_level_matched.json` and
`esim_character_level_mismatched.json`.

AllenNLP configuration files for training the reading comprehension models used for inoculation
can be found
in
[`inoculation-by-finetuning/training_configs/adversarial_squad/`](inoculation-by-finetuning/training_configs/adversarial_squad/) .

The BiDAF model was trained with `bidaf_original.json`, and the QANet model was trained with `qanet_original.jsonnet`.

To train each of these models again, use the `allennlp train` command like so:

```
allennlp train <path to config file> -s <path to directory to save results>
```

## Re-running the NLI Challenge Dataset Experiments

To reproduce the inoculation results for each of the NLI models (Decomposable
Attention, ESIM, ESIM + character-level component), use the scripts
in [`scripts/nli_stress_test/`](scripts/nli_stress_test/) . These scripts look
for configuration files
in [`training_configs/nli_stress_test/`](training_configs/nli_stress_test/).

Example running commands for each model are given below.

### Decomposable Attention

To rerun inoculation of the decomposable attention model on, say, the matched
negation dataset, we'd simply run:

```
./scripts/nli_stress_test/tune_decomposable_attention_fine_tune_lr.sh matched_negation_adversary
```

where `matched_negation_adversary` is a folder containing configs
at [`training_configs/nli_stress_test/decomposable_attention_original/`](training_configs/nli_stress_test/decomposable_attention_original/).

As another example, if we wanted to run the inoculation of the decomposable attention model on the mismatched spelling advesary, we can use:

```
./scripts/nli_stress_test/tune_decomposable_attention_fine_tune_lr.sh mismatched_spelling_adversary
```

where `mismatched_spelling_adversary` is another directory name
in
[`training_configs/nli_stress_test/decomposable_attention_original/`](training_configs/nli_stress_test/decomposable_attention_original/).

These scripts fine-tune the decomposable attention model on each of the
fine-tuning dataset sizes for a range of learning rates, saving the results to
disk.

### ESIM

To rerun inoculation of the ESIM model on, say, the matched negation dataset,
we'd simply run:

```
./scripts/nli_stress_test/tune_esim_fine_tune_lr.sh matched_negation_adversary
```

where `matched_negation_adversary` is a folder containing configs
at [`training_configs/nli_stress_test/esim_original/`](training_configs/nli_stress_test/esim_original/).

As another example, if we wanted to run the inoculation of the ESIM model on the mismatched spelling advesary, we can use:

```
./scripts/nli_stress_test/tune_esim_fine_tune_lr.sh mismatched_spelling_adversary
```

where `mismatched_spelling_adversary` is another directory name
in
[`training_configs/nli_stress_test/esim_original/`](training_configs/nli_stress_test/esim_original/).

These scripts fine-tune the ESIM model on each of the fine-tuning dataset sizes
for a range of learning rates, saving the results to disk.

### ESIM + Character-Level Component

To rerun inoculation of the ESIM model augmented with a character-level
component on the matched and mismatched spelling adversary datasets, we can run
the following commands:

```
./scripts/nli_stress_test/tune_esim_character_level_fine_tune_lr.sh matched_spelling_adversary
```

```
./scripts/nli_stress_test/tune_esim_character_level_fine_tune_lr.sh mismatched_spelling_adversary
```

One again, note that `matched_spelling_adversary` and
`mismatched_spelling_adversary` are the names of folder containing configs
at
[`training_configs/nli_stress_test/esim_character_level/`](training_configs/nli_stress_test/esim_character_level/).

## Re-running the Adversarial SQuAD Experiments

To reproduce the inoculation results for each of the reading comprehension
models (BiDAF and QANet), use the
scripts in [`scripts/adversarial_squad/`](scripts/adversarial_squad//) . These
scripts look for configuration files
in [`training_configs/adversarial_squad/`](training_configs/adversarial_squad/).

### BiDAF

To rerun BiDAF inoculation experiments, run:

```
./scripts/adversarial_squad/tune_bidaf_fine_tune_lr.sh
```

### QANet

To rerun QANet inoculation experiments, run:

```
./scripts/adversarial_squad/tune_qanet_fine_tune_lr.sh
```

## References

```
@InProceedings{liu-schwartz-smith:2019:NAACL,
  author    = {Liu, Nelson F.  and  Schwartz, Roy  and  Smith, Noah A.},
  title     = {Inoculation by Fine-Tuning: A Method for Analyzing Challenge Datasets},
  booktitle = {Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  year      = {2019}
}
```
