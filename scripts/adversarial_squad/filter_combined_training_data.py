"""
Given a file with the combined train+adversarial data and a copy
of the SQuAD dataset, filter to only the adversarial data.
"""
import argparse
import json
import logging
import os
import random

random.seed(0)
logger = logging.getLogger(__name__)


def main():
    with open(args.squad_data_path) as data_file:
        original_squad_data = json.load(data_file)

    # Go through original SQuAD dev set and get a set of all sentences
    original_squad_contexts = set()
    for document in original_squad_data["data"]:
        for paragraph in document["paragraphs"]:
            original_squad_contexts.add(paragraph["context"])
    logger.info("Read {} different SQuAD contexts".format(len(
        original_squad_contexts)))

    with open(args.combined_squad_adversarial_data_path) as data_file:
        all_data = json.load(data_file)

    # Go through the paragraphs in all_data, removing those that have a context
    # that is identical to one in the SQuAD dataset.
    num_removed = 0
    for article in all_data["data"]:
        original_size = len(article["paragraphs"])
        article["paragraphs"] = [x for x in article["paragraphs"] if
                                 x["context"] not in original_squad_contexts]
        num_removed += (original_size - len(article["paragraphs"]))

    logger.info("Removed {} contexts from combined dataset".format(num_removed))

    logger.info("Writing output to {}".format(args.output_path))
    with open(args.output_path, "w") as output_file:
        json.dump(all_data, output_file, indent=4)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s "
                        "- %(name)s - %(message)s",
                        level=logging.INFO)
    # Path to directory that this file is in
    project_root = os.path.abspath(os.path.join(os.path.realpath(
        os.path.dirname(os.path.realpath(__file__))), os.pardir, os.pardir))

    parser = argparse.ArgumentParser(
        description=("Subsample a dataset in the SQuAD format."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--squad-data-path", type=str,
                        default=os.path.join(
                            project_root, "data", "adversarial_squad",
                            "train-v1.1.json"),
                        help=("Path to SQuAD v1.1 data."))
    parser.add_argument("--combined-squad-adversarial-data-path", type=str,
                        default=os.path.join(
                            project_root, "data", "adversarial_squad",
                            "adversarial+squad-train-set.json"),
                        help=("Path to combined SQuAD + adversarial "
                              "generated data."))
    parser.add_argument("--output-path", type=str,
                        default=os.path.join(
                            project_root, "data", "adversarial_squad",
                            "adversarial-train-set.json"),
                        help=("Path to write the adversarial SQuAD data."))
    args = parser.parse_args()
    main()
