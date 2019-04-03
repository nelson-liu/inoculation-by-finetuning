import argparse
import json
import logging
import os
import random

from tqdm import tqdm

random.seed(0)
logger = logging.getLogger(__name__)


def main():
    with open(args.data_path) as data_file:
        all_data = json.load(data_file)

    num_paragraphs = 0
    for article in all_data["data"]:
        for paragraph in article["paragraphs"]:
            num_paragraphs += 1
    logger.info("{} paragraphs in the data".format(num_paragraphs))

    if args.num_examples <= len(all_data["data"]):
        # If we want to subsamples less or equal to # of paragraphs,
        # take one (or zero) examples for each document

        # Shuffle the list of documents
        random.shuffle(all_data["data"])

        # Slice off the number of documents we want.
        all_data["data"] = all_data["data"][:args.num_examples]

        for document in tqdm(all_data["data"]):
            # Randomly pick one paragraph from each of the remaining documents.
            selected_paragraph = random.choice(document["paragraphs"])
            # Randomly pick one question from the selected paragraph,
            # removing the rest.
            selected_paragraph["qas"] = [
                random.choice(selected_paragraph["qas"])]

            # Remove the rest of the paragraphs, leaving only the selected
            # one with its selected question.
            document["paragraphs"] = [selected_paragraph]
    elif args.num_examples <= num_paragraphs:
        # Pick the paragraphs we will use.
        paragraph_ids = list(range(num_paragraphs))
        random.shuffle(paragraph_ids)
        paragraphs_to_use = set(paragraph_ids[:args.num_examples])
        out_data = []
        out_obj = {'version': all_data['version'], 'data': out_data}
        # Iterate through the dataset
        paragraph_counter = 0
        for article in all_data["data"]:
            out_paragraphs = []
            out_article = {'title': article['title'],
                           'paragraphs': out_paragraphs}
            for paragraph in article["paragraphs"]:
                # Check if we will use this paragraph or not. Pass if not.
                if paragraph_counter in paragraphs_to_use:
                    # Construct a paragraph
                    new_paragraph = {'context': paragraph['context'],
                                     'qas': [random.choice(paragraph["qas"])]}
                    out_paragraphs.append(new_paragraph)
                paragraph_counter += 1
            if out_paragraphs:
                out_data.append(out_article)
        all_data = out_obj
    else:
        raise NotImplementedError("Sampling multiple questions per "
                                  "paragraph is not yet supported.")

    output_filename = (os.path.splitext(args.data_path)[0] +
                       ".{}".format(args.num_examples) + ".json")
    logger.info("Writing output to {}".format(output_filename))
    with open(output_filename, "w") as output_file:
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
    parser.add_argument("--data-path", type=str,
                        default=os.path.join(
                            project_root, "data", "adversarial_squad",
                            "adversarial-train-set.json"),
                        help=("Path to SQuAD formatted data to subsample."))
    parser.add_argument("--num-examples", type=int, required=True,
                        help=("The number of examples to subsample."))
    args = parser.parse_args()
    main()
