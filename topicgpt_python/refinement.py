import pandas as pd
import torch
import os
import regex
import traceback
import argparse
from topicgpt_python.utils import *
from anytree import RenderTree
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)


def gen_topic_pairs(topic_sent, verbose=False):
    """
    Return a list of topic pairs based on cosine similarity between topic sentences.

    Parameters:
    - topic_sent (list): List of topic sentences.
    - verbose (bool): If True, prints additional information.

    Returns:
    - list: list with index and cosine score for each topic pair.
    """
    embeddings = model.encode(topic_sent, convert_to_tensor=True)
    if verbose:
        print(f"Calculating cosine similarity between {len(embeddings)} embeddings...")
    cosine_scores = util.cos_sim(embeddings, embeddings).cpu()

    i, j = torch.triu_indices(*cosine_scores.shape, offset=1)
    i_np = i.numpy()
    j_np = j.numpy()
    scores_np = cosine_scores[i_np, j_np].numpy()
    pairs = [{"index": [i_np[k], j_np[k]], "score": scores_np[k]} for k in range(len(i_np))]
    return pairs


def select_topic_pairs(pairs, topic_sent, topics_seen, threshold=0.5):
    """
    Select pairs of topics based on cosine similarity scores and a threshold.
    
    Parameters:
    - pairs (list): List of topic pairs with cosine similarity scores.
    - all_pairs (list): List of all previously selected pairs.
    - topic_sent (list): List of topic sentences.
    - threshold (float): The threshold for selecting pairs.
    - num_pair (int): The number of pairs to select.

    Returns:
    - list: List of selected topic pairs.
    """
    selected_pairs = []
    selected_pairs_set = set(topics_seen)

    with tqdm(total=len(pairs), desc="Selecting topic pairs") as pbar:
        for pair in pairs:
            i, j = pair["index"]
            pair_tuple = tuple(sorted([topic_sent[i], topic_sent[j]]))

            if (
                pair["score"] > threshold
                and pair_tuple not in selected_pairs_set
            ):
                selected_pairs.append(pair_tuple)
                selected_pairs_set.add(pair_tuple)
            pbar.update(1)

    return selected_pairs


def merge_topics(
    topics_root,
    mapping,
    refinement_prompt,
    api_client,
    temperature,
    max_tokens,
    top_p,
    verbose,
    threshold=0.5,
    num_pair=5
):
    """
    Merge similar topics based on a given refinement prompt and API client settings.

    Parameters:
    - topics_root (TopicTree): The root of the topic tree.
    - mapping (dict): Dictionary mapping original topics to new topics.
    - refinement_prompt (str): The prompt to use for refining topics.
    - api_client (APIClient): The API client to use for calling the model.
    - temperature (float): The temperature for model sampling.
    - max_tokens (int): The maximum number of tokens to generate.
    - top_p (float): The nucleus sampling parameter.
    - verbose (bool): If True, prints each replacement made.

    Returns:
    - list: List of responses from the API.
    - TopicTree: The updated topic root with merged topics.
    - dict: The updated mapping of original topics to new topics.
    """
    topic_sent = topics_root.to_topic_list(desc=True, count=False)
    topic_pairs = gen_topic_pairs(
        topic_sent, verbose=verbose
    )
    new_pairs = select_topic_pairs(
        topic_pairs, topic_sent, [], threshold=threshold
    )   
    if len(new_pairs) <= 1 and verbose:
        print("No topic pairs to be merged.")

    responses, orig_new = [], mapping

    pattern_topic = regex.compile(
        r"^\[(\d+)\]([\w\s\-',]+)[^:]*:([\w\s,\.\-\/;']+) \(([^)]+)\)$"
    )
    pattern_original = regex.compile(r"\[(\d+)\]([\w\s\-',]+),?")
    merged_list = []
    topics_seen = set()

    while len(new_pairs) > 1:
        try:
            topics_seen.update(new_pairs[:num_pair])
            pairs = set()
            for pair in new_pairs[:num_pair]:
                pairs.update(pair)
            new_pairs = new_pairs[num_pair:]
            
            refiner_prompt = refinement_prompt.format(Topics="\n".join(pairs))
            if verbose:
                print(f"Prompting model to merge topics:\n{refiner_prompt}")

            response = api_client.iterative_prompt(
                refiner_prompt, max_tokens, temperature, top_p
            )
            merges = response.split("\n")

            responses.append(response)
            update = False
            for merge in merges:
                match = pattern_topic.match(merge.strip())
                if match:
                    lvl, name, desc, originals = (
                        int(match.group(1)),
                        match.group(2).strip(),
                        match.group(3).strip(),
                        match.group(4).strip(),
                    )
                    orig_topics = [
                        t[1].strip(", ")
                        for t in regex.findall(pattern_original, originals)
                    ]
                    orig_lvl = [
                        int(t[0]) for t in regex.findall(pattern_original, originals)
                    ]
                    original_topics = [
                        (orig_topics[i], orig_lvl[i]) for i in range(len(orig_topics))
                    ]
                    # Avoid merging a topic to itself
                    if len(original_topics) == 1 and original_topics[0][0] == name:
                        continue
                    # Avoids loops
                    if (original_topics, name) in merged_list:
                        continue
                    merged_list.append((original_topics, name))
                    topics_root = topics_root.update_tree(original_topics, name, desc)
                    for orig in original_topics:
                        orig_new[orig[0]] = name
                    update = True
                    print(f"Updated topic tree with [{lvl}] {name}: {desc}. Original topics: {original_topics}")

            if update:
                topic_sent = topics_root.to_topic_list(desc=True, count=False)
                if verbose:
                    print("Number of topics:", len(topic_sent))
                topic_pairs = gen_topic_pairs(
                    topic_sent, verbose=verbose
                )
                new_pairs = select_topic_pairs(
                    topic_pairs, topic_sent, topics_seen, threshold=threshold
                )
        except Exception as e:
            print("Error when calling API!")
            traceback.print_exc()
        
    return responses, topics_root, orig_new


def remove_topics(topics_root, verbose, threshold=0.01, threshold_count=None):
    """
    Remove low-frequency topics from topic tree.

    Parameters:
    - topics_root (TopicTree): The root of the topic tree.
    - verbose (bool): If True, prints each removal made.
    - threshold (float): The threshold for removing low-frequency topics.

    Returns:
    - TopicTree: The updated topic root with low-frequency topics removed.
    """
    total_count = sum(node.count for node in topics_root.root.children)
    threshold_count = total_count * threshold if threshold_count is None else threshold_count
    removed = False

    for node in topics_root.root.children:
        if node.count < threshold_count and node.lvl == 1:
            node.parent = None
            if verbose:
                print(f"Removing {node.name} ({node.count} counts)")
            removed = True

    if not removed and verbose:
        print("No topics removed.")

    return topics_root


def update_generation_file(
    generation_file,
    updated_file,
    mapping,
    verbose=False,
    mapping_file=None,
):
    """
    Update the generation JSON file with new topic mappings and save the mapping file.

    Parameters:
    - generation_file (str): Path to the original JSON file with generation data.
    - updated_file (str): Path to save the updated JSON file.
    - mapping (dict): Dictionary mapping original topics to new topics.
    - verbose (bool): If True, prints each replacement made.
    - mapping_file (str): Path to save the mapping as a JSON file.

    Returns:
    - None
    """
    df = pd.read_json(generation_file, lines=True)

    response_column = (
        "refined_responses" if "refined_responses" in df.columns else "responses"
    )
    responses = df[response_column].tolist()
    updated_responses = []
    for response in responses:
        updated_response = "\n".join(
            [replace_topic_key(s, mapping, verbose) for s in response.split("\n")]
        )
        updated_responses.append(updated_response)

    df["refined_responses"] = updated_responses
    df.to_json(updated_file, lines=True, orient="records")

    if mapping_file:
        with open(mapping_file, "w") as f:
            json.dump(mapping, f, indent=4)


def replace_topic_key(text, mapping, verbose=False):
    """
    Replace all occurrences of topic keys in the text based on the provided mapping.

    Parameters:
    - text (str): The input text where replacements are to be made.
    - mapping (dict): Dictionary mapping original topics to new topics.
    - verbose (bool): If True, prints each replacement made.

    Returns:
    - str: The text with topics replaced according to the mapping.
    """
    for key, value in mapping.items():
        if key != value and key in text:
            text = text.replace(key, value)
            if verbose:
                print(f"Replaced '{key}' with '{value}' in text.")
    return text


def refine_topics(
    api,
    model,
    prompt_file,
    generation_file,
    topic_file,
    out_file,
    updated_file,
    verbose,
    remove,
    mapping_file,
    threshold=0.5,
    num_pair=5,
    remove_threshold_count=None,
):
    """
    Main function to refine topics by merging and updating based on API response.

    Parameters:
    - api (str): API to use ('openai', 'vertex', 'vllm', 'gemini', 'azure').
    - model (str): Model to use.
    - prompt_file (str): Path to the prompt file.
    - generation_file (str): Path to the generation JSON file.
    - topic_file (str): Path to the topic file.
    - out_file (str): Path to save the refined topic file.
    - updated_file (str): Path to save the updated generation JSON file.
    - verbose (bool): If True, prints each replacement made.
    - remove (bool): If True, removes low-frequency topics.
    - mapping_file (str): Path to save the mapping as a JSON file.
    - threshold (float): The threshold for selecting pairs.
    - num_pair (int): The number of pairs to select.
    - remove_threshold_count (int): The threshold count for removing low-frequency topics.

    Returns:
    - None
    """
    api_client = APIClient(api=api, model=model)
    max_tokens, temperature, top_p = 1000, 0.0, 1.0
    topics_root = TopicTree().from_topic_list(topic_file, from_file=True)
    if verbose:
        print("-------------------")
        print("Initializing topic refinement...")
        print(f"Model: {model}")
        print(f"Input data file: {generation_file}")
        print(f"Prompt file: {prompt_file}")
        print(f"Output file: {out_file}")
        print(f"Topic file: {topic_file}")
        print("-------------------")

    mapping_org = (
        json.load(open(mapping_file, "r")) if os.path.exists(mapping_file) else {}
    )

    refinement_prompt = open(prompt_file, "r").read()
    responses, updated_topics_root, mapping = merge_topics(
        topics_root,
        mapping_org,
        refinement_prompt,
        api_client,
        temperature,
        max_tokens,
        top_p,
        verbose,
        threshold=threshold,
        num_pair=num_pair
    )

    if mapping_org != mapping and verbose:
        print("Mapping updated:", mapping)

    if remove:
        updated_topics_root = remove_topics(
            updated_topics_root, verbose, threshold_count=remove_threshold_count
        )

    update_generation_file(
        generation_file, updated_file, mapping, verbose, mapping_file
    )

    updated_topics_root.to_file(out_file)
    print(RenderTree(updated_topics_root.root))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api", type=str, help="API to use ('openai', 'vertex', 'vllm', 'gemini', 'azure')"
    )
    parser.add_argument("--model", type=str, help="Model to use")
    parser.add_argument("--prompt_file", type=str, default="prompt/refinement.txt")
    parser.add_argument(
        "--generation_file", type=str, default="data/output/generation_1.jsonl"
    )
    parser.add_argument("--topic_file", type=str, default="data/output/generation_1.md")
    parser.add_argument("--out_file", type=str, default="data/output/refinement_1.md")
    parser.add_argument(
        "--updated_file", type=str, default="data/output/refinement_1.jsonl"
    )
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--remove", type=bool, default=False)
    parser.add_argument(
        "--mapping_file", type=str, default="data/output/refiner_mapping.json"
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num_pair", type=int, default=5)
    parser.add_argument("--remove_threshold_count", type=int, default=None)

    args = parser.parse_args()
    refine_topics(
        args.api,
        args.model,
        args.prompt_file,
        args.generation_file,
        args.topic_file,
        args.out_file,
        args.updated_file,
        args.verbose,
        args.remove,
        args.mapping_file,
        threshold=args.threshold,
        num_pair=args.num_pair,
        remove_threshold_count=args.remove_threshold_count,
    )
