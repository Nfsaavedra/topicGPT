import pandas as pd
from topicgpt_python.utils import *

from tqdm import trange, tqdm
import traceback
from sentence_transformers import SentenceTransformer, util
import argparse
import os
import concurrent.futures
from typing import List, Tuple, Dict, Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def assignment_batch(
    api_client,
    topics_root,
    docs,
    assignment_prompt,
    context_len,
    temperature,
    top_p,
    max_tokens,
    verbose,
):
    """
    Return documents with topics assigned to them

    Parameters:
    - api_client: APIClient object
    - topics_root: TopicTree object
    - docs: list of documents
    - assignment_prompt: str
    - context_len: int
    - temperature: float
    - top_p: float
    - max_tokens: int
    - verbose: bool

    Returns:
    - res: list of responses
    """
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    tree_str = "\n".join(topics_root.to_topic_list(desc=True, count=False))
    prompted_docs, res = [], []
    prompts = []

    for i in trange(len(docs)):
        doc = docs[i]
        cos_sim = {}
        doc_emb = sbert.encode(doc, convert_to_tensor=True)

        # Include only most relevant topics such that the total length
        # of tree_str is less than max_top_len
        if api_client.estimate_token_count(tree_str) > context_len:
            for top in tree_str.split("\n"):
                top_emb = sbert.encode(top, convert_to_tensor=True)
                cos_sim[top] = util.cos_sim(top_emb, doc_emb)
            top_top = sorted(cos_sim, key=cos_sim.get, reverse=True)

            seed_len = 0
            seed_str = ""
            while seed_len < context_len and len(top_top) > 0:
                new_seed = top_top.pop(0)
                if (
                    seed_len + api_client.estimate_token_count(new_seed + "\n")
                    > context_len
                ):
                    break
                else:
                    seed_str += new_seed + "\n"
                    seed_len += api_client.estimate_token_count(seed_str)
        else:
            seed_str = tree_str

        # Truncate document if too long
        max_doc_len = (
            context_len
            - api_client.estimate_token_count(assignment_prompt)
            - api_client.estimate_token_count(seed_str)
        )
        if api_client.estimate_token_count(doc) > max_doc_len:
            print(
                f"Truncating document from {api_client.estimate_token_count(doc)} to {max_doc_len}"
            )
            doc = api_client.truncating(doc, max_doc_len)
        prompt = assignment_prompt.format(Document=doc, tree=seed_str)
        prompts.append(prompt)
        prompted_docs.append(doc)

    responses = api_client.batch_prompt(
        prompts, max_tokens, temperature, top_p, verbose
    )
    return responses, prompted_docs


def process_document_batch(
    batch_data: Dict[str, Any]
) -> Tuple[List[str], List[str]]:
    """
    Process a batch of documents in a separate thread
    
    Parameters:
    - batch_data: Dictionary containing all necessary data for processing
    
    Returns:
    - Tuple of (responses, prompted_docs)
    """
    api_client = batch_data["api_client"]
    topics_root = batch_data["topics_root"]
    docs = batch_data["docs"]
    assignment_prompt = batch_data["assignment_prompt"]
    context_len = batch_data["context_len"]
    temperature = batch_data["temperature"]
    top_p = batch_data["top_p"]
    max_tokens = batch_data["max_tokens"]
    verbose = batch_data["verbose"]
    
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    tree_str = "\n".join(topics_root.to_topic_list(desc=True, count=False))
    prompted_docs, res = [], []
    
    for doc in docs:
        cos_sim = {}
        doc_emb = sbert.encode(doc, convert_to_tensor=True)

        # Include only most relevant topics such that the total length
        # of tree_str is less than max_top_len
        if api_client.estimate_token_count(tree_str) > context_len:
            for top in tree_str.split("\n"):
                top_emb = sbert.encode(top, convert_to_tensor=True)
                cos_sim[top] = util.cos_sim(top_emb, doc_emb)
            top_top = sorted(cos_sim, key=cos_sim.get, reverse=True)

            seed_len = 0
            seed_str = ""
            while seed_len < context_len and len(top_top) > 0:
                new_seed = top_top.pop(0)
                token_count = api_client.estimate_token_count(new_seed + "\n")
                if seed_len + token_count > context_len:
                    break
                else:
                    seed_str += new_seed + "\n"
                    seed_len += token_count
        else:
            seed_str = tree_str

        # Truncate document if too long
        max_doc_len = (
            context_len
            - api_client.estimate_token_count(assignment_prompt)
            - api_client.estimate_token_count(seed_str)
        )
        if api_client.estimate_token_count(doc) > max_doc_len:
            if verbose:
                print(
                    f"Truncating document from {api_client.estimate_token_count(doc)} to {max_doc_len}"
                )
            doc = api_client.truncating(doc, max_doc_len)

        try:
            prompt = assignment_prompt.format(Document=doc, tree=seed_str)
            response = api_client.iterative_prompt(
                prompt, max_tokens, temperature, top_p=top_p, verbose=verbose
            )
            res.append(response)
        except Exception as e:
            response = "Error"
            res.append("Error")
            if verbose:
                traceback.print_exc()

        if verbose:
            print(f"Response: {response}")
            print("--------------------")
        prompted_docs.append(doc)
    
    return res, prompted_docs


def assignment_parallel(
    api_client,
    topics_root,
    docs,
    assignment_prompt,
    context_len,
    temperature,
    top_p,
    max_tokens,
    verbose,
    num_workers=4,
):
    """
    Return documents with topics assigned to them using parallel processing
    
    Parameters:
    - api_client: APIClient object
    - topics_root: TopicTree object
    - docs: list of documents
    - assignment_prompt: str
    - context_len: int
    - temperature: float
    - top_p: float
    - max_tokens: int
    - verbose: bool
    - num_workers: int, number of parallel workers
    
    Returns:
    - res: list of responses
    - prompted_docs: list of processed documents
    """
    # Determine batch size based on number of documents and workers
    batch_size = max(1, len(docs) // num_workers)
    batches = []
    
    # Create batches of documents
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i+batch_size]
        batch_data = {
            "api_client": api_client,
            "topics_root": topics_root,
            "docs": batch_docs,
            "assignment_prompt": assignment_prompt,
            "context_len": context_len,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "verbose": verbose
        }
        batches.append(batch_data)
    
    all_responses = []
    all_prompted_docs = []
    
    # Process batches in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_document_batch, batch) for batch in batches]
        
        # Show progress bar for all futures
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Processing document batches",
        ):
            try:
                responses, prompted_docs = future.result()
                all_responses.extend(responses)
                all_prompted_docs.extend(prompted_docs)
            except Exception as e:
                print(f"Error processing batch: {e}")
                traceback.print_exc()
    
    # Make sure the results are in the original document order
    # This is important since we're processing in parallel
    ordered_responses = []
    ordered_prompted_docs = []
    doc_to_result = {doc: (resp, doc) for doc, resp in zip(all_prompted_docs, all_responses)}
    
    for doc in docs:
        if doc in doc_to_result:
            resp, prompted_doc = doc_to_result[doc]
            ordered_responses.append(resp)
            ordered_prompted_docs.append(prompted_doc)
        else:
            # Handle case where a document wasn't processed
            ordered_responses.append("Error")
            ordered_prompted_docs.append(doc)
    
    return ordered_responses, ordered_prompted_docs


def assign_topics(api, model, data, prompt_file, out_file, topic_file, verbose, num_workers=1):
    """
    Assign topics to a list of documents

    Parameters:
    - api (str): API to use ('openai', 'vertex', 'vllm', 'azure', 'gemini')
    - model (str): Model to use
    - data (str): Data file
    - prompt_file (str): Prompt file
    - out_file (str): Output file
    - topic_file (str): File to write topics to
    - verbose (bool): Whether to print out results
    - num_workers (int): Number of parallel workers (default: 1)
    """
    api_client = APIClient(api=api, model=model)
    max_tokens, temperature, top_p = 1000, 0.0, 1.0

    if verbose:
        print("-------------------")
        print("Initializing topic assignment...")
        print(f"Model: {model}")
        print(f"Data file: {data}")
        print(f"Prompt file: {prompt_file}")
        print(f"Output file: {out_file}")
        print(f"Topic file: {topic_file}")
        print(f"Number of workers: {num_workers}")
        print("-------------------")

    # Model configuration
    context = (
        128000 if model.startswith("o1-mini")
        else 200000 if model.startswith(("o3-mini", "o1"))
        else 128000 if model not in ["gpt-3.5-turbo", "gpt-4"]
        else (4096 if model == "gpt-3.5-turbo" else 8000)
    )
    context_len = context - max_tokens

    # Load data ----
    df = pd.read_json(data, lines=True)
    docs = df["text"].tolist()
    assignment_prompt = open(prompt_file, "r").read()
    topics_root = TopicTree().from_topic_list(topic_file, from_file=True)

    # Prompting ----
    if api == "vllm":
        responses, prompted_docs = assignment_batch(
            api_client,
            topics_root,
            docs,
            assignment_prompt,
            context_len,
            temperature,
            top_p,
            max_tokens,
            verbose,
        )
    else:
        responses, prompted_docs = assignment_parallel(
            api_client,
            topics_root,
            docs,
            assignment_prompt,
            context_len,
            temperature,
            top_p,
            max_tokens,
            verbose,
            num_workers,
        )

    # Writing results ----
    try:
        df["prompted_docs"] = prompted_docs
        df["responses"] = responses
        df.to_json(out_file, lines=True, orient="records")
    except Exception as e:
        traceback.print_exc()
        with open(f"data/output/assignment_backup_{model}.txt", "w") as f:
            for line in responses:
                print(line, file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api",
        type=str,
        help="API to use ('openai', 'vertex', 'vllm', 'azure', 'gemini')",
        default="openai",
    )
    parser.add_argument("--model", type=str, help="Model to use", default="gpt-4")

    parser.add_argument(
        "--data",
        type=str,
        default="data/input/sample.jsonl",
        help="Data to run generation on",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="prompt/generation_1.txt",
        help="File to read prompts from",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="data/output/generation_1.jsonl",
        help="File to write results to",
    )
    parser.add_argument(
        "--topic_file",
        type=str,
        default="data/output/generation_1.md",
        help="File to write topics to",
    )

    parser.add_argument(
        "--verbose", type=bool, default=False, help="whether to print out results"
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of parallel workers"
    )

    args = parser.parse_args()
    assign_topics(
        args.api,
        args.model,
        args.data,
        args.prompt_file,
        args.out_file,
        args.topic_file,
        args.verbose,
        args.num_workers,
    )
