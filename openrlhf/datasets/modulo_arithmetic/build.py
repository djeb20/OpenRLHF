"""
Contains functions for building and preparing datasets for PPO training.
"""

import numpy as np
from typing import Dict, List
from datasets import Dataset

PROMPT_TEMPLATE = """
You are an expert mathematician. 

Your question: {question}

First, reason through the steps to solve the question. This reasoning process MUST be enclosed within <think> </think> tags.
Then, state ONLY your final numeric answer. This answer MUST be enclosed within <answer> </answer> tags.
"""

def build_modulo_dataset(
        train_size: int,
        a_limit: int = 10, 
        b_limit: int = 10, 
        modulus: int = 10
        ) -> Dataset:
    """
    Procedurally generates the modulo arithmetic dataset.

    Args:
        modulus (int): The modulus for the arithmetic operation.
        train_size (int): The number of samples to generate.

    Returns:
        Dataset: A Hugging Face Dataset object with 'query' and 'answer' columns.
    """
    data: List[Dict[str, str]] = []

    # TODO: Shouldn't be random, need unique datapoints for eval set

    # First generate the answers
    a_t = np.random.randint(0, a_limit, (train_size,))
    b_t = np.random.randint(0, b_limit, (train_size,))
    results = (a_t + b_t) % modulus

    for i in range(train_size):

        # Create the core message with the questions and instructions
        question = f"What is {a_t[i]} + {b_t[i]} mod {modulus}?"
        full_user_prompt = PROMPT_TEMPLATE.format(question=question)
        data.append({"prompt": full_user_prompt, "answer": str(results[i])})

    return Dataset.from_list(data)