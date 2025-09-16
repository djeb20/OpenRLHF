import torch
import re

def reward_func(queries, prompts, labels, **kwargs):
    """
    Reward function for calculating rewards of model outputs.

    Args:
        queries (torch.Tensor): Complete text sequences containing prompts and responses
        prompts (torch.Tensor): Input prompt sequences
        labels (torch.Tensor): Ground truth answer sequences
        **kwargs: Additional optional parameters

    Returns:
        dict: A dictionary containing the following key-value pairs:
            - rewards: Reward values used for calculating advantage function
            - scores: Reward values in range [0,1] used for dynamic filtering
            - extra_logs: Additional information to be logged in wandb
    """
    print(queries) # DEBUG
    print(prompts) # DEBUG
    print(labels)  # DEBUG
    
    for query in zip(queries):

        try:
            # 1. Parse the query to compute the ground truth on-the-fly
            a, b, modulus = map(int, re.search(r"What is (\d+) \+ (\d+) mod (\d+)\?", query).groups())
            correct_answer = str((a + b) % modulus)

            # 2. Parse the response to check format and extract the model's answer
            # This regex looks for <think>...</think> followed by <answer>...</answer>
            match_response = re.search(
                r"<think>(.*?)</think>\s*<answer>(.*?)</answer>", query, re.DOTALL
            )

            if match_response:
                # Format is correct, reward is at least 0.1
                model_answer = match_response.group(2).strip()
                
                if model_answer == correct_answer:
                    # Correct format AND correct answer
                    rewards.append(1.0)
                else:
                    # Correct format, but wrong answer
                    rewards.append(0.1)
            else:
                # Format is incorrect
                rewards.append(0.0)

        except Exception:
            # If any error occurs during parsing, assign a zero reward
            print(f"Error parsing query: {query}")
            rewards.append(0.0)

        # Return the final list of rewards as a tensor on the correct device
        rewards = torch.tensor(rewards, dtype=torch.float32, device=queries.device)
    
    return {
        "rewards": rewards,  # Rewards for advantage calculation
        "scores": rewards,  # Scores for dynamic filtering (0-1 reward)
        # "extra_logs": {"dummy_scores": reward},  # Additional logging info for wandb
    }