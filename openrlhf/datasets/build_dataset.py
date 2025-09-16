import argparse

known_datasets = ["modulo_arithmetic"]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default="modulo_arithmetic", help="Name of the dataset to load or path to custom dataset script.")
    parser.add_argument("--train_size", type=int, default=10000, help="Number of training samples to generate.")

    # File paths
    parser.add_argument("--train_save_path", type=str, default="./modulo_data/train", help="Path to save the training dataset.")
    parser.add_argument("--eval_save_path", type=str, default=None, help="Path to save the evaluation dataset.")
    parser.add_argument("--eval_size", type=float, default=0.2, help="Number of evaluation samples to generate.")

    # Modulo arithmetic specific args
    parser.add_argument("--a_limit", type=int, default=10, help="Upper limit for the first addend in modulo arithmetic questions.")
    parser.add_argument("--b_limit", type=int, default=10, help="Upper limit for the second addend in modulo arithmetic questions.")
    parser.add_argument("--modulus", type=int, default=10, help="Modulus for the modulo arithmetic questions.")

    args = parser.parse_args()

    print("Generating the full dataset...")
    if args.dataset_name == "modulo_arithmetic":
        from openrlhf.datasets.modulo_arithmetic import build_modulo_dataset
        full_dataset = build_modulo_dataset(
            train_size=args.train_size,
            a_limit=args.a_limit,
            b_limit=args.b_limit,
            modulus=args.modulus
        )
    else:
        raise NotImplementedError(f"Dataset {args.dataset_name} is not implemented.")

    if args.eval_save_path:
        print("Splitting into train and eval sets...")
        train_dataset = full_dataset.select(range(int(args.train_size * (1 - args.eval_size))))
        eval_dataset = full_dataset.select(range(int(args.train_size * (1 - args.eval_size)), args.train_size))
    
    else:
        train_dataset = full_dataset

    print(f"Saving training dataset to {args.train_save_path}...")
    train_dataset.save_to_disk(args.train_save_path)

    if args.eval_save_path:
        print(f"Saving evaluation dataset to {args.eval_save_path}...")
        eval_dataset.save_to_disk(args.eval_save_path)
        
    print("✅ Datasets created successfully!")
    

