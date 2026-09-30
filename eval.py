import argparse

from eval_lib import run_standalone_eval


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model on a dataset")
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=[
            "tooluse",
            "science",
            "math_contradiction",
            "spatial_contradiction2",
            "spatial_standard2",
        ],
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_model_len", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    summary = run_standalone_eval(
        dataset_name=args.dataset_name,
        model_path=args.model_path,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print(f"  Dataset: {args.dataset_name}")
    print(f"  Total samples: {summary['num_total']}")
    print(f"  Correct: {summary['num_correct']}")
    print(f"  Accuracy: {summary['accuracy']:.4f} ({summary['accuracy'] * 100:.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
