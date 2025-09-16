"""
Simple script to test build_modulo_dataset in openrlhf.datasets.modulo_arithmetic
Run: python tests/test_build_modulo_dataset.py
"""
from openrlhf.datasets.modulo_arithmetic import build_modulo_dataset

def main():
    ds = build_modulo_dataset(train_size=5, a_limit=10, b_limit=10, modulus=10)
    print(f"Dataset length: {len(ds)}")
    for i, item in enumerate(ds):
        print(f"Sample {i}: {item}")
        if 'query' not in item or 'answer' not in item:
            print("FAIL: Missing 'query' or 'answer' field.")
            return
        q = item['query']
        a = int(q.split()[2])
        b = int(q.split()[4])
        mod = int(q.split()[-1][:-1] if q.split()[-1].endswith('?') else q.split()[-1])
        expected = str((a + b) % mod)
        if item['answer'] != expected:
            print(f"FAIL: Incorrect answer for {q}. Got {item['answer']}, expected {expected}.")
            return
    print("All tests passed!")

if __name__ == "__main__":
    main()
