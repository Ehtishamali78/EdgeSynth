# scripts/prepare_prompts.py
import pandas as pd
from transformers import pipeline, set_seed
from pathlib import Path

BATCH_SIZE = 16  # bump to 32/64 if you have GPU VRAM

# ---- Paths (relative to this file) ----
HERE = Path(__file__).resolve().parent
DATA_DIR = (HERE / ".." / "data").resolve()
PROMPTS_PATH = DATA_DIR / "batched_prompts.csv"
RAW_OUT_PATH = DATA_DIR / "synthetic_raw.txt"  # aligns with postprocess.py

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not PROMPTS_PATH.exists():
        raise FileNotFoundError(f"Missing prompts file: {PROMPTS_PATH}")

    df = pd.read_csv(PROMPTS_PATH, header=None)
    prompts = df[0].tolist()
    if not prompts:
        raise RuntimeError("No prompts found in batched_prompts.csv")

    print("Loading GPT-2…")
    # device: -1=CPU, 0=first GPU
    gen = pipeline("text-generation", model="gpt2", device=-1)
    set_seed(42)

    sampling_kwargs = dict(
        max_new_tokens=120,
        num_return_sequences=1,
        temperature=0.95,       # 0.9–1.05: lower = tidier, higher = more variety
        top_p=0.95,
        top_k=100,
        repetition_penalty=1.1,
        truncation=True,
        pad_token_id=50256,
    )

    with RAW_OUT_PATH.open("w", encoding="utf-8") as f:
        for i in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[i:i+BATCH_SIZE]
            outputs = gen(batch, **sampling_kwargs)
            # outputs is list of lists (one list per prompt)
            for prompt_outputs in outputs:
                for out in prompt_outputs:
                    f.write(out["generated_text"].strip() + "\n---\n")
            print(f"✅ Processed {i+1}–{i+len(batch)}")

    print(f"\n🎉 Saved raw outputs → {RAW_OUT_PATH}")

if __name__ == "__main__":
    main()
