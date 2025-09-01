import os
import pandas as pd
from transformers import pipeline, set_seed

BATCH_SIZE   = 16                 # bump to 32/64 if you have GPU VRAM
PROMPTS_PATH = "../data/batched_prompts.csv"
RAW_OUT_PATH = "../data/gpt2_raw_output_week6.txt"

def main():
    os.makedirs("data", exist_ok=True)
    df = pd.read_csv(PROMPTS_PATH, header=None)
    prompts = df[0].tolist()

    print("Loading GPT-2…")
    gen = pipeline("text-generation", model="gpt2", device=-1)  # -1 CPU, 0 GPU
    set_seed(42)

    # NOTE: balanced settings: diverse but still log-like
    sampling_kwargs = dict(
        max_new_tokens=120,
        num_return_sequences=1,
        temperature=0.95,      # 0.9–1.05: lower = tidier, higher = more variety
        top_p=0.95,
        top_k=100,
        repetition_penalty=1.1,  # discourage verbatim repeats
        truncation=True,
        pad_token_id=50256,
    )

    with open(RAW_OUT_PATH, "w", encoding="utf-8") as f:
        for i in range(0, len(prompts), BATCH_SIZE):
            batch = prompts[i:i+BATCH_SIZE]
            outputs = gen(batch, **sampling_kwargs)

            # outputs is list-of-lists (one list per prompt)
            for prompt_outputs in outputs:
                for out in prompt_outputs:
                    f.write(out["generated_text"].strip() + "\n---\n")

            print(f"✅ Processed {i+1}–{i+len(batch)}")

    print(f"\n🎉 Saved raw outputs → {RAW_OUT_PATH}")

if __name__ == "__main__":
    main()