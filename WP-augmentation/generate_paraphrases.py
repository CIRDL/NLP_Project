import os
import time
import random
import numpy as np
from openai import OpenAI, OpenAIError

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load original data
data = np.load("Baseline_Code_Repo/data/WP-train.npy", allow_pickle=True)
final_entries = []

# Helper function to create the Paraphrased entry
def create_paraphrased_entry(base_entry, paraphrased_question, new_id_suffix="_PARA"):
    answer = base_entry["answer"]
    distractors = [
        base_entry["distractor1"],
        base_entry["distractor2"],
        base_entry["distractor(unsure)"]
    ]
    all_choices = [answer] + distractors
    choice_list = all_choices.copy()
    random.shuffle(choice_list)
    label = choice_list.index(answer)
    choice_order = [all_choices.index(c) for c in choice_list]

    return {
        "id": base_entry["id"].split("_")[0] + new_id_suffix,
        "question": paraphrased_question,
        "answer": answer,
        "distractor1": base_entry["distractor1"],
        "distractor2": base_entry["distractor2"],
        "distractor(unsure)": base_entry["distractor(unsure)"],
        "label": label,
        "choice_list": choice_list,
        "choice_order": choice_order
    }

# Loop through in chunks of 3 (OG, SR, CR)
for i in range(0, len(data), 3):
    try:
        og = data[i]
        sr = data[i+1]
        cr = data[i+2]

        # Confirm OG is clean
        if "_SR" in og["id"] or "_CR" in og["id"]:
            continue

        original_question = og["question"]
        correct_answer = og["answer"]

        # Prompt OpenAI to paraphrase
        prompt = (
            f"Paraphrase the following Word Puzzle question while preserving its meaning and logic.\n"
            f"Original: {original_question}\n"
            f"Correct Answer: {correct_answer}\n\n"
            f"Make sure your paraphrased question is NOT similar to these versions:\n"
            f"- SR: {sr['question']}\n"
            f"- CR: {cr['question']}\n\n"
            f"Paraphrased Question:"
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )

        paraphrased_question = response.choices[0].message.content.strip()
        para_entry = create_paraphrased_entry(base_entry=og, paraphrased_question=paraphrased_question)

        # Add OG, SR, CR, and new PARA
        final_entries.extend([og, sr, cr, para_entry])
        print(f"[{i//3}] Augmented: {og['id']}")

        time.sleep(1)

    except Exception as e:
        print(f"[{i}] Error: {e}")
        continue

# Save result
np.save("WP-augmentation/data/WP-train-with-para.npy", final_entries)
print("Saved WP-train-with-para.npy with paraphrased entries.")
