import os
import time
import random
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv


# Load variables from .env file into the environment
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load original SP data
data = np.load("data/SP-train.npy", allow_pickle=True)
final_entries = []

# Helper: generate question from answer
def create_generated_entry(base_id, answer, generated_question, distractors, suffix="_GEN"):
    all_choices = [answer] + distractors
    choice_list = all_choices.copy()
    random.shuffle(choice_list)
    label = choice_list.index(answer)
    choice_order = [all_choices.index(c) for c in choice_list]

    return {
        "id": base_id + suffix,
        "question": generated_question,
        "answer": answer,
        "distractor1": distractors[0],
        "distractor2": distractors[1],
        "distractor(unsure)": distractors[2],
        "label": label,
        "choice_list": choice_list,
        "choice_order": choice_order
    }

for i, entry in enumerate(data):
    try:
        answer = entry["answer"]
        base_id = entry["id"].split("_")[0]

        prompt = f"""Create a Sentence Puzzle-style question where the correct answer is "{answer}".
The question should be clever, indirect, and require reasoning.
Do NOT include the answer in the question. 

Question:"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=150
        )

        generated_question = response.choices[0].message.content.strip()

        distractors = [entry["distractor1"], entry["distractor2"], entry["distractor(unsure)"]]
        gen_entry = create_generated_entry(base_id, answer, generated_question, distractors)

        final_entries.append(entry)  # original
        final_entries.append(gen_entry)  # synthetic
        print(f"[{i}] Created: {gen_entry['id']}")
        time.sleep(1)

    except Exception as e:
        print(f"[{i}] Error: {e}")
        continue

np.save("data/SP-train-with-generated.npy", final_entries)
print("Saved SP-train-with-generated.npy with synthetic questions.")
