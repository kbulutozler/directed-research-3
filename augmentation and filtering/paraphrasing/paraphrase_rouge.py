from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, pipeline
import random
import pandas as pd
import torch

# this file creates augmented data with paraphrasing from the original training set and saves the resulting combined training set. 
# it also filters the augmented data with paraphrasing with given text quality metric and saves the resulting combined training set.

rouge = pipeline("text2text-generation", 
                     task="text-generation", 
                     metric="rouge")
                     

model = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality")
tokenizer = AutoTokenizer.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality")

def get_paraphrase(context):
    text = "paraphrase: "+context + " </s>"

    encoding = tokenizer.encode_plus(text,max_length =128, padding=True, return_tensors="pt")
    input_ids,attention_mask  = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
    model.eval()
    diverse_beam_outputs = model.generate(
        input_ids=input_ids,attention_mask=attention_mask,
        max_length=128,
        early_stopping=True,
        num_beams=5,
        num_beam_groups = 5,
        num_return_sequences=1,
        diversity_penalty = 0.70

    )
    for beam_output in diverse_beam_outputs:
        paraphrase = tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return paraphrase

dataset = load_dataset("tweet_eval", "abortion")
"""
you can use any of the following config names as a second argument:
"emoji", "emotion", "hate", "irony", 
"offensive", "sentiment", "stance_abortion", "stance_atheism", 
"stance_climate", "stance_feminist", "stance_hillary"
"""
data = {
    'text': [],
    'label': [],
}
device = torch.cuda.current_device()
model = model.to(device)

for sample in dataset["train"]:
    text = sample["text"]
    label = sample["label"]

    paraphrased = get_paraphrase(text)
    data['text'].append(paraphrased)
    data['label'].append(label)
    score = rouge(text, paraphrased)[0]['rougeLsum']['f']
    scores.append(score)
mean_score = sum(scores) / len(scores)

df = pd.DataFrame(data)

# save the DataFrame to a CSV file
df.to_csv('./synthetic_data/tweet_eval_train_abortion.csv', index=False)


filtered_df = df.copy()

for sample in dataset["train"]:
    text = sample["text"]
    label = sample["label"]
    for i, (text2, label2) in df[['text', 'label']].iterrows():
        score = rouge(text, text2)[0]['rougeLsum']['f']
        if score < mean_score:
            filtered_df = filtered_df.drop(i)
            break

filtered_df.to_csv('./synthetic_data/tweet_eval_train_abortion_filtered.csv', index=False)


