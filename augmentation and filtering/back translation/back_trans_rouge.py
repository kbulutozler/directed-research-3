from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, pipeline
import random
from transformers import pipeline
import pandas as pd
import torch

# this file creates augmented data with back translation from the original training set and saves the resulting combined training set. 
# it also filters the augmented data with back translation with given text quality metric and saves the resulting combined training set.

rouge = pipeline("text2text-generation", 
                     task="text-generation", 
                     metric="rouge")
                     
    
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

translator_en_to_x = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es", device=device, num_beams=2, length_penalty=0.5)
translator_x_to_y = pipeline("translation", model="Helsinki-NLP/opus-mt-es-ru", device=device, num_beams=2, length_penalty=0.5)
translator_y_to_en = pipeline("translation", model="Helsinki-NLP/opus-mt-ru-en", device=device, num_beams=2, length_penalty=0.5)
for sample in dataset["train"]:
    text = sample["text"]
    label = sample["label"]
    #print(text)
    trans_text = translator_en_to_x(text)[0]['translation_text']
    inter_trans_text = translator_x_to_y(trans_text)[0]['translation_text']
    back_trans_text = translator_y_to_en(inter_trans_text)[0]['translation_text']
    #print(back_trans_text)
    data['text'].append(back_trans_text)
    data['label'].append(label)
    score = rouge(text, back_trans_text)[0]['rougeLsum']['f']
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


