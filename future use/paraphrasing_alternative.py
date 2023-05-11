import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
     

dataset = load_dataset('tapaco', 'en')

def process_tapaco_dataset(dataset, out_file):
    tapaco = []
    # The dataset has only train split.
    for data in tqdm(dataset["train"]):
        keys = data.keys()
        tapaco.append([data[key] for key in keys])
    tapaco_df = pd.DataFrame(
        data=tapaco,
        columns=[
            "language",
            "lists",
            "paraphrase",
            "paraphrase_set_id",
            "sentence_id",
            "tags",
        ],
    )
    tapaco_df.to_csv(out_file, sep="\t", index=None)
    return tapaco_df
     

tapaco_df = process_tapaco_dataset(dataset,"tapaco_huggingface.csv")
     

tapaco_df.head()
     


     

tapaco_df = pd.read_csv("tapaco_huggingface.csv",sep="\t")
     

def generate_tapaco_paraphrase_dataset(dataset, out_file):
    dataset_df = dataset[["paraphrase", "paraphrase_set_id"]]
    non_single_labels = (
        dataset_df["paraphrase_set_id"]
        .value_counts()[dataset_df["paraphrase_set_id"].value_counts() > 1]
        .index.tolist()
    )
    tapaco_df_sorted = dataset_df.loc[
        dataset_df["paraphrase_set_id"].isin(non_single_labels)
    ]
    tapaco_paraphrases_dataset = []

    for paraphrase_set_id in tqdm(tapaco_df_sorted["paraphrase_set_id"].unique()):
        id_wise_paraphrases = tapaco_df_sorted[
            tapaco_df_sorted["paraphrase_set_id"] == paraphrase_set_id
        ]
        len_id_wise_paraphrases = (
            id_wise_paraphrases.shape[0]
            if id_wise_paraphrases.shape[0] % 2 == 0
            else id_wise_paraphrases.shape[0] - 1
        )
        for ix in range(0, len_id_wise_paraphrases, 2):
            current_phrase = id_wise_paraphrases.iloc[ix][0]
            for count_ix in range(ix + 1, ix + 2):
                next_phrase = id_wise_paraphrases.iloc[ix + 1][0]
                tapaco_paraphrases_dataset.append([current_phrase, next_phrase])
    tapaco_paraphrases_dataset_df = pd.DataFrame(
        tapaco_paraphrases_dataset, columns=["Text", "Paraphrase"]
    )
    tapaco_paraphrases_dataset_df.to_csv(out_file, sep="\t", index=None)
    return tapaco_paraphrases_dataset_df
     

dataset_df = generate_tapaco_paraphrase_dataset(tapaco_df,"tapaco_paraphrases_dataset.csv")
     

dataset_df.head()
     

# !wget https://github.com/hetpandya/paraphrase-datasets-pretrained-models/raw/main/datasets/tapaco/tapaco_paraphrases_dataset.csv
     

dataset_df = pd.read_csv("tapaco_paraphrases_dataset.csv",sep="\t")
     

from simpletransformers.t5 import T5Model
from sklearn.model_selection import train_test_split
import sklearn
     

dataset_df.columns = ["input_text","target_text"]
dataset_df["prefix"] = "paraphrase"
     

train_data,test_data = train_test_split(dataset_df,test_size=0.1)
     




args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 256,
    "num_train_epochs": 4,
    "num_beams": None,
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.95,
    "use_multiprocessing": False,
    "save_steps": -1,
    "save_eval_checkpoints": True,
    "evaluate_during_training": False,
    'adam_epsilon': 1e-08,
    'eval_batch_size': 6,
    'fp_16': False,
    'gradient_accumulation_steps': 16,
    'learning_rate': 0.0003,
    'max_grad_norm': 1.0,
    'n_gpu': 1,
    'seed': 42,
    'train_batch_size': 6,
    'warmup_steps': 0,
    'weight_decay': 0.0
}
     

model = T5Model("t5","t5-small", args=args)

model.train_model(train_data, eval_data=test_data, use_cuda=True,acc=sklearn.metrics.accuracy_score)



root_dir = os.getcwd()
trained_model_path = os.path.join(root_dir,"outputs")
     

args = {
    "overwrite_output_dir": True,
    "max_seq_length": 256,
    "max_length": 50,
    "top_k": 50,
    "top_p": 0.95,
    "num_return_sequences": 5,
}
     

trained_model = T5Model("t5",trained_model_path,args=args)
     

prefix = "paraphrase"
pred = trained_model.predict([f"{prefix}: The house will be cleaned by me every Saturday."])
pprint(pred)