from transformers import BertModel, BertTokenizer
import torch
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
model.eval()
embedding_file = './glove.6B.200d.txt'
new_embedding_file = './bert_768d.txt'
word_l = []
with open(embedding_file, 'r',encoding='utf-8') as file:
    lines = file.readlines()
    for line in tqdm(lines, desc='bert'):
        splits = line.split()
        word = splits[0]
        word_l.append(word)

with open(new_embedding_file, 'w',encoding='utf-8') as file:
    for word in word_l:
        with torch.no_grad():
            inputs = tokenizer(word, return_tensors="pt")
            outputs = model(**inputs)

            last_hidden_states = outputs.last_hidden_state
            vector = last_hidden_states[0][1].detach().numpy()
            file.write(str(word) + " ")
            for i in vector:
                file.write(str(i) + " ")
            file.write('\n')
