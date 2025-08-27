# F2Attack

This is the implementation code of the paper *F2Attack: Two-Factors Scoring Method For Query-Efficient Hard-Label Black-Box Textual Adversarial Attacks*

## Requirements
•	Pytorch = 1.10.1
•	Tensorflow-gpu=2.6.0
•	Numpy=1.22.4
•	Python = 3.9.0
•	Tensorflow 2.1.0
•	Tensorflow-Hub=0.13.0

## Download Dependencies
•	Download the counter-fitted-vectors from https://drive.google.com/file/d/1bayGomljWb6HeYDMTDKXrh0HackKtSlx/view .

•	Download mat.txt from https://drive.google.com/file/d/1AIz8Imvv8OmHxVwY5kx10iwKAUzD6ODx/view.

•	Download the glove 200 dimensional vectors from https://nlp.stanford.edu/projects/glove/ unzip it.

## Generate Bert Embdding 
•	Run train_tgt_cnn_models.py 

## How to Run:
## implementation Instruction
We take the target model MR_wordCNN as an example and give the running instructions and specific sequence of the algorithm in this paper
•	Train joint attention model F_attn, run train_joint_attn_model.py
   follow the instruction: python train_joint_attn_model.py -dataset_name imdb -max_epoch 100 -lr 1e-3
   
•	Train target_models, run train_tgt_cnn_models.py 
   follow the instruction: python train_tgt_cnn_models.py  --cnn --dataset mr --max_epoch 70 --lr 1e-3
   
•	F2Attack.py is our attack file
   follow the instruction: python F2Attack.py  -model_type wordCNN -target_dataset_name mr -data_size 500 -allowed_qrs 100 -synonym_num 5




 



