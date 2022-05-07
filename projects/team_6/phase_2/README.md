## Instruction

All these scripts should be run on a machine with working CUDA graphics card.

#### Training and testing
```
git clone https://github.com/sdadas/polish-roberta.git
git clone https://github.com/erfactor/NLP-2022L.git
cd polish-roberta
cp ../NLP-2022L/projects/team_6/phase_2/tasks.py
mkdir ./data
mkdir ./data/KLEJ
mkdir ./data/KLEJ/CBD
cp -r ../NLP-2022L/data ./data/KLEJ/CBD
wget https://github.com/sdadas/polish-roberta/releases/download/models-v2/roberta_large_fairseq.zip
unzip roberta_large_fairseq.zip -d ./roberta_large
pip install -r requirements.txt
python run_tasks.py --arch roberta_large --model_dir roberta_large --train-epochs 2 --tasks KLEJ-CBD --fp16 True --max-sentences 8 --update-freq 4 --resample 0:1,1:15,2:10
```

Results should appear in polish_roberta/checkpoints/roberta_large/KLEJ/CBD in .txt files.
Example results are in results folder.

#### Evaluation

In order to perform solution evaluation, run script evaluate2.pl:
```
evaluate2.pl klej_roberta_base-075-5-3.txt
```
