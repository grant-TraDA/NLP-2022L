## Instruction

All these scripts should be run on a machine with working CUDA graphics card (google colab).

#### Training and testing
```
git clone https://github.com/sdadas/polish-roberta.git
cd polish-roberta
mkdir ./data
mkdir ./data/KLEJ
mkdir ./data/KLEJ/CBD
wget https://github.com/sdadas/polish-roberta/releases/download/models-v2/roberta_large_fairseq.zip
unzip roberta_large_fairseq.zip -d ./roberta_large
pip install -r requirements.txt
```

Also you need to put prepared data to ./data/KLEJ/CBD directory (train.tsv and test_features.tsv) and overwrite tasks.py file with tasks.py provided in this repository.

Then run following command:
```
python run_tasks.py --arch roberta_large --model_dir roberta_large --train-epochs 2 --tasks KLEJ-CBD --fp16 True --max-sentences 8 --update-freq 4 --resample 0:1,1:15,2:10
```

Results should appear in polish_roberta/checkpoints/roberta_large/KLEJ/CBD in .txt files.

#### Results

Example results are in results folder.
Results are described in report in Solution section.

#### Evaluation

In order to perform solution evaluation, run script evaluate2.pl, for example:
```
evaluate2.pl klej_roberta_base-075-5-3.txt
```
