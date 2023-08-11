import numpy as np
import fastwer
from matplotlib import pyplot as plt
import os

####################### Evaluation metrics ###########################

ref = 'my name is sindhura'
output = 'my name is sindhura'

out_filepath = '/home/vicky122/workspace/ga/DL/dl_project/OpticalCharacterRecognition/LARES/Good/T-40411/text_T-40411a.txt'
target_filepath = '/home/vicky122/workspace/ga/DL/dl_project/OpticalCharacterRecognition/LARES/Good/T-40411/text_T-40411a_GT.txt'

with open(out_filepath, 'r') as file:
    target_text = file.read()

with open(target_filepath, 'r') as file:
    predicted_text = file.read()

# Obtain Sentence-Level Character Error Rate (CER)
wer = fastwer.score_sent(predicted_text, target_text, char_level=True)

# Obtain Sentence-Level Word Error Rate (WER)
cer = fastwer.score_sent(predicted_text, target_text)

head, tail = os.path.split(out_filepath)
print(f'Word Error Rate for {tail}: {wer}')
print(f'Character Error Rate for {tail}: {cer}')
