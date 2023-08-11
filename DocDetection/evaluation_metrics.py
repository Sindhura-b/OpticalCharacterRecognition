import numpy as np
import fastwer
from matplotlib import pyplot as plt
import os

####################### Evaluation metrics ###########################

ref = 'my name is sindhura'
output = 'my name is sindhura'

out_filepath = '/home/vicky122/workspace/ga/DL/dl_project/OpticalCharacterRecognition/LARES/Good/T-40411/text_T-40411a.txt'
target_filepath = '/home/vicky122/workspace/ga/DL/dl_project/OpticalCharacterRecognition/LARES/Good/T-40411/text_T-40411a_GT.txt'

file_dir = '/home/vicky122/workspace/ga/DL/dl_project/OpticalCharacterRecognition/LARES/Good/T-40411_output/sub/'
files = os.listdir(file_dir)
GT_filename = []
doctr_filename = []
AWS_filename = []
tesseract_filename = []
wer_doctr=[]
wer_AWS=[]
wer_tesseract=[]
cer_doctr=[]
cer_AWS=[]
cer_tesseract=[]

for filename in files:
    # if ground truth files exist
    if '_GT' in filename:
        GT_filename.append(filename)
        doctr_filename.append(filename.replace('_GT',''))
        AWS_filename.append(filename.replace('_GT', '_AWS'))
        tesseract_filename.append(filename.replace('_GT','_TSRCT'))

for i in range(len(doctr_filename)):

    with open(os.path.join(file_dir, GT_filename[i]), 'r') as file:
        target_text = file.read()

    with open(os.path.join(file_dir, doctr_filename[i]), 'r') as file:
        predicted_text_doctr = file.read()

    with open(os.path.join(file_dir, AWS_filename[i]), 'r') as file:
        predicted_text_AWS = file.read()

    with open(os.path.join(file_dir, tesseract_filename[i]), 'r') as file:
        predicted_text_tesseract = file.read()

    # Obtain Sentence-Level Character Error Rate (CER)
    wer_doctr.append(fastwer.score_sent(predicted_text_doctr, target_text, char_level=True))
    wer_AWS.append(fastwer.score_sent(predicted_text_AWS, target_text, char_level=True))
    wer_tesseract.append(fastwer.score_sent(predicted_text_tesseract, target_text, char_level=True))

    print(f'Word Error Rate for {doctr_filename[i]}: doctr: {wer_doctr[i]}, AWS: {wer_AWS[i]}, tesseract: {wer_tesseract[i]}')
    # Obtain Sentence-Level Word Error Rate (WER)
    cer_doctr.append(fastwer.score_sent(predicted_text_doctr, target_text))
    cer_AWS.append(fastwer.score_sent(predicted_text_AWS, target_text))
    cer_tesseract.append(fastwer.score_sent(predicted_text_tesseract, target_text))

    print(f'Character Error Rate for {doctr_filename[i]}: doctr: {cer_doctr[i]}, AWS: {cer_AWS[i]}, tesseract: {cer_tesseract[i]}')

# head, tail = os.path.split(out_filepath)
print(f'Average Word Error Rate for doctr: {np.mean(wer_doctr)}, AWS: {np.mean(wer_AWS)}, Tesseract: {np.mean(wer_tesseract)}')
print(f'Average Character Error Rate for doctr: {np.mean(cer_doctr)}, AWS: {np.mean(cer_AWS)}, Tesseract: {np.mean(cer_tesseract)}')


################### Output for 1 file ##################
# Word Error Rate for text_T-40411a.txt: doctr: 15.2632, AWS: 12.5101, tesseract: 36.5992
# Character Error Rate for text_T-40411a.txt: doctr: 30.75, AWS: 21.0, tesseract: 69.25
# Average Word Error Rate for doctr: 15.2632, AWS: 12.5101, Tesseract: 36.5992
# Average Character Error Rate for doctr: 30.75, AWS: 21.0, Tesseract: 69.25