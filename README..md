# Model Overview

Model was developed for machine [translation MLops challenge](https://zindi.africa/competitions/melio-mlops-competition) to translate Dyula to French.

# Model Name

model_name: dyula_fr_translation
version: 0.01

# Model Details
Model is a neural machine translation transformer model using TensorFlow. Model was trained using [openNMT-tf](https://github.com/OpenNMT/OpenNMT-tf) toolkit.
 OpenNMT-tf-2.32.0

Tokenization and detokenization was done using sentencepiece bpe model.

# Model Date
1st September 2024

# Model Type
TensorFlow Transformer Model converted to [ctranslate2](https://opennmt.net/CTranslate2/), a library for efficient inference with Transformer models. 

# Basic Model Use

```python 
pip install ctranslate2 sentencepiece

import ctranslate2
import sentencepiece as spm

# Load the SentencePiece tokenizers
dyu_processor = spm.SentencePieceProcessor(model_file="path/to/source_tokenizer.model")
fr_processor = spm.SentencePieceProcessor(model_file="path/to/target_tokenizer.model")

# Load the CTranslate2 model
translator = ctranslate2.Translator("path/to/ctranslate2_model", device="cpu")  # Use "cuda" for GPU if available

# Example input text in source language
input_text = "Your input text here"

# Tokenize the input text
tokens = dyu_processor.encode(input_text, out_type=str)

# Translate using the CTranslate2 model
results = translator.translate_batch([tokens])

# Detokenize the output using the target tokenizer
output_tokens = results[0].hypotheses[0]
output_text = fr_processor.decode(output_tokens)

print("Translated text:", output_text)
```

# Data 

Model was trained using the following dataset.

[Parallel Dyula - French Dataset for Machine Learning](https://huggingface.co/datasets/uvci/Koumankan_mt_dyu_fr)

The training data was cleaned, duplicates removed and resplit. The train data folder used can found here:
[Train data](https://drive.google.com/drive/folders/1yDDNuBBYBji0SPaLhnxPBTSS0egKQfGl?usp=sharing)

[Training data (Kaggle Datasets)](https://www.kaggle.com/datasets/sitwala/dyu-fr-train-splits-v2/data) (Use Version 1)

[Preprocessing Notebook Used to create splits](https://colab.research.google.com/drive/1UUHhGprv__nTi7VtG0ZDS2tF5Pf2H1Ew?usp=sharing)

# Training Details

Training details and config specifics can be found in this [Colab Notebook](https://drive.google.com/file/d/1uFUCEc5m6yI_ya6IVis6ProN0V3ubj5X/view?usp=sharing)
Training was done with the following dependencies using kaggle notebook on T4 x 2 GPUs. Trainining time was approximately 3 hours.
Training Requirements:

```
tensorflow: 2.6.2
CUDA Version: 12.4 
OpenNMT-tf: 2.32.0
ctranslate2: 3.18.0
sentencepiece: 0.2.0
```

Note that the old tensorflow/cuda depedency can be difficult to obtain. The Kaggle Notebook with pinned environment has therefore been provided here:

[Kaggle Notebook with pinned environment](https://www.kaggle.com/code/sitwala/zindi-train-sub-v2/notebook?scriptVersionId=197161458) (Use this for GPU Training)

# Limitations and Risks

The trainining dataset is not enough to provide very useful translation, training should be extended as more translation data becomes available.

# HighWind Deployment

For deployment to HighWind, place the folder produced from training "saved_model" in the same path as the Dockerfile and the requirements.txt provided. 

requirements.txt
```text
kserve==0.11.2
ctranslate2==4.3.1
sentencepiece==0.2.0
```

Dockerfile
```text
FROM --platform=linux/amd64 python:3.11.8-slim
#python:3.11.8-slim

WORKDIR /app

# Dependencies
COPY ./requirements.txt .
RUN pip install --no-cache-dir  -r requirements.txt

# Trained model and definition with main script
COPY ./saved_model /app/saved_model
COPY ./main.py /app/main.py

# Set entrypoint
ENTRYPOINT ["python", "-m", "main"]
```
dcoker-compose.yml
```text
version: "3.9"
services:
  melio_predict:
    container_name: dyu_fr
    image: dyula_fr_translation_2:0.01
    command: --model_name=model
    working_dir: /app
    ports:
      - "80:8080"   
```
main.py file with inference can be found below

# Submission Zip File
Find full submission zipfile [here](https://drive.google.com/file/d/1-o9tSUPh4mUqHqouj8nIC_Ts_wlD6MWc/view?usp=drive_link)

[Saved Model Files](https://drive.google.com/drive/folders/1-oUEfVMv_mE65zPU0T1J3rDJ3xMsnOYj?usp=drive_link)
