"""Kserve inference script."""
import re
import argparse
from typing import List
import os
import requests
import ctranslate2
import sentencepiece as spm
from kserve import (model_server, Model, ModelServer, InferRequest, InferOutput, InferResponse)
from kserve.utils.utils import generate_uuid

os.environ["CT2_USE_EXPERIMENTAL_PACKED_GEMM"] = "1"
CLEANING_REGEX = re.compile(r'[!"&\(\),-./:;=?+.\[\]«»]')

class CTranslate2Model:
    """
    CTranslate2Model which loads CTranslate2 model for inference.
    """
    # pylint: disable=too-few-public-methods
    def __init__(self, model_dir: str, sp_model_path: str) -> None:
        self.translator = ctranslate2.Translator(model_dir)
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(sp_model_path)

    def translate(self, sentence: str) -> str:
        """
        Translate the given sentence.
        :param sentence: Sentence to be translated
        :return: Translated sentence.
        """
        prepared_data = clean_text(sentence)
        tokens = self.sp.Encode(prepared_data, out_type=str)
        translation = self.translator.translate_batch([tokens], beam_size=1,
                                                      return_scores=False, disable_unk=True)
        translated_sentence = self.sp.Decode(translation[0].hypotheses[0])
        return translated_sentence

class MyModel(Model):
    """
    MyModel class for KServe inference.
    """
    def __init__(self, name: str, model_dir: str, sp_model_path: str):
        # pylint: disable=too-few-public-methods
        super().__init__(name)
        self.name = name
        self.model = None
        self.model_dir = model_dir
        self.sp_model_path = sp_model_path
        self.ready = False
        self.load()

    def load(self):
        self.model = CTranslate2Model(model_dir=self.model_dir, sp_model_path=self.sp_model_path)
        self.ready = True

    async def preprocess(self, payload: InferRequest, *_args, **_kwargs) -> List[str]:
        infer_inputs: List[str] = payload.inputs[0].data
        cleaned_texts: List[str] = [clean_text(i) for i in infer_inputs]
        return cleaned_texts
    # pylint: disable=arguments-differ
    async def predict(self, data: List[str], *_predict_args, **_kwargs) -> InferResponse:
        response_id = generate_uuid()
        results = [self.model.translate(sentence) for sentence in data]
        infer_output = InferOutput(
            name="output-0", shape=[len(results)], datatype="STR", data=results
        )
        infer_response = InferResponse(
            model_name=self.name, infer_outputs=[infer_output], response_id=response_id
        )
        return infer_response

def clean_text(text: str) -> str:
    """
    Clean input text by removing special characters and converting to lower case.
    """
    text = text.lower()
    #text = re.sub(r'\.{2,}', '', text)
    #text = re.sub(r'\s+', ' ', text)
    #text = text.replace("’", "'").replace('—', '').replace('…', '')
    #text = re.sub(r'[!"&\(\),-./:;=?+.\[\]«»]', ' ', text)
    text = CLEANING_REGEX.sub(' ', text)  # Use the pre-compiled regex
    return text.strip()

def warm_up_model(model_server_address: str):
    """
    Warm-up the model by sending a dummy request.
    """
    dummy_data = ["This is a warm-up request."]
    try:
        response = requests.post(f"http://{model_server_address}",
                                 json={"inputs": [{"data": dummy_data}]}, timeout=10)
        if response.status_code == 200:
            print("Model warm-up successful.")
        else:
            print("Model warm-up failed with status code:", response.status_code)
    except requests.exceptions.RequestException as e:
        print("Model warm-up failed due to request exception:", e)

parser = argparse.ArgumentParser(parents=[model_server.parser],
                                 conflict_handler='resolve')
parser.add_argument("--model_name", default="model",
                    help="The name that the model is served under.")
parser.add_argument("--model_dir", default="/app/model_dir",
                    help="Directory where the model is stored.")
parser.add_argument("--sp_model_path", default="/app/model_dir/combined_model_2000.model",
                    help="Path to the SentencePiece model.")
parser.add_argument("--model_server_address", default="localhost:8080",
                    help="Address of the model server.")
parsed_args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = MyModel(parsed_args.model_name, parsed_args.model_dir, parsed_args.sp_model_path)
    ModelServer().start([model])
    warm_up_model(parsed_args.model_server_address)
