## Model Description

This machine translation model translates text from Dyula to French. It is built on a `fairseq` model architecture proposed by Facebook. The architecture was replicated using a Dyula-French translation dataset created by [data354](https://data354.com/en/). The model was later quantized into int8 and exported to ctranslate2 format for fast inference frameweork. The model is designed to support a variety of educational applications by providing accurate and contextually relevant translations between these languages.

## Intended Use

The model is specifically designed to support **AI Student Learning Assistant (AISLA)**, a free educational tool aimed at helping students learn and communicate in their native language.

The model is particularly valuable for enhancing educational accessibility for Dyula-speaking students by enabling reliable translations from Dyula to French. It is intended to be integrated into platforms like Discord to provide seamless support within educational environments.

## Example Payload

To test the model's inference capabilities, you can use the following example payload. This JSON payload includes a Dyula sentence for translation:

```json
{
    "inputs": [
        {
            "name": "input-0",
            "shape": [1],
            "datatype": "BYTES",
            "parameters": null,
            "data": [
                "i tɔgɔ bi cogodɔ"
            ]
        }
    ]
}
