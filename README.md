# Model Description

This machine translation model translates text from Dyula to French. It is built on a `fairseq` model architecture proposed by [Facebook](https://fairseq.readthedocs.io/en/latest/). The architecture was replicated using a Dyula-French translation dataset created by data354. The model was later quantized into int8 and exported to [ctranslate2](https://opennmt.net/CTranslate2/) format for fast inference. The model is designed to support a variety of educational applications by providing accurate and contextually relevant translations between these languages.

## Intended Use

The model is specifically designed to support **AI Student Learning Assistant (AISLA)**, a free educational tool aimed at helping students learn and communicate in their native language.

The model is particularly valuable for enhancing educational accessibility for Dyula-speaking students by enabling reliable translations from Dyula to French. It is intended to be integrated into platforms like Discord to provide seamless support within educational environments.

# Deployment

This folder contains the resources required for deploying the trained model onto Highwind.

## Usage

> All commands below are run from the deployment directory.

### Building the Model Image

This step builds the Kserve predictor image that contains the model.

1. Ensure the trained model folder `model_dir` contains:
    - `model.bin` (the model itself)
    - `config.json` (model configuration file)
    - `combined_model_2000.model` (sentence piece model for tokenization)
    - `shared_vocabulary.json` (shared vocabulary)

2. The deployment folder should include:
    - `Dockerfile` (steps to build the image)
    - `main.py` (starts the Kserve server and runs the model for translation inference tasks)
    - `serve-requirements.txt` (model dependencies)

3. Build the container locally without caching and tag it:
    ```bash
    docker build --no-cache -t dyula-french-seqf_8-25-5beams:latest .
    ```

### Local Testing

1. After building the Kserve predictor image, spin it up to test the model inference:
    ```bash
    docker compose up -d
    docker compose logs
    ```

2. Send a payload to the model to test its response using the `curl` command to send a `POST` request with an example JSON payload.

    > Run this from another terminal (remember to navigate to this folder first)

    Linux/Mac Bash/zsh:
    ```bash
    curl -X POST http://localhost:8080/v2/models/model/infer -H 'Content-Type: application/json' -d @./input.json
    ```

    Windows PowerShell:
    ```PowerShell
    $json = Get-Content -Raw -Path ./input.json
    $response = Invoke-WebRequest -Uri http://localhost:8080/v2/models/model/infer -Method Post -ContentType 'application/json' -Body ([System.Text.Encoding]::UTF8.GetBytes($json))
    $responseObject = $response.Content | ConvertFrom-Json
    $responseObject | ConvertTo-Json -Depth 10
    ```

    Using Postman:
    Send a `POST` request to `http://localhost:8080/v2/models/model/infer` with a JSON body:
    ```json
    {
        "inputs": [
            {
                "name": "input-0",
                "shape": [1],
                "datatype": "BYTES",
                "parameters": null,
                "data": ["i tɔgɔ bi cogodɔ"]
            }
        ]
    }
    ```

### Deployment on Highwind

Once the translation is successfully generated through a POST request, proceed to deploy the model on Highwind:

1. Create a new asset in Highwind and follow the push commands.
2. Create a new use case and add the asset to it.
3. Deploy the use case.

Now, the model is ready for inference through API calls!
