## Requirements
- Python 3.x
- Required Python packages can be installed via `pip install -r requirements.txt`

## Datasets
To run the experiments, you need to download the following datasets:
- [Natural Questions](https://ai.google.com/research/NaturalQuestions/download)
- [Hotpot QA](https://huggingface.co/datasets/hotpotqa/hotpot_qa)
- [Commonsense QA](https://huggingface.co/datasets/tau/commonsense_qa)


### Preprocess data:
run:
- Natural Questions:
  ```bash
  python datasets/natural_questions_preprocessing.py
  ```
- Hotpot QA:
  ```bash
  python datasets/hotpot_qa_preprocessing.py
  ```
- Commonsense QA: 
  ```bash
  python datasets/commonsense_qa_preprocessing.py
  ```
## Usage
To run the experiments, use the following command:

```bash
python run_experiments.py --dataset_name=<naturalquestions/hotpotqa/commonsenseqa> --ans_model=<llama3:8b/qwen2:7b/phi3:8b/gemma2:9b> --embedding_model_name=<ada002/sbert>
```

- `--dataset_name`: Specify the dataset on which the experiments will be run (`naturalquestions`, `hotpotqa`, or `commonsenseqa`).
- `--ans_model`: Specify the language model to use for answering queries (`llama3:8b`, `qwen2:7b`, `qwen2:7b`, or `gemma2:9b`).
- `--embedding_model_name`: Specify the embedding model to use for checking similarity between the reconstructed question and the original question (`ada002` or `sbert`).

## Example
To run example use the following command:

```bash
python run_example.py --ans_model=<llama3:8b/qwen2:7b/phi3:8b/gemma2:9b> --embedding_model_name=<ada002/sbert> --reconstruction_models=<llama3:8b/qwen2:7b/phi3:8b/gemma2:9b> --iterations=<number>
```

- `--ans_model`: Specify the language model to use for answering queries (`llama3:8b`, `qwen2:7b`, `qwen2:7b`, or `gemma2:9b`).
- `--embedding_model_name`: Specify the embedding model to use for checking similarity between the reconstructed question and the original question (`ada002` or `sbert`).
- `--reconstruction_models`: Specify the language models to employ for reconstructing the query from the predicted answer. The options include permutations of llama3:8b, qwen2:7b, qwen2:7b， and gemma2:9b.
- `--iterations`: number of iterations to reconstruct the query for each model.

To examine a different query, modify the query variable along with the corresponding few-shot example in the run_example.py file.
