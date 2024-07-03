from .dependency import *

APP = typer.Typer()

PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
DATASET_DIR = os.path.join(PROJECT_DIR, "datasets")
MODEL_SAVE_DIR = os.path.join(PROJECT_DIR, "checkpoints")
MODEL_NAME = "microsoft/layoutlm-base-uncased"
TOKENIZER = LayoutLMTokenizer.from_pretrained(MODEL_NAME)
RESULT_DIR = os.path.join(PROJECT_DIR, "results")
