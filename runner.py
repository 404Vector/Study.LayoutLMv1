from study_llmv1.dependency import *
from study_llmv1.const import *
import study_llmv1 as llmv1


@APP.command()
def train(
    target_dataset_name: str = "kaggle.document-classification-d",
    test_size: float = 0.09,
    num_epochs: int = 3,
    lr: float = 0.00004,
    seed: int = 0,
):
    llmv1.train(
        target_dataset_name=target_dataset_name,
        test_size=test_size,
        num_epochs=num_epochs,
        lr=lr,
        seed=seed,
    )


@APP.command()
def test(
    path_like_pattern: str,
    target_dataset_name: str = "kaggle.document-classification-dataset",
    batch_size: int = 4,
):
    llmv1.test(
        path_like_pattern=path_like_pattern,
        target_dataset_name=target_dataset_name,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    APP()
