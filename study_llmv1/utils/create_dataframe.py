from ..dependency import *
from ..const import *


def create_dataframe(dataset_path: str) -> pd.DataFrame:
    images = []
    labels = []
    for label in [label for label in os.listdir(dataset_path)]:
        images.extend(
            [
                f"{dataset_path}/{label}/{img_name}"
                for img_name in os.listdir(f"{dataset_path}/{label}")
            ]
        )
        labels.extend(
            [label for _ in range(len(os.listdir(f"{dataset_path}/{label}")))]
        )
    df = pd.DataFrame({"image_path": images, "label": labels})
    return df
