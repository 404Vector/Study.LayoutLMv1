from ..dependency import *
from ..const import *
from .apply_ocr import apply_ocr
from .encode_training_example import encode_training_example


def training_dataloader_from_df(
    data,
    label2idx,
    tokenizer,
    training_features,
) -> Tuple[DataLoader, int]:
    dataset = Dataset.from_pandas(data)
    dataset = dataset.map(apply_ocr)

    encoded_dataset = dataset.map(
        lambda example: encode_training_example(
            example=example,
            label2idx=label2idx,
            tokenizer=tokenizer,
        ),
        features=training_features,
    )
    encoded_dataset.set_format(
        type="torch",
        columns=["input_ids", "bbox", "attention_mask", "token_type_ids", "label"],
    )
    dataloader = DataLoader(encoded_dataset, batch_size=1, shuffle=True)
    batch = next(iter(dataloader))
    return dataloader, batch
