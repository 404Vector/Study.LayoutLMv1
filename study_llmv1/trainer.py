from .dependency import *
from .const import *
from .utils import *


def train(
    target_dataset_name: str = "kaggle.document-classification-dataset",
    test_size=0.09,
    num_epochs=3,
    lr=4e-5,
    seed=0,
):
    tokenizer = TOKENIZER
    dataset_path = os.path.join(DATASET_DIR, target_dataset_name)
    #
    labels = [label for label in os.listdir(dataset_path)]
    idx2label, label2idx = label_mapping(labels=labels)
    print(label2idx)

    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: (
        LayoutLMForSequenceClassification
    ) = LayoutLMForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2idx),
    )  # type: ignore
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    #
    data = create_dataframe(dataset_path=dataset_path)
    train_data, valid_data = train_test_split(
        data,
        test_size=test_size,
        random_state=seed,
        stratify=data.label,
    )

    train_data = train_data.reset_index(drop=True)
    valid_data = valid_data.reset_index(drop=True)
    print(f"{len(train_data)} training examples, {len(valid_data)} validation examples")
    print(data.head())

    # we need to define the features ourselves as the bbox of LayoutLM are an extra feature
    training_features = Features(
        {
            "input_ids": Sequence(feature=Value(dtype="int64")),
            "bbox": Array2D(dtype="int64", shape=(512, 4)),
            "attention_mask": Sequence(Value(dtype="int64")),
            "token_type_ids": Sequence(Value(dtype="int64")),
            "label": ClassLabel(names=list(idx2label.keys())),
            "image_path": Value(dtype="string"),
            "words": Sequence(feature=Value(dtype="string")),
        }
    )
    train_dataloader, train_batch = training_dataloader_from_df(
        data=train_data,
        label2idx=label2idx,
        tokenizer=tokenizer,
        training_features=training_features,
    )
    valid_dataloader, valid_batch = training_dataloader_from_df(
        data=train_data,
        label2idx=label2idx,
        tokenizer=tokenizer,
        training_features=training_features,
    )

    for epoch in range(num_epochs):
        print("Epoch:", epoch)
        training_loss = 0.0
        training_correct = 0
        # put the model in training mode
        model.train()
        for batch in tqdm(train_dataloader):
            labels = batch["label"].to(device)
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                bbox=batch["bbox"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                token_type_ids=batch["token_type_ids"].to(device),
                labels=labels,
            )
            loss = outputs.loss

            training_loss += loss.item()
            predictions = outputs.logits.argmax(-1)
            training_correct += (predictions == labels).float().sum()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("Training Loss:", training_loss / batch["input_ids"].shape[0])
        training_accuracy = 100 * training_correct / len(train_data)
        print("Training accuracy:", training_accuracy.item())

        validation_loss = 0.0
        validation_correct = 0
        for batch in tqdm(valid_dataloader):
            labels = batch["label"].to(device)
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                bbox=batch["bbox"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                token_type_ids=batch["token_type_ids"].to(device),
                labels=labels,
            )
            loss = outputs.loss

            validation_loss += loss.item()
            predictions = outputs.logits.argmax(-1)
            validation_correct += (predictions == labels).float().sum()

        print("Validation Loss:", validation_loss / batch["input_ids"].shape[0])
        validation_accuracy = 100 * validation_correct / len(valid_data)
        print("Validation accuracy:", validation_accuracy.item())

    #
    model.save_pretrained("saved_model/")
