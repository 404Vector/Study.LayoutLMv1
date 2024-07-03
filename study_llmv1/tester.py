from .dependency import *
from .const import *
from .utils import *


def test(
    path_like_pattern: str,
    target_dataset_name: str = "kaggle.document-classification-dataset",
    batch_size=16,
    tokenizer=TOKENIZER,
    dataset_dir=DATASET_DIR,
    result_render_dir: str = RESULT_DIR,
    model_save_dir=MODEL_SAVE_DIR,
):
    os.makedirs(result_render_dir, exist_ok=True)
    assert os.path.exists(
        result_render_dir
    ), f"ERROR, result_render_dir is NOT exist!({result_render_dir})"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: LayoutLMForSequenceClassification = (
        LayoutLMForSequenceClassification.from_pretrained(model_save_dir)
    )  # type: ignore
    model.to(device)
    testing_features = Features(
        {
            "input_ids": Sequence(feature=Value(dtype="int64")),
            "bbox": Array2D(dtype="int64", shape=(512, 4)),
            "attention_mask": Sequence(Value(dtype="int64")),
            "token_type_ids": Sequence(Value(dtype="int64")),
            "image_path": Value(dtype="string"),
            "words": Sequence(feature=Value(dtype="string")),
        }
    )
    dataset_path = os.path.join(dataset_dir, target_dataset_name)
    labels = [label for label in os.listdir(dataset_path)]
    idx2label, label2idx = label_mapping(labels=labels)
    query_df = pd.DataFrame({"image_path": [p for p in glob.glob(path_like_pattern)]})
    assert (
        len(query_df) > 0
    ), f"ERROR, There are no queryed images with {path_like_pattern}"
    query = Dataset.from_pandas(query_df)
    query = query.map(apply_ocr)
    query = query.map(
        lambda example: encode_testing_example(
            example=example,
            tokenizer=tokenizer,
        ),
        features=testing_features,
    )
    query.set_format(
        type="torch", columns=["input_ids", "bbox", "attention_mask", "token_type_ids"]
    )
    loader = DataLoader(query, batch_size=batch_size, shuffle=False)
    preds = []
    for batch in iter(loader):

        outputs = model(
            input_ids=batch["input_ids"].to(device),
            bbox=batch["bbox"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            token_type_ids=batch["token_type_ids"].to(device),
        )
        batch_preds = torch.softmax(outputs.logits, dim=1).tolist()
        preds += batch_preds
    for idx, pred in enumerate(preds):
        label_idx = np.argmax(pred)
        pred_label = idx2label[label_idx]
        raw_image_path = query_df["image_path"][idx]
        image_name = f"{pred_label}-" + raw_image_path.split("/")[-1]
        raw_img = Image.open(raw_image_path)
        raw_img.save(os.path.join(result_render_dir, image_name))
        print(f"image saved at ... {os.path.join(result_render_dir, image_name)}")
