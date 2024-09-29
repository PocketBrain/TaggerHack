# train.py

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import os
import multiprocessing
import numpy as np

from ml.HMC.model import HierarchicalMultimodalClassifier
from ml.HMC.data_loader import VideoDataset
from ml.HMC.loss import hierarchical_loss
from ml.HMC.metrics import calculate_iou
from ml.HMC.utils import build_parent_mapping, decode_predictions, compute_class_weights


def train(imagebind, whisper, summary_tokenizer, summary_model, path_to_save, device):
    multiprocessing.set_start_method("spawn", force=True)

    data_df = pd.read_csv("data/dataset.csv")

    # Filter DataFrame to only include existing videos
    data_df["video_exists"] = data_df["video_id"].apply(
        lambda x: os.path.exists(f"data/videos/{x}.mp4")
    )
    data_df = data_df[data_df["video_exists"]].reset_index(drop=True)
    data_df = data_df.drop(columns=["video_exists"])

    def parse_tags(tag_string):
        tags = tag_string.split(",")
        level1 = []
        level2 = []
        level3 = []
        for tag in tags:
            levels = tag.strip().split(":")
            if len(levels) >= 1 and levels[0] and len(levels[0].strip()) > 2:
                level1.append(levels[0].strip())
            if len(levels) >= 2 and levels[1]:
                level2.append(":".join(levels[:2]).strip())
            if len(levels) == 3 and levels[2]:
                level3.append(":".join(levels[:3]).strip())
        return level1, level2, level3

    level1_tags = []
    level2_tags = []
    level3_tags = []

    for idx, row in data_df.iterrows():
        l1, l2, l3 = parse_tags(row["tags"])
        level1_tags.append(l1 if l1 else ["no_tag"])
        level2_tags.append(l2 if l2 else ["no_tag"])
        level3_tags.append(l3 if l3 else ["no_tag"])

    # Collect all unique tags for each level
    all_level1_tags = set([tag for tags in level1_tags for tag in tags])
    all_level2_tags = set([tag for tags in level2_tags for tag in tags])
    all_level3_tags = set([tag for tags in level3_tags for tag in tags])

    # Initialize MultiLabelBinarizer and exclude 'no_tag' later
    level1_mlb = MultiLabelBinarizer(classes=sorted(all_level1_tags))
    level2_mlb = MultiLabelBinarizer(classes=sorted(all_level2_tags))
    level3_mlb = MultiLabelBinarizer(classes=sorted(all_level3_tags))

    # Fit and transform labels
    level1_encoded = level1_mlb.fit_transform(level1_tags)
    level2_encoded = level2_mlb.fit_transform(level2_tags)
    level3_encoded = level3_mlb.fit_transform(level3_tags)

    # Exclude 'no_tag' from labels and classes
    def exclude_no_tag_labels(encoded_labels, mlb):
        if "no_tag" in mlb.classes_:
            no_tag_index = np.where(mlb.classes_ == "no_tag")[0][0]
            mask = np.ones(len(mlb.classes_), dtype=bool)
            mask[no_tag_index] = False
            mlb.classes_ = mlb.classes_[mask]
            encoded_labels = encoded_labels[:, mask]
        return encoded_labels, mlb

    level1_encoded, level1_mlb = exclude_no_tag_labels(level1_encoded, level1_mlb)
    level2_encoded, level2_mlb = exclude_no_tag_labels(level2_encoded, level2_mlb)
    level3_encoded, level3_mlb = exclude_no_tag_labels(level3_encoded, level3_mlb)

    data_df["level1_encoded"] = list(level1_encoded)
    data_df["level2_encoded"] = list(level2_encoded)
    data_df["level3_encoded"] = list(level3_encoded)

    # Compute class weights for each level
    class_weights = {
        "level1": compute_class_weights(level1_encoded),
        "level2": compute_class_weights(level2_encoded),
        "level3": compute_class_weights(level3_encoded),
    }

    train_df, val_df = train_test_split(data_df, test_size=0.1, random_state=42)

    train_dataset = VideoDataset(
        imagebind,
        whisper,
        summary_tokenizer,
        summary_model,
        train_df,
        level1_mlb,
        level2_mlb,
        level3_mlb,
        device,
        preprocess=True,
    )
    val_dataset = VideoDataset(
        imagebind,
        whisper,
        summary_tokenizer,
        summary_model,
        val_df,
        level1_mlb,
        level2_mlb,
        level3_mlb,
        device,
        preprocess=True,
    )

    batch_size = 4
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    level1_size = len(level1_mlb.classes_)
    level2_size = len(level2_mlb.classes_)
    level3_size = len(level3_mlb.classes_)
    embedding_dim = 1024 * 5  # Adjust based on your embeddings
    hidden_dim = 512

    model = HierarchicalMultimodalClassifier(
        embedding_dim, hidden_dim, level1_size, level2_size, level3_size
    )
    model.to(device)

    # Build parent mapping after excluding 'no_tag'
    parent_mapping = build_parent_mapping(level1_mlb, level2_mlb, level3_mlb, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    num_epochs = 50

    best_val_iou = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for embeddings, y1_true, y2_true, y3_true in train_loader:
            embeddings = embeddings.to(device)
            y1_true = y1_true.to(device)
            y2_true = y2_true.to(device)
            y3_true = y3_true.to(device)

            optimizer.zero_grad()

            y1_pred, y2_pred, y3_pred = model(embeddings)

            loss = hierarchical_loss(
                y1_true,
                y1_pred,
                y2_true,
                y2_pred,
                y3_true,
                y3_pred,
                parent_mapping,
                class_weights,
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        model.eval()

        all_y1_true = []
        all_y2_true = []
        all_y3_true = []
        all_y1_pred = []
        all_y2_pred = []
        all_y3_pred = []

        with torch.no_grad():
            for embeddings, y1_true, y2_true, y3_true in val_loader:
                embeddings = embeddings.to(device)
                y1_true = y1_true.to(device)
                y2_true = y2_true.to(device)
                y3_true = y3_true.to(device)

                y1_pred, y2_pred, y3_pred = model(embeddings)

                all_y1_true.append(y1_true.cpu())
                all_y2_true.append(y2_true.cpu())
                all_y3_true.append(y3_true.cpu())

                all_y1_pred.append(y1_pred.cpu())
                all_y2_pred.append(y2_pred.cpu())
                all_y3_pred.append(y3_pred.cpu())

        # Concatenate all batches
        y1_true_all = torch.cat(all_y1_true, dim=0)
        y2_true_all = torch.cat(all_y2_true, dim=0)
        y3_true_all = torch.cat(all_y3_true, dim=0)

        y1_pred_all = torch.cat(all_y1_pred, dim=0)
        y2_pred_all = torch.cat(all_y2_pred, dim=0)
        y3_pred_all = torch.cat(all_y3_pred, dim=0)

        # Apply thresholds (you can experiment with different thresholds)
        y1_pred_binary = (y1_pred_all.numpy() > 0.5).astype(float)
        y2_pred_binary = (y2_pred_all.numpy() > 0.5).astype(float)
        y3_pred_binary = (y3_pred_all.numpy() > 0.5).astype(float)

        # Convert back to tensors
        y1_pred_binary = torch.tensor(y1_pred_binary)
        y2_pred_binary = torch.tensor(y2_pred_binary)
        y3_pred_binary = torch.tensor(y3_pred_binary)

        # Apply hierarchical masks
        mask2 = y1_pred_binary[:, parent_mapping["l2_to_l1"].cpu()]
        y2_pred_binary = y2_pred_binary * mask2

        mask3 = y2_pred_binary[:, parent_mapping["l3_to_l2"].cpu()]
        y3_pred_binary = y3_pred_binary * mask3

        # Decode predictions and true labels
        pred_tags = decode_predictions(
            y1_pred_binary.numpy(),
            y2_pred_binary.numpy(),
            y3_pred_binary.numpy(),
            level1_mlb,
            level2_mlb,
            level3_mlb,
        )
        true_tags = decode_predictions(
            y1_true_all.numpy(),
            y2_true_all.numpy(),
            y3_true_all.numpy(),
            level1_mlb,
            level2_mlb,
            level3_mlb,
        )

        # Print predicted and true tags for samples
        for i in range(len(pred_tags)):
            print(f"Sample {i}:")
            print(f"True Tags: {true_tags[i]}")
            print(f"Predicted Tags: {pred_tags[i]}")
            print("-" * 50)

        # Concatenate predictions and ground truths across levels
        y_true_all_levels = torch.cat([y1_true_all, y2_true_all, y3_true_all], dim=1)
        y_pred_all_levels = torch.cat(
            [y1_pred_binary, y2_pred_binary, y3_pred_binary], dim=1
        )

        # Calculate total IoU
        total_iou = calculate_iou(y_true_all_levels, y_pred_all_levels)

        print(f"Total Validation IoU: {total_iou:.4f}")

        # Save the model if total IoU improves
        if total_iou > best_val_iou:
            best_val_iou = total_iou
            torch.save(model.state_dict(), f"{path_to_save}/hmc.pth")
            print("Model saved.")

    # Save the final model, encoders
    torch.save(model.state_dict(), f"{path_to_save}/hmc.pth")
    torch.save(
        {
            "level1_mlb": level1_mlb,
            "level2_mlb": level2_mlb,
            "level3_mlb": level3_mlb,
        },
        f"{path_to_save}/label_encoders.pth",
    )


if __name__ == "__main__":
    from ml.lifespan import imagebind, whisper, summary_tokenizer, summary_model, device

    train(imagebind, whisper, summary_tokenizer, summary_model, "ml/models", device)
