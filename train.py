"""Contains the training of the normalizing flow model."""

import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy
import pandas
import torch
import torch.backends.cudnn
from sklearn import metrics
from torch import optim

from configuration import Configuration
from normalizing_flow import NormalizingFlow, get_loss, get_loss_per_sample
from voraus_ad import ANOMALY_CATEGORIES, Signals, load_torch_dataloaders

# If deterministic CUDA is activated, some calculations cannot be calculated in parallel on the GPU.
# The training will take much longer but is reproducible.
DETERMINISTIC_CUDA = False
DATASET_PATH = Path.home() / "Downloads" / "voraus-ad-dataset-100hz.parquet"
MODEL_PATH: Optional[Path] = Path.cwd() / "model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the training configuration and hyperparameters of the model.
configuration = Configuration(
    columns="machine",
    epochs=70,
    frequencyDivider=1,
    trainGain=1.0,
    seed=177,
    batchsize=32,
    nCouplingBlocks=4,
    clamp=1.2,
    learningRate=8e-4,
    normalize=True,
    pad=True,
    nHiddenLayers=0,
    scale=2,
    kernelSize1=13,
    dilation1=2,
    kernelSize2=1,
    dilation2=1,
    kernelSize3=1,
    dilation3=1,
    milestones=[11, 61],
    gamma=0.1,
)

# Make the training reproducible.
torch.manual_seed(configuration.seed)
torch.cuda.manual_seed_all(configuration.seed)
numpy.random.seed(configuration.seed)
random.seed(configuration.seed)
if DETERMINISTIC_CUDA:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Disable pylint too-many-variables here for readability.
# The whole training should run in a single function call.
def train() -> List[Dict]:  # pylint: disable=too-many-locals
    """Trains the model with the paper-given parameters.

    Returns:
        The auroc (mean over categories) and loss per epoch.
    """
    # Load the dataset as torch data loaders.
    train_dataset, _, train_dl, test_dl = load_torch_dataloaders(
        dataset=DATASET_PATH,
        batch_size=configuration.batchsize,
        columns=Signals.groups()[configuration.columns],
        seed=configuration.seed,
        frequency_divider=configuration.frequency_divider,
        train_gain=configuration.train_gain,
        normalize=configuration.normalize,
        pad=configuration.pad,
    )

    # Retrieve the shape of the data for the model initialization.
    n_signals = train_dataset.tensors[0].shape[1]
    n_times = train_dataset.tensors[0].shape[0]
    # Initialize the model, optimizer and scheduler.
    model = NormalizingFlow((n_signals, n_times), configuration).float().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=configuration.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=configuration.milestones, gamma=configuration.gamma
    )

    training_results: List[Dict] = []
    # Iterate over all epochs.
    for epoch in range(configuration.epochs):
        # TRAIN THE MODEL.
        model.train()
        loss: float = 0
        for tensors, _ in train_dl:
            tensors = tensors.float().to(DEVICE)

            # Execute the forward and jacobian calculation.
            optimizer.zero_grad()
            latent_z, jacobian = model.forward(tensors.transpose(2, 1))
            jacobian = torch.sum(jacobian, dim=tuple(range(1, jacobian.dim())))

            # Back propagation and loss calculation.
            batch_loss = get_loss(latent_z, jacobian)
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.item()

        # Calculate the mean loss over all samples.
        loss = loss / len(train_dl)

        # VALIDATE THE MODEL.
        model.eval()
        with torch.no_grad():
            result_list: List[Dict] = []
            for _, (tensors, labels) in enumerate(test_dl):
                tensors = tensors.float().to(DEVICE)

                # Calculate forward and jacobian.
                latent_z, jacobian = model.forward(tensors.transpose(2, 1))
                jacobian = torch.sum(jacobian, dim=tuple(range(1, jacobian.dim())))
                # Calculate the anomaly score per sample.
                loss_per_sample = get_loss_per_sample(latent_z, jacobian)

                # Append the anomaly score and the labels to the results list.
                for j in range(loss_per_sample.shape[0]):
                    result_labels = {k: v[j].item() if isinstance(v, torch.Tensor) else v[j] for k, v in labels.items()}
                    result_labels.update(score=loss_per_sample[j].item())
                    result_list.append(result_labels)

        results = pandas.DataFrame(result_list)

        # Calculate AUROC per anomaly category.
        aurocs = []
        for category in ANOMALY_CATEGORIES:
            dfn = results[(results["category"] == category.name) | (~results["anomaly"])]
            fpr, tpr, _ = metrics.roc_curve(dfn["anomaly"], dfn["score"].values, pos_label=True)
            auroc = metrics.auc(fpr, tpr)
            aurocs.append(auroc)

        # Calculate the AUROC mean over all categories.
        aurocs_array = numpy.array(aurocs)
        auroc_mean = aurocs_array.mean()
        training_results.append({"epoch": epoch, "aurocMean": auroc_mean, "loss": loss})
        print(f"Epoch {epoch:0>3d}: auroc(mean)={auroc_mean:5.3f}, loss={loss:.6f}")

        scheduler.step()

    if MODEL_PATH is not None:
        torch.save(model.state_dict(), MODEL_PATH)

    return training_results


if __name__ == "__main__":
    train()
