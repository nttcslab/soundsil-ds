import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from utils.loaddataset import SoundfieldDatasetLoader
from utils.modelhandler import createmodel
from utils.util import load_config_yaml

import argparse

# set gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(config_train, train_loader, val_loader, timestamp):
    # initialize model
    numepochs = config_train["epoch"]
    model_name = config_train["model"]
    weight = config_train["weight"]
    net, lossfun, optimizer, scheduler = createmodel(model_name, float(config_train["lr"]))
    net.to(device)
    

    # checkpoint directory
    if config_train["cpt_dir"]:
        cpt_dir = os.path.join(config_train["cpt_dir"], timestamp)
        if not os.path.exists(cpt_dir):
            os.makedirs(cpt_dir)
        with open(os.path.join(cpt_dir, "config.yml"), "w") as f:
            yaml.dump(config_train, f)
    else:
        cpt_dir = None

    # initialize losses
    train_losses = torch.zeros(numepochs)
    val_losses = torch.zeros(numepochs)

    print("--- training starts ---")
    for epoch in tqdm(range(numepochs)):
        # training
        net.train()
        batch_loss = []
        for X, y, label in train_loader:
            # push data to GPU
            X = X.to(device)
            y = y.to(device)
            label = label.to(device)

            # forward pass and loss
            yHat_denoise, yHat_seg = net(X)
            loss = lossfun['denoise'](yHat_denoise, y) + weight*lossfun['seg'](yHat_seg, label)
            
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss from this batch
            batch_loss.append(loss.item())
        train_losses[epoch] = np.mean(batch_loss)
        del X, y, yHat_denoise, yHat_seg, label
        torch.cuda.empty_cache()

        scheduler.step()

        # validation
        net.eval()
        batch_val_loss = []
        for X, y, label in val_loader:
            X = X.to(device)
            y = y.to(device)
            label = label.to(device)

            with torch.no_grad():
                yHat_denoise, yHat_seg = net(X)

            loss_tmp =  lossfun['denoise'](yHat_denoise, y).item() + weight*lossfun['seg'](yHat_seg, label).item()
            batch_val_loss.append(loss_tmp)
        val_losses[epoch] = np.mean(batch_val_loss)
        del X, y, yHat_denoise, yHat_seg, label
        torch.cuda.empty_cache()

        if cpt_dir:
            save_checkpoint(cpt_dir, epoch, net, train_losses, val_losses)

    print("--- training ends ---")

    # function output
    return net, train_losses, val_losses


def save_checkpoint(cpt_dir, epoch, net, tloss, vloss):
    torch.save(net.state_dict(), os.path.join(cpt_dir, f"checkpoint_{epoch}.pth"))
    np.save(os.path.join(cpt_dir, f"checkpoint_{epoch}_trainloss"), tloss)
    np.save(os.path.join(cpt_dir, f"checkpoint_{epoch}_validloss"), vloss)


def save_results(save_dir, config, net, tloss, vloss, training_time):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, "config.yml"), "w") as f:
        yaml.dump(config, f)

    # Save weights and losses
    save_name = f'{config["model"]}_white'
    torch.save(net.state_dict(), os.path.join(save_dir, f"{save_name}.pth"))
    np.save(os.path.join(save_dir, f"{save_name}_trainloss"), tloss)
    np.save(os.path.join(save_dir, f"{save_name}_validloss"), vloss)

    # Save training time
    with open(os.path.join(save_dir, f"{save_name}_trainingtime.txt"), "w") as f:
        f.write(str(training_time))

    # Plot loss curves
    fig_loss, ax = plt.subplots(1, 1)
    ax.plot(tloss, "s-", label="train")
    ax.plot(vloss, "o-", label="validation")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title(config["model"] + " loss")
    ax.legend()
    fig_loss.savefig(os.path.join(save_dir, f"{save_name}_loss.png"))


def main(args):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load config file
    config_file_path = args.config
    yaml_contents = load_config_yaml(config_file_path)
    config_train = yaml_contents["train"]
    config_valid = yaml_contents["validation"]

    # Load train dataset
    datasetloader = SoundfieldDatasetLoader(config_train["dataset"])
    train_dataset = datasetloader.load()
    train_loader = DataLoader(
        train_dataset,
        batch_size=config_train["batch_size"],
        shuffle=True,
        drop_last=True,
    )

    # Load valid dataset
    datasetloader = SoundfieldDatasetLoader(config_valid["dataset"])
    valid_dataset = datasetloader.load()
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config_valid["batch_size"],
    )

    # Training
    time_start = time.perf_counter()
    net, tloss, vloss = train(
        config_train,
        train_loader=train_loader,
        val_loader=valid_loader,
        timestamp=timestamp,
    )
    time_end = time.perf_counter()
    training_time = time_end - time_start

    # Save results
    if config_train["save_dir"]:
        save_dir = os.path.join(config_train["save_dir"], timestamp)
        save_results(save_dir, config_train, net, tloss, vloss, training_time)

    del net, tloss, vloss
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yml')
    args = parser.parse_args()

    main(args)
