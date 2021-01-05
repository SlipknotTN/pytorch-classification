import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from classification.config.ConfigParams import ConfigParams
from classification.data.Preprocessing import Preprocessing
from classification.data.StandardDataset import StandardDataset
from classification.model.ModelsFactory import ModelsFactory


def do_parsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="PyTorch training script")
    parser.add_argument("--dataset_train_dir", required=True, type=str, help="Dataset train directory")
    parser.add_argument("--dataset_val_dir", required=True, type=str, help="Dataset validation directory")
    parser.add_argument("--config_file", required=True, type=str, help="Config file path")
    parser.add_argument("--model_output_path", required=False, type=str, default="./export/model.pth",
                        help="Filepath where to save the PyTorch model")
    args = parser.parse_args()
    return args


def main():

    # Load params
    args = do_parsing()
    print(args)

    # Load config file with model, hyperparameters and preprocessing
    config = ConfigParams(args.config_file)

    # Prepare preprocessing transform pipeline
    preprocessing_transforms = Preprocessing(config)
    preprocessing_transforms_train = preprocessing_transforms.get_transforms_train()
    preprocessing_transforms_val = preprocessing_transforms.get_transforms_val()

    # Read Dataset
    dataset_train = StandardDataset(args.dataset_train_dir, preprocessing_transforms_train)
    print("Train - Classes: {0}, Samples: {1}".format(str(len(dataset_train.get_classes())), str(len(dataset_train))))
    dataset_val = StandardDataset(args.dataset_val_dir, preprocessing_transforms_val)
    print("Validation - Classes: {0}, Samples: {1}".
          format(str(len(dataset_val.get_classes())), str(len(dataset_val))))
    print("Classes " + str(dataset_train.get_classes()))

    # Load model and apply .train() and .cuda()
    model = ModelsFactory.create(config, len(dataset_train.get_classes()))
    print(model)
    model.cuda()
    model.train()

    # Create a PyTorch DataLoader from CatDogDataset (two of them: train + val)
    train_loader = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(dataset_val, batch_size=config.batch_size, shuffle=False, num_workers=8)

    # Set Optimizer and Loss
    # CrossEntropyLoss add LogSoftmax to the model while NLLLoss doesn't do it
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)

    for epoch in range(config.epochs):

        running_loss = 0.0

        # Iterate on train batches and update weights using loss
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding labels
            images = data['image']
            gts = data['gt']

            # Move to GPU
            gts = gts.type(torch.cuda.LongTensor)
            images = images.type(torch.cuda.FloatTensor)

            # forward pass to get outputs
            output = model(images)

            # calculate the loss between predicted and target class
            loss = criterion(output, gts)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()

            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # print loss statistics
            running_loss += loss.item()
            if batch_i % 10 == 9:  # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i + 1, running_loss / 10))
                running_loss = 0.0

        # Iterate on validation batches
        print("Calculating validation accuracy...")
        exact_match = 0
        for batch_i, data in enumerate(val_loader):
            # get the input images and their corresponding labels
            images = data['image']
            gts = data['gt']

            # Move to GPU
            gts = gts.type(torch.cuda.LongTensor)
            images = images.type(torch.cuda.FloatTensor)

            # forward pass to get outputs
            output = model(images)

            # Calculate Accuracy
            output = output.cpu().data.numpy()
            predictions = np.argmax(output, axis=1)
            gts_np = gts.cpu().data.numpy()
            correct = np.count_nonzero(predictions == gts_np)
            exact_match += correct

        print('Epoch: {}, Validation Accuracy: {}'.format(epoch + 1, exact_match / len(dataset_val)))

    # Save model
    torch.save(model.state_dict(), args.model_output_path)

    print("End")


if __name__ == "__main__":
    main()
