import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from classification.config.ConfigParams import ConfigParams
from classification.kaggle.export import export_results
from classification.data.Preprocessing import Preprocessing
from classification.data.SingleDirDataset import SingleDirDataset
from classification.model.ModelsFactory import ModelsFactory

# FIXME: Adapt to any class names
classes = ['cat', 'dog']


def do_parsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="PyTorch test script")
    parser.add_argument("--dataset_test_dir", required=True, type=str, help="Dataset test directory")
    parser.add_argument("--config_file", required=True, type=str, help="Config file path")
    parser.add_argument("--model_path", required=False, type=str, default="./export/model.pth",
                        help="Filepath with trained model")
    parser.add_argument("--kaggle_export_file", required=False, type=str, default=None,
                        help="CSV file in kaggle format for challenge upload")
    args = parser.parse_args()
    return args


def main():

    # Load params
    args = do_parsing()
    print(args)

    # Load config file with model and preprocessing (must be the same used in training to be coherent)
    config = ConfigParams(args.config_file)

    # Prepare preprocessing transform pipeline (same processing of validation dataset)
    preprocessing_transforms = Preprocessing(config)
    preprocessing_transforms_test = preprocessing_transforms.get_transforms_val()

    # Read test Dataset,
    dataset_test = SingleDirDataset(args.dataset_test_dir, preprocessing_transforms_test)
    print("Test - Samples: {0}".format(str(len(dataset_test))))

    # Load model and apply .eval() and .cuda()
    model = ModelsFactory.create(config, len(classes))
    print(model)
    model.cuda()
    model.eval()

    # Load trained weights
    model.load_state_dict(torch.load(args.model_path))

    # Create a PyTorch DataLoader from CatDogDataset
    test_loader = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False, num_workers=8)

    # Results for each image compatible with kaggle format (we need to export the dog probability)
    results = []

    print("Evaluating test dataset...")

    for batch_i, data in enumerate(test_loader):

        # Retrieve images
        images = data["image"]

        # Move to GPU
        images = images.type(torch.cuda.FloatTensor)

        # forward pass to get outputs
        output = model(images)
        probs = nn.Softmax(dim=1)(output)
        probs_np = probs.cpu().data.numpy()

        files = data["file"]
        for index, file in enumerate(files):
            results.append((os.path.basename(file)[:os.path.basename(file).rfind('.')],
                            probs_np[index][classes.index("dog")]))

    print("Test finished")

    if args.kaggle_export_file is not None:
        export_results(results, args.kaggle_export_file)


if __name__ == "__main__":
    main()
