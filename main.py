import argparse
import pathlib

import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from models.cnn import CNN
from models.mlp import MLP
from pipeline import Pipeline

from plotcm import plot_confusion_matrix
#from pytorch_image_classification import get_default_config, update_config, create_model, create_dataloader


# def load_config():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', type=str)
#     parser.add_argument('--resume', type=str, default='')
#     parser.add_argument('--local_rank', type=int, default=0)
#     parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
#     args = parser.parse_args()
#
#     configs = get_default_config()
#     if args.config is not None:
#         configs.merge_from_file(args.config)
#     configs.merge_from_list(args.options)
#     if not torch.cuda.is_available():
#         configs.device = 'cpu'
#         configs.train.dataloader.pin_memory = False
#     if args.resume != '':
#         config_path = pathlib.Path(args.resume) / 'config.yaml'
#         configs.merge_from_file(config_path.as_posix())
#         configs.merge_from_list(['train.resume', True])
#     configs.merge_from_list(['train.dist.local_rank', args.local_rank])
#     configs = update_config(configs)
#     configs.freeze()
#     return configs


@torch.no_grad()
def get_all_preds(model, loader):
    _, (images, labels) = next(enumerate(loader))

    output = model(images)
    preds = output.data.max(1, keepdim=True)[1]

    return preds.data.view_as(labels)


# config = load_config()
# model = create_model(config)
# checkpoint = torch.load('data/checkpoint_00200.pth', map_location=torch.device('cpu'))
#
# # total correct: 9866
# # Accuracy: 99%
# model.load_state_dict(checkpoint['model'])
# model.eval()
#
# test_loader = create_dataloader(config, is_train=False)
#
# with torch.no_grad():
#     prediction_loader = torch.utils.data.DataLoader(test_loader.dataset, batch_size=10000)
#     test_preds = get_all_preds(model, prediction_loader)
#
# preds_correct = test_preds.eq(test_loader.dataset.targets).sum()
# print('total correct: {:d}'.format(preds_correct))
# print('Accuracy: {:.0f}% \n'.format(100. * preds_correct / len(test_loader.dataset)))
#
# cm = confusion_matrix(test_loader.dataset.targets, test_preds)
#
# plt.figure(figsize=(10, 10))
# plot_confusion_matrix(cm, test_loader.dataset.classes)
# plt.show()

def plot_result(results, names):
    """
    Take a 2D list/array, where row is accuracy at each epoch of training for given model, and
    names of each model, and display training curves
    """
    for i, r in enumerate(results):
        plt.plot(range(len(r)), r, label=names[i])
    plt.legend()
    plt.title("KMNIST")
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # plt.savefig("./part_2_plot.png")


def main():
    models = [MLP(), CNN()]
    epochs = 200
    results = []

    # Can comment the below out during development
    images, labels = Pipeline(MLP(), 0.003).view_batch()
    print(labels)
    plt.imshow(images, cmap="Greys")
    plt.show()

    for model in models:
        print(f"Training {model.__class__.__name__}...")
        m = Pipeline(model, 0.003)

        accuracies = [0]
        for e in range(epochs):
            m.train_epoch()
            accuracy = m.eval()
            print(f"Epoch: {e}/{epochs}.. Test Accuracy: {accuracy}")
            accuracies.append(accuracy)
        results.append(accuracies)

    #plot_result(results, [m.__class__.__name__ for m in models])
    [torch.save(m.state_dict(), f'results/{m.__class__.__name__}.pth') for m in models]


if __name__ == "__main__":
    main()
