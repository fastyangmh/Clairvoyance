# import
from src.project_parameters import ProjectParameters
from pytorch_lightning import LightningDataModule
from src.utils import get_transforms
from torchvision.datasets import CIFAR10, MNIST, VOCDetection, ImageFolder
from os.path import join
from torch.utils.data import random_split, DataLoader

# def


def get_VOCDetection_dataset(root, train, download, transform):
    year = '2012' if train else '2007'
    image_set = 'train' if train else 'test'
    return VOCDetection(root=root, year=year, image_set=image_set, download=download, transform=transform)

# class


class dataModule(LightningDataModule):
    def __init__(self, project_parameters):
        super().__init__()
        self.project_parameters = project_parameters
        self.transforms_dict = get_transforms(
            project_parameters=project_parameters)
        self.task_dict = {'cifar10': 'CIFAR10',
                          'mnist': 'MNIST', 'vocd': 'get_VOCDetection_dataset'}

    def prepare_data(self):
        if self.project_parameters.predefined_dataset is None:
            self.dataset = {}
            for stage in ['train', 'val', 'test']:
                # get data by ImageFolder
                self.dataset[stage] = ImageFolder(root=join(
                    self.project_parameters.data_path, stage), transform=self.transforms_dict[stage])
                # modify the maximum number of files
                if self.project_parameters.max_files is not None:
                    self.dataset[stage] = random_split(dataset=self.dataset[stage], lengths=(
                        self.project_parameters.max_files, len(self.dataset[stage])-self.project_parameters.max_files))[0]
            # check whether the classes are the same
            if self.project_parameters.max_files is not None:
                assert self.dataset['train'].dataset.class_to_idx == self.project_parameters.classes, 'the classes is not the same. ImageFolder: {} classes: {}'.format(
                    self.dataset['train'].dataset.class_to_idx, self.project_parameters.classes)
            else:
                assert self.dataset['train'].class_to_idx == self.project_parameters.classes, 'the classes is not the same. ImageFolder: {} classes: {}'.format(
                    self.dataset['train'].class_to_idx, self.project_parameters.classes)
        else:
            train_set = eval('{}(root=self.project_parameters.data_path, train=True, download=True, transform=self.transforms_dict["train"])'.format(
                self.task_dict[self.project_parameters.predefined_dataset]))
            test_set = eval('{}(root=self.project_parameters.data_path, train=False, download=True, transform=self.transforms_dict["test"])'.format(
                self.task_dict[self.project_parameters.predefined_dataset]))
            # modify the maximum number of files
            if self.project_parameters.max_files is not None:
                for v in [train_set, test_set]:
                    v.data = v.data[:self.project_parameters.max_files]
                    v.targets = v.targets[:self.project_parameters.max_files]
            train_val_size = [int((1-self.project_parameters.val_size)*len(train_set)),
                              int(self.project_parameters.val_size*len(train_set))]
            train_set, val_set = random_split(
                dataset=train_set, lengths=train_val_size)
            self.dataset = {'train': train_set,
                            'val': val_set,
                            'test': test_set}
            self.project_parameters.classes = self.dataset['train'].dataset.class_to_idx

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.dataset['train'], batch_size=self.project_parameters.batch_size, shuffle=True, pin_memory=self.project_parameters.use_cuda, num_workers=self.project_parameters.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.dataset['val'], batch_size=self.project_parameters.batch_size, shuffle=True, pin_memory=self.project_parameters.use_cuda, num_workers=self.project_parameters.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.dataset['test'], batch_size=self.project_parameters.batch_size, shuffle=True, pin_memory=self.project_parameters.use_cuda, num_workers=self.project_parameters.num_workers)


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # get dataset
    data_module = dataModule(project_parameters=project_parameters)

    # display the dataset information
    data_module.prepare_data()
    for stage in ['train', 'val', 'test']:
        print(stage, data_module.dataset[stage])
