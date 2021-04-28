# import
from torch.utils.data.dataset import random_split
from src.project_parameters import ProjectParameters
from pytorch_lightning import LightningDataModule
from src.utils import get_files, get_transforms, voc_to_yolo_format
from torchvision.datasets import CIFAR10, MNIST, VOCDetection, ImageFolder
from os.path import join
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# def


def get_VOCDetection_dataset(root, train, download, transform):
    year = '2012' if train else '2007'
    image_set = 'train' if train else 'test'
    dataset = VOCDetection(
        root=root, year=year, image_set=image_set, download=download, transform=transform)
    annotations_path = dataset.annotations[0].rsplit('/', 1)[0]
    labels_path = join(annotations_path.rsplit('/', 1)[0], 'labels')
    classes, labels = voc_to_yolo_format(
        annotations_path=annotations_path, labels_path=labels_path)
    return YOLOFormatDataset(classes=classes, images=dataset.images, labels=labels, transform=transform)

# class


class YOLOFormatDataset(Dataset):
    def __init__(self, classes, images, labels, transform) -> None:
        super().__init__()
        self.class_to_idx = classes
        self.data = images
        self.targets = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        x, y = self.data[index], self.targets[index]
        x = Image.open(x)
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __getattribute__(self, name: str):
        return super().__getattribute__(name)


class dataModule(LightningDataModule):
    def __init__(self, project_parameters):
        super().__init__()
        self.project_parameters = project_parameters
        self.transforms_dict = get_transforms(
            project_parameters=project_parameters)
        self.predefined_dataset_dict = {
            'cifar10': 'CIFAR10', 'mnist': 'MNIST', 'vocd': 'get_VOCDetection_dataset'}

    def prepare_data(self):
        if self.project_parameters.predefined_dataset is None:
            self.dataset = {}
            for stage in ['train', 'val', 'test']:
                if self.project_parameters.task == 'classification':
                    self.dataset[stage] = ImageFolder(root=join(
                        self.project_parameters.data_path, stage), transform=self.transforms_dict[stage])
                elif self.project_parameters.task == 'detection':
                    annotations_path = join(
                        self.project_parameters.data_path, '{}/annotations'.format(stage))
                    labels_path = join(
                        self.project_parameters.data_path, '{}/labels'.format(stage))
                    classes, labels = voc_to_yolo_format(
                        annotations_path=annotations_path, labels_path=labels_path)
                    images = get_files(
                        path=join(self.project_parameters.data_path, '{}/images'.format(stage)), file_type=['jpg', 'png'])
                    self.dataset[stage] = YOLOFormatDataset(
                        classes=classes, images=images, labels=labels, transform=self.transforms_dict[stage])
                # modify the maximum number of files
                if self.project_parameters.max_files is not None:
                    lengths = (self.project_parameters.max_files,
                               len(self.dataset[stage])-self.project_parameters.max_files)
                    self.dataset[stage] = random_split(
                        dataset=self.dataset[stage], lengths=lengths)[0]
            # set the project_parameters.classes
            if self.project_parameters.max_files is not None:
                self.project_parameters.classes = self.dataset['train'].dataset.class_to_idx
            else:
                self.project_parameters.classes = self.dataset['train'].class_to_idx
        else:
            train_set = eval('{}(root=self.project_parameters.data_path, train=True, download=True, transform=self.transforms_dict["train"])'.format(
                self.predefined_dataset_dict[self.project_parameters.predefined_dataset]))
            test_set = eval('{}(root=self.project_parameters.data_path, train=False, download=True, transform=self.transforms_dict["test"])'.format(
                self.predefined_dataset_dict[self.project_parameters.predefined_dataset]))
            # modify the maximum number of files
            if self.project_parameters.max_files is not None:
                for v in [train_set, test_set]:
                    v.data = v.data[:self.project_parameters.max_files]
                    v.targets = v.targets[:self.project_parameters.max_files]
            train_val_size = [round((1-self.project_parameters.val_size)*len(train_set)),
                              round(self.project_parameters.val_size*len(train_set))]
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

    # create dataModule object
    data_module = dataModule(project_parameters=project_parameters)

    # display the dataset information
    data_module.prepare_data()
    for stage in ['train', 'val', 'test']:
        print(stage, data_module.dataset[stage])
