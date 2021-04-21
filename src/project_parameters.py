# import
import argparse
from os.path import abspath, join, basename
from glob import glob
import torch

# class


class ProjectParameters:
    def __init__(self) -> None:
        self._parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # base parameters
        self._parser.add_argument('--mode', type=str, choices=['train', 'predict', 'tune', 'evaluate'], required=True,
                                  help='if the mode equals train, will train the model. if the mode equals predict, will use the pre-trained model to predict. if the mode equals tune, will hyperparameter tuning the model. if the mode equals evaluate, will evaluate the model by the k-fold validation.')
        self._parser.add_argument('--task', type=str, choices=['classification', 'detection'],
                                  required=True, help='the task that provided image classification and object detection.')
        self._parser.add_argument(
            '--data_path', type=str, default='data/Intel_Image_Classification', help='the data path.')
        self._parser.add_argument('--predefined_dataset', type=str, choices=[
                                  'mnist', 'cifar10', 'vocd'], help='the predefined dataset that provided the mnist, cifar10, Pascal VOC Detection dataset.')
        self._parser.add_argument('--classes', type=self._str_to_str_list, default=None,
                                  help='the classes of data. if the value equals None will automatically get the classes of data from the train directory of data_path.')
        self._parser.add_argument('--val_size', type=float, default=0.1,
                                  help='the validation data size used for the predefined task.')
        self._parser.add_argument('--no_cuda', action='store_true', default=False,
                                  help='whether to use Cuda to train the model. if True which will train the model on CPU. if False which will train on GPU.')
        self._parser.add_argument('--num_workers', type=int, default=torch.get_num_threads(
        ), help='how many subprocesses to use for data loading.')

        # config path
        self._parser.add_argument('--transforms_config_path', type=str,
                                  default='config/transforms.yaml', help='the config path.')

        # train
        self._parser.add_argument(
            '--batch_size', type=int, default=32, help='how many samples per batch to load.')

        # debug
        self._parser.add_argument(
            '--max_files', type=self._str_to_int, default=None, help='the maximum number of files.')

    def _str_to_str_list(self, s):
        return [str(v) for v in s.split(',') if len(v) > 0]

    def _str_to_int(self, s):
        if s == 'None' or s == 'none':
            return None
        else:
            return int(s)

    def parse(self):

        project_parameters = self._parser.parse_args()

        # base parameters
        project_parameters.data_path = abspath(project_parameters.data_path)
        if project_parameters.predefined_dataset is not None:
            project_parameters.data_path = join(project_parameters.data_path.rsplit(
                '/', 1)[0], project_parameters.predefined_dataset)
        elif project_parameters.classes is None:
            project_parameters.classes = {c: idx for idx, c in enumerate(sorted(
                [basename(c[:-1]) for c in glob(join(project_parameters.data_path, 'train/*/'))]))}
            project_parameters.num_classes = len(project_parameters.classes)
            assert project_parameters.num_classes, 'there does not get any classes.'
        else:
            project_parameters.classes = {
                c: idx for idx, c in enumerate(sorted(project_parameters.classes))}
            project_parameters.num_classes = len(project_parameters.classes)
        project_parameters.use_cuda = torch.cuda.is_available(
        ) and not project_parameters.no_cuda

        # config path
        project_parameters.transforms_config_path = abspath(
            project_parameters.transforms_config_path)

        return project_parameters


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # display each parameter
    for k, v in vars(project_parameters).items():
        print('{:<12}= {}'.format(k, v))
