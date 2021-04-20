# import
import argparse
from os.path import abspath

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

    def parse(self):
        project_parameters = self._parser.parse_args()

        # base parameters
        project_parameters.data_path = abspath(project_parameters.data_path)

        # check all parameters
        for k, v in vars(project_parameters).items():
            assert v, '{} is {}'.format(k, v)
        return project_parameters


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # display each parameter
    for k, v in vars(project_parameters).items():
        print('{:<12}= {}'.format(k, v))
