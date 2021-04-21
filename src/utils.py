# import
from ruamel.yaml import safe_load
from torchvision import transforms

# def


def load_yaml(file_path):
    with open(file_path, 'r') as f:
        content = safe_load(f)
    return content


def get_transforms(project_parameters):
    transforms_dict = {}
    transforms_config = load_yaml(
        file_path=project_parameters.transforms_config_path)
    for stage in transforms_config.keys():
        transforms_dict[stage] = []
        for k, v in transforms_config[stage].items():
            if v is None:
                transforms_dict[stage].append(
                    eval('transforms.{}()'.format(k)))
            elif type(v) is dict:
                v = ('{},'*len(v)).format(*['{}={}'.format(a, b) for a, b in v.items()])
                transforms_dict[stage].append(
                    eval('transforms.{}({})'.format(k, v)))
            else:
                transforms_dict[stage].append(
                    eval('transforms.{}({})'.format(k, v)))
        transforms_dict[stage] = transforms.Compose(transforms_dict[stage])
    return transforms_dict
