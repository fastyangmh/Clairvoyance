# import
from ruamel.yaml import safe_load
from torchvision import transforms
from os.path import isdir, join, basename
from glob import glob
import xml.etree.ElementTree as ET
from os import makedirs

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
                v = ('{},'*len(v)).format(*
                                          ['{}={}'.format(a, b) for a, b in v.items()])
                transforms_dict[stage].append(
                    eval('transforms.{}({})'.format(k, v)))
            else:
                transforms_dict[stage].append(
                    eval('transforms.{}({})'.format(k, v)))
        transforms_dict[stage] = transforms.Compose(transforms_dict[stage])
    return transforms_dict


def check_path(path, create):
    if not isdir(s=path) and create:
        makedirs(name=path, exist_ok=True)
    else:
        assert isdir(s=path), 'the {} does not exist.'.format(path)


def get_files(path, file_type):
    files = []
    if type(file_type) != list:
        file_type = [file_type]
    for v in file_type:
        files += sorted(glob(join(path, '*.{}'.format(v))))
    return files


def voc_to_yolo_format(annotations_path, labels_path):
    classes = []
    labels = []

    # check path whether exist
    check_path(path=annotations_path, create=False)
    check_path(path=labels_path, create=True)

    # get xml files
    files = get_files(path=annotations_path, file_type='xml')

    # parse each xml
    for f in files:
        tree_root = ET.parse(f).getroot()
        width = int(tree_root.find('size').find('width').text)
        height = int(tree_root.find('size').find('height').text)
        labels.append(join(labels_path, '{}.txt'.format(basename(f)[:-4])))
        with open(labels[-1], 'w') as text_file:
            for obj in tree_root.iter('object'):
                object_name = obj.find('name').text
                if object_name not in classes:
                    classes.append(object_name)
                box = {'xmin': int(eval(obj.find('bndbox').find('xmin').text)),
                       'ymin': int(eval(obj.find('bndbox').find('ymin').text)),
                       'xmax': int(eval(obj.find('bndbox').find('xmax').text)),
                       'ymax': int(eval(obj.find('bndbox').find('ymax').text))}
                x = (box['xmin']+(box['xmax']-box['xmin'])/2)*1.0/width
                y = (box['ymin']+(box['ymax']-box['ymin'])/2)*1.0/height
                w = (box['xmax']-box['xmin'])*1.0/width
                h = (box['ymax']-box['ymin'])*1.0/height
                box = (x, y, w, h)
                text_file.write(
                    '{} {} {} {} {}\n'.format(object_name, *box))
    classes = sorted(classes)

    # write the classes to classes.txt
    with open(join(labels_path, 'classes.txt'), 'w') as text_file:
        for v in classes:
            text_file.write('{}\n'.format(v))

    # convert the list of classes to the dict of classes
    classes = {v: idx for idx, v in enumerate(classes)}
    return classes, labels
