import os
import yaml
from easydict import EasyDict as edict
class YamlParser(edict):
    """
    This is yaml parser based on EasyDict.
    """
    def __init__(self, cfg_dict=None, config_file=None):
        if cfg_dict is None:
            cfg_dict = {}

        if config_file is not None:
            assert(os.path.isfile(config_file))
            with open(config_file, 'r') as fo:
                yaml_ = yaml.safe_load(fo)
                cfg_dict.update(yaml_)
        super(YamlParser, self).__init__(cfg_dict)
if __name__ == '__main__':
    cfgs = YamlParser(config_file = 'E:/Work/BRIEFCAM/deep_sort_pytorch/configs/deep_sort.yaml')
    print(cfgs)