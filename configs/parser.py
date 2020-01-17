import os
import yaml
import logging
import argparse


class YamlStructure(dict):
    def __init__(self, data):
        super().__init__()
        assert isinstance(data, dict), "Check Data Type"
        self.update(data)

    def __getattr__(self, name):
        if name in self.keys():
            return self[name]

    def __repr__(self, path=None):
        tmp = {}

        def update(src, dst):
            for k, v in src.items():
                if isinstance(v, dict):
                    dst[k] = {}
                    update(v, dst[k])
                else:
                    dst[k] = v

        update(self, tmp)

        if path is not None:
            with open(path, 'w') as f:
                yaml.dump(tmp, f)
            return f"Yaml Dump in {path}"
        else:
            return yaml.dump(tmp)


class Parser:
    @staticmethod
    def load(path):
        with open(path, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        C = YamlStructure(data)

        def to_structure(d):
            for k, v in d.items():
                if isinstance(v, (YamlStructure, dict)):
                    d[k] = YamlStructure(v)
                    to_structure(v)

        to_structure(C)
        return C

    def __init__(self, path, args=None):
        if args is not None:
            raise NotImplementedError("Don't use args")
            assert isinstance(args, argparse.Namespace), "Check args"

        path = f"{os.getcwd()}/{path}"
        default_path = f"{os.getcwd()}/configs/default.yaml"
        self.init_yaml(default_path)
        self.update_yaml(path)

    def init_yaml(self, path):
        with open(path, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        self.C = YamlStructure(data)

        def to_structure(d):
            for k, v in d.items():
                if isinstance(v, (YamlStructure, dict)):
                    d[k] = YamlStructure(v)
                    to_structure(v)

        to_structure(self.C)

    def update_yaml(self, path):
        with open(path, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        def update(src, dst):
            for k, v in src.items():
                if isinstance(v, dict):
                    update(v, dst[k])
                elif k in dst.keys():
                    dst[k] = v
                else:
                    raise EnvironmentError(f"key({k}) must be in default.yaml")

        update(data, self.C)

    def dump(self, path=None):
        self.C.__repr__(path)


def arg_parse():
    # projects description
    desc = "Test Parser"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--save_dir', type=str, default="asdf")
    parser.add_argument('--gpu', type=int, default=0, help="Only single gpu")
    parser.add_argument('--const_top_k', type=float, default=0.01)
    return parser.parse_args()


if __name__ == "__main__":
    path = "/data2/DW/automl/auto-dip/config/default.yaml"
    a = {"b": 1}
    args = arg_parse()
    p = Parser(path)
    print(p.C)
    print(p.C.data.name)
    print(p.C.search)
    p.dump()
