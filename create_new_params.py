#coding: utf-8
import os
import yaml
import argparse
from deepscreening.utils import load_params

class DictParamProcessor(argparse.Action):
    def __call__(self, parser, namespace, values, option_strings=None):
        param_dict = getattr(namespace,self.dest,[])
        if param_dict is None:
            param_dict = {}

        k, v = values.split("=")
        param_dict[k] = v
        setattr(namespace, self.dest, param_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--base-params',  type=str)
    parser.add_argument('-u', '--updates', action=DictParamProcessor)
    args = parser.parse_args()

    base_params_path = args.base_params
    updates = args.updates

    *model_dir, base_dir, fn = base_params_path.split("/")
    os.chdir("/".join(model_dir))

    params = load_params(os.path.join(base_dir, fn))
    for k,v in updates.items():
        ori = params.get(k)
        if isinstance(ori, int):
            params[k] = int(v)
        elif isinstance(ori, float):
            params[k] = float(v)
        elif isinstance(ori, bool):
            params[k] = bool(v)
        else:
            params[k] = v
    model_name = params.get("model", "")

    dirname = model_name + ".".join([f"{k}:{v}" for k,v in updates.items()])
    if os.path.exists(dirname):
        raise ValueError(f"{dirname} is already exists.")
    else:
        os.mkdir(dirname)

    with open(os.path.join(dirname, "params.yml"), mode="w") as f:
        f.write(yaml.dump(params))

    print(f"\033[34m{'/'.join(model_dir + [dirname, 'params.yml'])}\033[0m is created.")
