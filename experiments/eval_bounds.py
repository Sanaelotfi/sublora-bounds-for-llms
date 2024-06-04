from argparse import ArgumentParser
from fastargs import get_current_config
from sublora.sublora_pipeline import SubLoRA


def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description="SubLoRA GPT-2 bound evaluation")
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode="stderr")
    if not quiet:
        config.summary()
    return config

if __name__ == "__main__":
    yaml_config = make_config()
    yaml_config = {key[1]: value for key, value in yaml_config.content.items()}
    yaml_config["action"] = "eval_bounds"
    method = SubLoRA(yaml_config)
    method.get_bounds()