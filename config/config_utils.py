def get_config(
    config_path: str
):
    import yaml
    f = open(config_path, "r")
    config = yaml.load(f.read(), yaml.Loader)
    f.close()
    return config

def save_config(
    config: dict,
    file_path: str,
):
    pass