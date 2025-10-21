import yaml
from pydicti import odicti


def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=odicti):
    """
    Load YAML mappings as ordered dictionaries.
    """

    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )

    return yaml.load(stream, OrderedLoader)
