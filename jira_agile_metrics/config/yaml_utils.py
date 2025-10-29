"""YAML utilities for configuration processing.

This module provides utilities for loading YAML configuration files with
ordered dictionaries.
"""

import yaml
from pydicti import odicti


def ordered_load(stream, loader=yaml.SafeLoader, object_pairs_hook=odicti):
    """
    Load YAML mappings as ordered dictionaries.
    """

    # Create a custom constructor function instead of a class
    def construct_mapping(loader, node, _deep=False):
        """Construct mapping with preserved order."""
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    # Dynamically create a new Loader subclass to avoid mutating the original loader
    # Ensure the new class has its own copy of yaml_constructors
    if hasattr(loader, "yaml_constructors"):
        # Copy the parent's constructors to avoid mutation
        parent_constructors = dict(loader.yaml_constructors)
        NewLoader = type(
            "NewLoader", (loader,), {"yaml_constructors": parent_constructors}
        )
    else:
        NewLoader = type("NewLoader", (loader,), {})

    NewLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )

    return yaml.load(stream, NewLoader)
