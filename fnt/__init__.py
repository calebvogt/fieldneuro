import importlib
import pkgutil
import inspect

def _import_submodules(package_name):
    """Dynamically imports all submodules and makes their functions available."""
    package = importlib.import_module(package_name)
    
    for _, module_name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        module = importlib.import_module(module_name)

        # Import only functions and make them directly accessible in fnt
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)
            if inspect.isfunction(attribute):  # Import only functions
                globals()[attribute_name] = attribute  # Make it accessible as fnt.function_name

# Automatically import all submodules
_import_submodules("fnt")
