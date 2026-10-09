"Bind shared functions to crop-specific parameter namespaces."
from types import FunctionType

def bind_functions(namespace, kernels, names):
    for name in names:
        source = getattr(kernels, name)
        function = FunctionType(source.__code__, namespace, name,
                                source.__defaults__, source.__closure__)
        function.__kwdefaults__ = source.__kwdefaults__
        function.__annotations__ = source.__annotations__
        function.__doc__ = source.__doc__
        function.__module__ = namespace['__name__']
        function.__qualname__ = source.__qualname__
        namespace[name] = function
