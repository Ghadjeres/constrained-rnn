from torch import optim


def optimizer_from_name(name: str, **kwargs):
    if name == 'adam':
        return lambda params: optim.Adam(params, **kwargs)

    elif name == 'rmsprop':
        return lambda params: optim.RMSprop(params, **kwargs)

    else:
        raise NotImplementedError
