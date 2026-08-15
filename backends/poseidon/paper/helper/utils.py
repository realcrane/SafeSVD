def set_module_by_name(model, name, new_module):
    levels = name.split('.')
    if len(levels) == 1:
        setattr(model, levels[0], new_module)
    else:
        parent_name = ".".join(levels[:-1])
        child_name = levels[-1]
        parent = model
        for level in levels[:-1]:
            parent = getattr(parent, level)
        setattr(parent, child_name, new_module)