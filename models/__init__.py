


def create_forward(opt, *argv):
    print(opt.model)
    from .trainer import Trainer
    model = Trainer()
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model

