from classification.model.SqNet import SqNet
from classification.model.ResNet50 import ResNet50
from classification.model.SimpleNetBW import SimpleNetBW


class ModelsFactory(object):

    @classmethod
    def create(cls, config, num_classes):

        if config.architecture == "sqnet":

            # Transfer learning
            return SqNet(num_classes)

        elif config.architecture == "resnet50":

            # Transfer learning
            return ResNet50(num_classes)

        elif config.architecture == "simplenet_bw":

            # Transfer learning
            return SimpleNetBW(config.input_size, num_classes)

        else:

            raise Exception("Model architecture " + config.architecture + " not supported")