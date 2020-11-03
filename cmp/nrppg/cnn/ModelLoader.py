import torch
import os
import logging

__logging_format__ = '[%(levelname)s] %(message)s'
logging.basicConfig(format=__logging_format__)
logger = logging.getLogger("model_loader_log")
logging.getLogger("model_loader_log").setLevel(logging.INFO)


class ModelLoader(object):
    @staticmethod
    def initialize_model(net_architecture_name, model_type='extractor', use_gpu=True):
        rgb = False
        if 'RGB' in net_architecture_name:
            rgb = True
            logger.debug("Using RGB")

        net_architecture_name = net_architecture_name.replace('RGB', '')
        module = __import__('cmp.nrppg.cnn.%s.%s' % (model_type, net_architecture_name), fromlist=[net_architecture_name])
        class_ = getattr(module, net_architecture_name)
        if model_type == 'extractor':
            model = class_(rgb)
        elif model_type == 'estimator':
            model = class_()
        else:
            raise NotImplementedError('Trying to initialize unknown Network Architecture...')

        # Let's use more GPU!
        if use_gpu and torch.cuda.device_count() > 1:
            logger.info("Let's use %d GPUs!" % torch.cuda.device_count())
            model = torch.nn.DataParallel(model)
            torch.cuda.manual_seed(0)
            torch.cuda.manual_seed_all(0)        

        return model, rgb

    @staticmethod
    def load_parameters_into_model(model, model_absolute_path, use_gpu=True):
        model_filename = os.path.basename(os.path.normpath(model_absolute_path))
        logger.info("Loading model %s." % model_filename)
        # load params
        if use_gpu:
            state_dict = torch.load(model_absolute_path)
        else:
            state_dict = torch.load(model_absolute_path, map_location=lambda storage, loc: storage)

        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            logger.warn('Was not able to load state_dictionary, trying to add/remove "module."...')

            # create new OrderedDict that does not contain `module.`
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            # original saved file with DataParallel
            for k, v in state_dict.items():
                if torch.cuda.device_count() == 1 or not use_gpu:
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                else:
                    new_state_dict['module.' + k] = v

            model.load_state_dict(new_state_dict)

        logger.info("Model %s loaded." % model_filename)

        return model

    @staticmethod
    def load_model(model_absolute_path, model_type='extractor', use_gpu=True):
        model_filename = os.path.basename(os.path.normpath(model_absolute_path))

        # net_architecture_name = model_filename.split('-')[6].split('_')[0]
        net_architecture_name = model_filename.split('_')[2].split('=')[1]

        model, rgb = ModelLoader.initialize_model(net_architecture_name, model_type, use_gpu)

        model = ModelLoader.load_parameters_into_model(model, model_absolute_path, use_gpu)


        return model, rgb
