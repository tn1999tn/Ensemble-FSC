import argparse
from config import get_config
from config1 import get_config1
from config2 import get_config2
from config3 import get_config3
from config4 import get_config4
from config5 import get_config5
from config6 import get_config6
from config7 import get_config7
from config8 import get_config8
from config9 import get_config9
from config10 import get_config10
from config11 import get_config11
from config12 import get_config12
from config13 import get_config13
from config14 import get_config14
import os
from logger import create_logger
from data import create_torch_dataloader
from data.dataset_spec import Split
import torch
import numpy as np
import random
import json
from utils import accuracy, accuracy1, AverageMeter, delete_checkpoint, save_checkpoint, load_pretrained, \
    auto_resume_helper, load_checkpoint
import torch
import datetime
from models import get_model
from optimizer import build_optimizer, build_scheduler
import time
import copy
import math
from torch.utils.tensorboard import SummaryWriter
import collections


def setup_seed(seed):
    """
    Fix some seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_seed1(seed):
    """
    Fix some seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_seed2(seed):
    """
    Fix some seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_seed3(seed):
    """
    Fix some seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_seed4(seed):
    """
    Fix some seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_seed5(seed):
    """
    Fix some seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_seed6(seed):
    """
    Fix some seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_seed7(seed):
    """
    Fix some seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_seed8(seed):
    """
    Fix some seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_seed9(seed):
    """
    Fix some seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_seed10(seed):
    """
    Fix some seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_seed11(seed):
    """
    Fix some seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_seed12(seed):
    """
    Fix some seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_seed13(seed):
    """
    Fix some seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_seed14(seed):
    """
    Fix some seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_option():
    parser = argparse.ArgumentParser('Meta-Dataset Pytorch implementation', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--train_batch_size', type=int, help="training batch size for single GPU")
    parser.add_argument('--valid_batch_size', type=int, help="validation batch size for single GPU")
    parser.add_argument('--test_batch_size', type=int, help="test batch size for single GPU")
    parser.add_argument('--output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--is_train', type=int, choices=[0, 1], help="training or testing")
    parser.add_argument('--pretrained', type=str, help="pretrained path")
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--resume', help='resume path')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    parser1 = argparse.ArgumentParser('Meta-Dataset Pytorch implementation', add_help=False)
    parser1.add_argument('--cfg1', type=str, required=True, metavar="FILE", help='path to config file', )
    parser1.add_argument(
        "--opts1",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser1.add_argument('--train_batch_size1', type=int, help="training batch size for single GPU")
    parser1.add_argument('--valid_batch_size1', type=int, help="validation batch size for single GPU")
    parser1.add_argument('--test_batch_size1', type=int, help="test batch size for single GPU")
    parser1.add_argument('--output1', type=str, metavar='PATH',
                         help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser1.add_argument('--is_train1', type=int, choices=[0, 1], help="training or testing")
    parser1.add_argument('--pretrained1', type=str, help="pretrained path")
    parser1.add_argument('--tag1', help='tag of experiment')
    parser1.add_argument('--resume1', help='resume path')

    args1, unparsed1 = parser1.parse_known_args()
    config1 = get_config1(args1)

    parser2 = argparse.ArgumentParser('Meta-Dataset Pytorch implementation', add_help=False)
    parser2.add_argument('--cfg2', type=str, required=True, metavar="FILE", help='path to config file', )
    parser2.add_argument(
        "--opts2",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser2.add_argument('--train_batch_size2', type=int, help="training batch size for single GPU")
    parser2.add_argument('--valid_batch_size2', type=int, help="validation batch size for single GPU")
    parser2.add_argument('--test_batch_size2', type=int, help="test batch size for single GPU")
    parser2.add_argument('--output2', type=str, metavar='PATH',
                         help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser2.add_argument('--is_train2', type=int, choices=[0, 1], help="training or testing")
    parser2.add_argument('--pretrained2', type=str, help="pretrained path")
    parser2.add_argument('--tag2', help='tag of experiment')
    parser2.add_argument('--resume2', help='resume path')

    args2, unparsed2 = parser2.parse_known_args()
    config2 = get_config2(args2)

    parser3 = argparse.ArgumentParser('Meta-Dataset Pytorch implementation', add_help=False)
    parser3.add_argument('--cfg3', type=str, required=True, metavar="FILE", help='path to config file', )
    parser3.add_argument(
        "--opts3",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser3.add_argument('--train_batch_size3', type=int, help="training batch size for single GPU")
    parser3.add_argument('--valid_batch_size3', type=int, help="validation batch size for single GPU")
    parser3.add_argument('--test_batch_size3', type=int, help="test batch size for single GPU")
    parser3.add_argument('--output3', type=str, metavar='PATH',
                         help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser3.add_argument('--is_train3', type=int, choices=[0, 1], help="training or testing")
    parser3.add_argument('--pretrained3', type=str, help="pretrained path")
    parser3.add_argument('--tag3', help='tag of experiment')
    parser3.add_argument('--resume3', help='resume path')

    args3, unparsed3 = parser3.parse_known_args()
    config3 = get_config3(args3)

    parser4 = argparse.ArgumentParser('Meta-Dataset Pytorch implementation', add_help=False)
    parser4.add_argument('--cfg4', type=str, required=True, metavar="FILE", help='path to config file', )
    parser4.add_argument(
        "--opts4",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser4.add_argument('--train_batch_size4', type=int, help="training batch size for single GPU")
    parser4.add_argument('--valid_batch_size4', type=int, help="validation batch size for single GPU")
    parser4.add_argument('--test_batch_size4', type=int, help="test batch size for single GPU")
    parser4.add_argument('--output4', type=str, metavar='PATH',
                         help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser4.add_argument('--is_train4', type=int, choices=[0, 1], help="training or testing")
    parser4.add_argument('--pretrained4', type=str, help="pretrained path")
    parser4.add_argument('--tag4', help='tag of experiment')
    parser4.add_argument('--resume4', help='resume path')

    args4, unparsed4 = parser4.parse_known_args()
    config4 = get_config4(args4)

    parser5 = argparse.ArgumentParser('Meta-Dataset Pytorch implementation', add_help=False)
    parser5.add_argument('--cfg5', type=str, required=True, metavar="FILE", help='path to config file', )
    parser5.add_argument(
        "--opts5",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser5.add_argument('--train_batch_size5', type=int, help="training batch size for single GPU")
    parser5.add_argument('--valid_batch_size5', type=int, help="validation batch size for single GPU")
    parser5.add_argument('--test_batch_size5', type=int, help="test batch size for single GPU")
    parser5.add_argument('--output5', type=str, metavar='PATH',
                         help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser5.add_argument('--is_train5', type=int, choices=[0, 1], help="training or testing")
    parser5.add_argument('--pretrained5', type=str, help="pretrained path")
    parser5.add_argument('--tag5', help='tag of experiment')
    parser5.add_argument('--resume5', help='resume path')

    args5, unparsed5 = parser5.parse_known_args()
    config5 = get_config5(args5)

    parser6 = argparse.ArgumentParser('Meta-Dataset Pytorch implementation', add_help=False)
    parser6.add_argument('--cfg6', type=str, required=True, metavar="FILE", help='path to config file', )
    parser6.add_argument(
        "--opts6",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser6.add_argument('--train_batch_size6', type=int, help="training batch size for single GPU")
    parser6.add_argument('--valid_batch_size6', type=int, help="validation batch size for single GPU")
    parser6.add_argument('--test_batch_size6', type=int, help="test batch size for single GPU")
    parser6.add_argument('--output6', type=str, metavar='PATH',
                         help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser6.add_argument('--is_train6', type=int, choices=[0, 1], help="training or testing")
    parser6.add_argument('--pretrained6', type=str, help="pretrained path")
    parser6.add_argument('--tag6', help='tag of experiment')
    parser6.add_argument('--resume6', help='resume path')

    args6, unparsed6 = parser6.parse_known_args()
    config6 = get_config6(args6)

    parser7 = argparse.ArgumentParser('Meta-Dataset Pytorch implementation', add_help=False)
    parser7.add_argument('--cfg7', type=str, required=True, metavar="FILE", help='path to config file', )
    parser7.add_argument(
        "--opts7",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser7.add_argument('--train_batch_size7', type=int, help="training batch size for single GPU")
    parser7.add_argument('--valid_batch_size7', type=int, help="validation batch size for single GPU")
    parser7.add_argument('--test_batch_size7', type=int, help="test batch size for single GPU")
    parser7.add_argument('--output7', type=str, metavar='PATH',
                         help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser7.add_argument('--is_train7', type=int, choices=[0, 1], help="training or testing")
    parser7.add_argument('--pretrained7', type=str, help="pretrained path")
    parser7.add_argument('--tag7', help='tag of experiment')
    parser7.add_argument('--resume7', help='resume path')

    args7, unparsed7 = parser7.parse_known_args()
    config7 = get_config7(args7)

    parser8 = argparse.ArgumentParser('Meta-Dataset Pytorch implementation', add_help=False)
    parser8.add_argument('--cfg8', type=str, required=True, metavar="FILE", help='path to config file', )
    parser8.add_argument(
        "--opts8",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser8.add_argument('--train_batch_size8', type=int, help="training batch size for single GPU")
    parser8.add_argument('--valid_batch_size8', type=int, help="validation batch size for single GPU")
    parser8.add_argument('--test_batch_size8', type=int, help="test batch size for single GPU")
    parser8.add_argument('--output8', type=str, metavar='PATH',
                         help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser8.add_argument('--is_train8', type=int, choices=[0, 1], help="training or testing")
    parser8.add_argument('--pretrained8', type=str, help="pretrained path")
    parser8.add_argument('--tag8', help='tag of experiment')
    parser8.add_argument('--resume8', help='resume path')

    args8, unparsed8 = parser8.parse_known_args()
    config8 = get_config8(args8)

    parser9 = argparse.ArgumentParser('Meta-Dataset Pytorch implementation', add_help=False)
    parser9.add_argument('--cfg9', type=str, required=True, metavar="FILE", help='path to config file', )
    parser9.add_argument(
        "--opts9",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser9.add_argument('--train_batch_size9', type=int, help="training batch size for single GPU")
    parser9.add_argument('--valid_batch_size9', type=int, help="validation batch size for single GPU")
    parser9.add_argument('--test_batch_size9', type=int, help="test batch size for single GPU")
    parser9.add_argument('--output9', type=str, metavar='PATH',
                         help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser9.add_argument('--is_train9', type=int, choices=[0, 1], help="training or testing")
    parser9.add_argument('--pretrained9', type=str, help="pretrained path")
    parser9.add_argument('--tag9', help='tag of experiment')
    parser9.add_argument('--resume9', help='resume path')

    args9, unparsed9 = parser9.parse_known_args()
    config9 = get_config9(args9)

    parser10 = argparse.ArgumentParser('Meta-Dataset Pytorch implementation', add_help=False)
    parser10.add_argument('--cfg10', type=str, required=True, metavar="FILE", help='path to config file', )
    parser10.add_argument(
        "--opts10",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser10.add_argument('--train_batch_size10', type=int, help="training batch size for single GPU")
    parser10.add_argument('--valid_batch_size10', type=int, help="validation batch size for single GPU")
    parser10.add_argument('--test_batch_size10', type=int, help="test batch size for single GPU")
    parser10.add_argument('--output10', type=str, metavar='PATH',
                          help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser10.add_argument('--is_train10', type=int, choices=[0, 1], help="training or testing")
    parser10.add_argument('--pretrained10', type=str, help="pretrained path")
    parser10.add_argument('--tag10', help='tag of experiment')
    parser10.add_argument('--resume10', help='resume path')

    args10, unparsed10 = parser10.parse_known_args()
    config10 = get_config10(args10)

    parser11 = argparse.ArgumentParser('Meta-Dataset Pytorch implementation', add_help=False)
    parser11.add_argument('--cfg11', type=str, required=True, metavar="FILE", help='path to config file', )
    parser11.add_argument(
        "--opts11",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser11.add_argument('--train_batch_size11', type=int, help="training batch size for single GPU")
    parser11.add_argument('--valid_batch_size11', type=int, help="validation batch size for single GPU")
    parser11.add_argument('--test_batch_size11', type=int, help="test batch size for single GPU")
    parser11.add_argument('--output11', type=str, metavar='PATH',
                          help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser11.add_argument('--is_train11', type=int, choices=[0, 1], help="training or testing")
    parser11.add_argument('--pretrained11', type=str, help="pretrained path")
    parser11.add_argument('--tag11', help='tag of experiment')
    parser11.add_argument('--resume11', help='resume path')

    args11, unparsed11 = parser11.parse_known_args()
    config11 = get_config11(args11)

    parser12 = argparse.ArgumentParser('Meta-Dataset Pytorch implementation', add_help=False)
    parser12.add_argument('--cfg12', type=str, required=True, metavar="FILE", help='path to config file', )
    parser12.add_argument(
        "--opts12",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser12.add_argument('--train_batch_size12', type=int, help="training batch size for single GPU")
    parser12.add_argument('--valid_batch_size12', type=int, help="validation batch size for single GPU")
    parser12.add_argument('--test_batch_size12', type=int, help="test batch size for single GPU")
    parser12.add_argument('--output12', type=str, metavar='PATH',
                          help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser12.add_argument('--is_train12', type=int, choices=[0, 1], help="training or testing")
    parser12.add_argument('--pretrained12', type=str, help="pretrained path")
    parser12.add_argument('--tag12', help='tag of experiment')
    parser12.add_argument('--resume12', help='resume path')

    args12, unparsed12 = parser12.parse_known_args()
    config12 = get_config12(args12)

    parser13 = argparse.ArgumentParser('Meta-Dataset Pytorch implementation', add_help=False)
    parser13.add_argument('--cfg13', type=str, required=True, metavar="FILE", help='path to config file', )
    parser13.add_argument(
        "--opts13",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser13.add_argument('--train_batch_size13', type=int, help="training batch size for single GPU")
    parser13.add_argument('--valid_batch_size13', type=int, help="validation batch size for single GPU")
    parser13.add_argument('--test_batch_size13', type=int, help="test batch size for single GPU")
    parser13.add_argument('--output13', type=str, metavar='PATH',
                          help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser13.add_argument('--is_train13', type=int, choices=[0, 1], help="training or testing")
    parser13.add_argument('--pretrained13', type=str, help="pretrained path")
    parser13.add_argument('--tag13', help='tag of experiment')
    parser13.add_argument('--resume13', help='resume path')

    args13, unparsed13 = parser13.parse_known_args()
    config13 = get_config13(args13)

    parser14 = argparse.ArgumentParser('Meta-Dataset Pytorch implementation', add_help=False)
    parser14.add_argument('--cfg14', type=str, required=True, metavar="FILE", help='path to config file', )
    parser14.add_argument(
        "--opts14",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser14.add_argument('--train_batch_size14', type=int, help="training batch size for single GPU")
    parser14.add_argument('--valid_batch_size14', type=int, help="validation batch size for single GPU")
    parser14.add_argument('--test_batch_size14', type=int, help="test batch size for single GPU")
    parser14.add_argument('--output14', type=str, metavar='PATH',
                          help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser14.add_argument('--is_train14', type=int, choices=[0, 1], help="training or testing")
    parser14.add_argument('--pretrained14', type=str, help="pretrained path")
    parser14.add_argument('--tag14', help='tag of experiment')
    parser14.add_argument('--resume14', help='resume path')

    args14, unparsed14 = parser14.parse_known_args()
    config14 = get_config14(args14)

    return args, config, config1, config2, config3, config4, config5, config6, config7, config8, config9, config10, config11, config12, config13, config14


def test(config, config1, config2, config3, config4, config5, config6, config7, config8, config9, config10, config11,
         config12, config13, config14):
    test_dataloader, test_dataset = create_torch_dataloader(Split.TEST, config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    logger1.info(f"Creating model:{config1.MODEL.TYPE}/{config1.MODEL.NAME}")
    logger2.info(f"Creating model:{config2.MODEL.TYPE}/{config2.MODEL.NAME}")
    logger3.info(f"Creating model:{config3.MODEL.TYPE}/{config3.MODEL.NAME}")
    logger4.info(f"Creating model:{config4.MODEL.TYPE}/{config4.MODEL.NAME}")
    logger5.info(f"Creating model:{config5.MODEL.TYPE}/{config5.MODEL.NAME}")
    logger6.info(f"Creating model:{config6.MODEL.TYPE}/{config6.MODEL.NAME}")
    logger7.info(f"Creating model:{config7.MODEL.TYPE}/{config7.MODEL.NAME}")
    logger8.info(f"Creating model:{config8.MODEL.TYPE}/{config8.MODEL.NAME}")
    logger9.info(f"Creating model:{config9.MODEL.TYPE}/{config9.MODEL.NAME}")
    logger10.info(f"Creating model:{config10.MODEL.TYPE}/{config10.MODEL.NAME}")
    logger11.info(f"Creating model:{config11.MODEL.TYPE}/{config11.MODEL.NAME}")
    logger12.info(f"Creating model:{config12.MODEL.TYPE}/{config12.MODEL.NAME}")
    logger13.info(f"Creating model:{config13.MODEL.TYPE}/{config13.MODEL.NAME}")
    logger14.info(f"Creating model:{config14.MODEL.TYPE}/{config14.MODEL.NAME}")

    model = get_model(config).cuda()
    model1 = get_model(config1).cuda()
    model2 = get_model(config2).cuda()
    model3 = get_model(config3).cuda()
    model4 = get_model(config4).cuda()
    model5 = get_model(config5).cuda()
    model6 = get_model(config6).cuda()
    model7 = get_model(config7).cuda()
    model8 = get_model(config8).cuda()
    model9 = get_model(config9).cuda()
    model10 = get_model(config10).cuda()
    model11 = get_model(config11).cuda()
    model12 = get_model(config12).cuda()
    model13 = get_model(config13).cuda()
    model14 = get_model(config14).cuda()

    if config.MODEL.PRETRAINED:
        load_pretrained(config, model, logger)

    if config1.MODEL.PRETRAINED:
        load_pretrained(config1, model1, logger1)

    if config2.MODEL.PRETRAINED:
        load_pretrained(config2, model2, logger2)

    if config3.MODEL.PRETRAINED:
        load_pretrained(config3, model3, logger3)

    if config4.MODEL.PRETRAINED:
        load_pretrained(config4, model4, logger4)

    if config5.MODEL.PRETRAINED:
        load_pretrained(config5, model5, logger5)

    if config6.MODEL.PRETRAINED:
        load_pretrained(config6, model6, logger6)

    if config7.MODEL.PRETRAINED:
        load_pretrained(config7, model7, logger7)

    if config8.MODEL.PRETRAINED:
        load_pretrained(config8, model8, logger8)

    if config9.MODEL.PRETRAINED:
        load_pretrained(config9, model9, logger9)

    if config10.MODEL.PRETRAINED:
        load_pretrained(config10, model10, logger10)

    if config11.MODEL.PRETRAINED:
        load_pretrained(config11, model11, logger11)

    if config12.MODEL.PRETRAINED:
        load_pretrained(config12, model12, logger12)

    if config13.MODEL.PRETRAINED:
        load_pretrained(config13, model13, logger13)

    if config14.MODEL.PRETRAINED:
        load_pretrained(config14, model14, logger14)

    # if model has adapters like in TSA
    if hasattr(model, 'mode') and model.mode == "NCC":
        model.append_adapter()
    if hasattr(model1, 'mode') and model1.mode == "NCC":
        model1.append_adapter()
    if hasattr(model2, 'mode') and model2.mode == "NCC":
        model2.append_adapter()
    if hasattr(model3, 'mode') and model3.mode == "NCC":
        model3.append_adapter()
    if hasattr(model4, 'mode') and model4.mode == "NCC":
        model4.append_adapter()
    if hasattr(model5, 'mode') and model5.mode == "NCC":
        model5.append_adapter()
    if hasattr(model6, 'mode') and model6.mode == "NCC":
        model6.append_adapter()
    if hasattr(model7, 'mode') and model7.mode == "NCC":
        model7.append_adapter()
    if hasattr(model8, 'mode') and model8.mode == "NCC":
        model8.append_adapter()
    if hasattr(model9, 'mode') and model9.mode == "NCC":
        model9.append_adapter()
    if hasattr(model10, 'mode') and model10.mode == "NCC":
        model10.append_adapter()
    if hasattr(model11, 'mode') and model11.mode == "NCC":
        model11.append_adapter()
    if hasattr(model12, 'mode') and model12.mode == "NCC":
        model12.append_adapter()
    if hasattr(model13, 'mode') and model13.mode == "NCC":
        model13.append_adapter()
    if hasattr(model14, 'mode') and model14.mode == "NCC":
        model14.append_adapter()

    logger.info("Start testing")

    with torch.no_grad():
        acc, ci = testing(config, test_dataset, test_dataloader, model, model1, model2, model3, model4, model5, model6,
                          model7, model8, model9, model10, model11, model12, model13, model14)
    logger.info(f"Test Accuracy of {config.DATA.TEST.DATASET_NAMES[0]}: {acc:.2f}%+-{ci:.2f}")
    # logging testing results in config.OUTPUT/results.json
    path = os.path.join(config.OUTPUT, "results.json")

    if os.path.exists(path):
        with open(path, 'r') as f:
            result_dic = json.load(f)
    else:
        result_dic = {}

    # by default, we assume there is only one dataset to be tested at a time.
    # result_dic[f"{config.DATA.TEST.DATASET_NAMES[0]}"]=[acc, ci]
    result_dic[f"{config.DATA.TEST.DATASET_NAMES[0]}"] = [acc]

    with open(path, 'w') as f:
        json.dump(result_dic, f)


@torch.no_grad()
def testing(config, dataset, data_loader, model, model1, model2, model3, model4, model5, model6, model7, model8, model9,
            model10, model11, model12, model13, model14):
    model.eval()
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    model6.eval()
    model7.eval()
    model8.eval()
    model9.eval()
    model10.eval()
    model11.eval()
    model12.eval()
    model13.eval()
    model14.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    end = time.time()

    dataset.set_epoch()
    acc_ci = []
    for idx, batches in enumerate(data_loader):
        dataset_index, imgs, labels = batches

        models = [model, model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, model11,
                  model12, model13, model14]

        for i, img in enumerate(imgs):
            img["support"] = img["support"].squeeze_().cuda()
            img["query"] = img["query"].squeeze_().cuda()
            labels[i]["support"] = labels[i]["support"].squeeze_().cuda()
            labels[i]["query"] = labels[i]["query"].squeeze_().cuda()

        ori_labels_no0 = copy.deepcopy(labels)
        ori_imgs_no0 = copy.deepcopy(imgs)
        ori_labels_no1 = copy.deepcopy(labels)
        ori_imgs_no1 = copy.deepcopy(imgs)
        ori_labels_no2 = copy.deepcopy(labels)
        ori_imgs_no2 = copy.deepcopy(imgs)
        ori_labels_no3 = copy.deepcopy(labels)
        ori_imgs_no3 = copy.deepcopy(imgs)
        ori_labels_no4 = copy.deepcopy(labels)
        ori_imgs_no4 = copy.deepcopy(imgs)
        ori_labels_no5 = copy.deepcopy(labels)
        ori_imgs_no5 = copy.deepcopy(imgs)
        ori_labels_no6 = copy.deepcopy(labels)
        ori_imgs_no6 = copy.deepcopy(imgs)
        ori_labels_no7 = copy.deepcopy(labels)
        ori_imgs_no7 = copy.deepcopy(imgs)
        ori_labels_no8 = copy.deepcopy(labels)
        ori_imgs_no8 = copy.deepcopy(imgs)
        ori_labels_no9 = copy.deepcopy(labels)
        ori_imgs_no9 = copy.deepcopy(imgs)
        ori_labels_no10 = copy.deepcopy(labels)
        ori_imgs_no10 = copy.deepcopy(imgs)
        ori_labels_no11 = copy.deepcopy(labels)
        ori_imgs_no11 = copy.deepcopy(imgs)
        ori_labels_no12 = copy.deepcopy(labels)
        ori_imgs_no12 = copy.deepcopy(imgs)
        ori_labels_no13 = copy.deepcopy(labels)
        ori_imgs_no13 = copy.deepcopy(imgs)
        ori_labels_no14 = copy.deepcopy(labels)
        ori_imgs_no14 = copy.deepcopy(imgs)

        all_pre_no0 = []
        all_pre_no1 = []
        all_pre_no2 = []
        all_pre_no3 = []
        all_pre_no4 = []
        all_pre_no5 = []
        all_pre_no6 = []
        all_pre_no7 = []
        all_pre_no8 = []
        all_pre_no9 = []
        all_pre_no10 = []
        all_pre_no11 = []
        all_pre_no12 = []
        all_pre_no13 = []
        all_pre_no14 = []

        score_0 = []
        all_pre = []
        for i, model in enumerate(models):
            score_no0 = model.test_forward(imgs, labels, dataset_index)
            score_no0 = torch.stack(score_no0)
            score_no0 = torch.softmax(score_no0, dim=2)
            score_0.append(score_no0)
            max_indices = torch.argmax(score_no0, dim=2)
            max_indices = max_indices.view(-1)
            # print(max_indices)
            all_pre.append(max_indices)
            if i != 0:
                max_indices_no0 = torch.argmax(score_no0, dim=2)
                max_indices_no0 = max_indices_no0.view(-1)
                # print(max_indices)
                all_pre_no0.append(max_indices_no0)

            if i != 1:
                max_indices_no1 = torch.argmax(score_no0, dim=2)
                max_indices_no1 = max_indices_no1.view(-1)
                # print(max_indices)
                all_pre_no1.append(max_indices_no1)

            if i != 2:
                max_indices_no2 = torch.argmax(score_no0, dim=2)
                max_indices_no2 = max_indices_no2.view(-1)
                # print(max_indices)
                all_pre_no2.append(max_indices_no2)

            if i != 3:
                max_indices_no3 = torch.argmax(score_no0, dim=2)
                max_indices_no3 = max_indices_no3.view(-1)
                # print(max_indices)
                all_pre_no3.append(max_indices_no3)

            if i != 4:
                max_indices_no4 = torch.argmax(score_no0, dim=2)
                max_indices_no4 = max_indices_no4.view(-1)
                # print(max_indices)
                all_pre_no4.append(max_indices_no4)

            if i != 5:
                max_indices_no5 = torch.argmax(score_no0, dim=2)
                max_indices_no5 = max_indices_no5.view(-1)
                # print(max_indices)
                all_pre_no5.append(max_indices_no5)

            if i != 6:
                max_indices_no6 = torch.argmax(score_no0, dim=2)
                max_indices_no6 = max_indices_no6.view(-1)
                # print(max_indices)
                all_pre_no6.append(max_indices_no6)

            if i != 7:
                max_indices_no7 = torch.argmax(score_no0, dim=2)
                max_indices_no7 = max_indices_no7.view(-1)
                # print(max_indices)
                all_pre_no7.append(max_indices_no7)

            if i != 8:
                max_indices_no8 = torch.argmax(score_no0, dim=2)
                max_indices_no8 = max_indices_no8.view(-1)
                # print(max_indices)
                all_pre_no8.append(max_indices_no8)

            if i != 9:
                max_indices_no9 = torch.argmax(score_no0, dim=2)
                max_indices_no9 = max_indices_no9.view(-1)
                # print(max_indices)
                all_pre_no9.append(max_indices_no9)

            if i != 10:
                max_indices_no10 = torch.argmax(score_no0, dim=2)
                max_indices_no10 = max_indices_no10.view(-1)
                # print(max_indices)
                all_pre_no10.append(max_indices_no10)

            if i != 11:
                max_indices_no11 = torch.argmax(score_no0, dim=2)
                max_indices_no11 = max_indices_no11.view(-1)
                # print(max_indices)
                all_pre_no11.append(max_indices_no11)

            if i != 12:
                max_indices_no12 = torch.argmax(score_no0, dim=2)
                max_indices_no12 = max_indices_no12.view(-1)
                # print(max_indices)
                all_pre_no12.append(max_indices_no12)

            if i != 13:
                max_indices_no13 = torch.argmax(score_no0, dim=2)
                max_indices_no13 = max_indices_no13.view(-1)
                # print(max_indices)
                all_pre_no13.append(max_indices_no13)

            if i != 14:
                max_indices_no14 = torch.argmax(score_no0, dim=2)
                max_indices_no14 = max_indices_no14.view(-1)
                # print(max_indices)
                all_pre_no14.append(max_indices_no14)

        all_pre = torch.stack(all_pre)
        all_pre_no0 = torch.stack(all_pre_no0)

        no0_counts_0 = torch.sum(torch.eq(all_pre_no0, 0), dim=0)
        no0_counts_1 = torch.sum(torch.eq(all_pre_no0, 1), dim=0)
        no0_counts_2 = torch.sum(torch.eq(all_pre_no0, 2), dim=0)
        no0_counts_3 = torch.sum(torch.eq(all_pre_no0, 3), dim=0)
        no0_counts_4 = torch.sum(torch.eq(all_pre_no0, 4), dim=0)

        no0_indices_0 = torch.where(no0_counts_0 >= 10)[0]
        no0_indices_1 = torch.where(no0_counts_1 >= 10)[0]
        no0_indices_2 = torch.where(no0_counts_2 >= 10)[0]
        no0_indices_3 = torch.where(no0_counts_3 >= 10)[0]
        no0_indices_4 = torch.where(no0_counts_4 >= 10)[0]

        for i in no0_indices_0:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no0[j]["support"] = torch.cat((ori_labels_no0[j]["support"], torch.tensor([0]).cuda()))
            ori_imgs_no0[j]["support"] = torch.cat(
                (ori_imgs_no0[j]["support"], ori_imgs_no0[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no0_indices_1:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no0[j]["support"] = torch.cat((ori_labels_no0[j]["support"], torch.tensor([1]).cuda()))
            ori_imgs_no0[j]["support"] = torch.cat(
                (ori_imgs_no0[j]["support"], ori_imgs_no0[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no0_indices_2:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no0[j]["support"] = torch.cat((ori_labels_no0[j]["support"], torch.tensor([2]).cuda()))
            ori_imgs_no0[j]["support"] = torch.cat(
                (ori_imgs_no0[j]["support"], ori_imgs_no0[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no0_indices_3:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no0[j]["support"] = torch.cat((ori_labels_no0[j]["support"], torch.tensor([3]).cuda()))
            ori_imgs_no0[j]["support"] = torch.cat(
                (ori_imgs_no0[j]["support"], ori_imgs_no0[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no0_indices_4:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no0[j]["support"] = torch.cat((ori_labels_no0[j]["support"], torch.tensor([4]).cuda()))
            ori_imgs_no0[j]["support"] = torch.cat(
                (ori_imgs_no0[j]["support"], ori_imgs_no0[j]["query"][k].unsqueeze(0)), dim=0)

        all_pre_no1 = torch.stack(all_pre_no1)
        no1_counts_0 = torch.sum(torch.eq(all_pre_no1, 0), dim=0)
        no1_counts_1 = torch.sum(torch.eq(all_pre_no1, 1), dim=0)
        no1_counts_2 = torch.sum(torch.eq(all_pre_no1, 2), dim=0)
        no1_counts_3 = torch.sum(torch.eq(all_pre_no1, 3), dim=0)
        no1_counts_4 = torch.sum(torch.eq(all_pre_no1, 4), dim=0)

        no1_indices_0 = torch.where(no1_counts_0 >= 10)[0]
        no1_indices_1 = torch.where(no1_counts_1 >= 10)[0]
        no1_indices_2 = torch.where(no1_counts_2 >= 10)[0]
        no1_indices_3 = torch.where(no1_counts_3 >= 10)[0]
        no1_indices_4 = torch.where(no1_counts_4 >= 10)[0]

        for i in no1_indices_0:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no1[j]["support"] = torch.cat((ori_labels_no1[j]["support"], torch.tensor([0]).cuda()))
            ori_imgs_no1[j]["support"] = torch.cat(
                (ori_imgs_no1[j]["support"], ori_imgs_no1[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no1_indices_1:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no1[j]["support"] = torch.cat((ori_labels_no1[j]["support"], torch.tensor([1]).cuda()))
            ori_imgs_no1[j]["support"] = torch.cat(
                (ori_imgs_no1[j]["support"], ori_imgs_no1[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no1_indices_2:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no1[j]["support"] = torch.cat((ori_labels_no1[j]["support"], torch.tensor([2]).cuda()))
            ori_imgs_no1[j]["support"] = torch.cat(
                (ori_imgs_no1[j]["support"], ori_imgs_no1[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no1_indices_3:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no1[j]["support"] = torch.cat((ori_labels_no1[j]["support"], torch.tensor([3]).cuda()))
            ori_imgs_no1[j]["support"] = torch.cat(
                (ori_imgs_no1[j]["support"], ori_imgs_no1[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no1_indices_4:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no1[j]["support"] = torch.cat((ori_labels_no1[j]["support"], torch.tensor([4]).cuda()))
            ori_imgs_no1[j]["support"] = torch.cat(
                (ori_imgs_no1[j]["support"], ori_imgs_no1[j]["query"][k].unsqueeze(0)), dim=0)
        all_pre_no2 = torch.stack(all_pre_no2)
        no2_counts_0 = torch.sum(torch.eq(all_pre_no2, 0), dim=0)
        no2_counts_1 = torch.sum(torch.eq(all_pre_no2, 1), dim=0)
        no2_counts_2 = torch.sum(torch.eq(all_pre_no2, 2), dim=0)
        no2_counts_3 = torch.sum(torch.eq(all_pre_no2, 3), dim=0)
        no2_counts_4 = torch.sum(torch.eq(all_pre_no2, 4), dim=0)

        no2_indices_0 = torch.where(no2_counts_0 >= 10)[0]
        no2_indices_1 = torch.where(no2_counts_1 >= 10)[0]
        no2_indices_2 = torch.where(no2_counts_2 >= 10)[0]
        no2_indices_3 = torch.where(no2_counts_3 >= 10)[0]
        no2_indices_4 = torch.where(no2_counts_4 >= 10)[0]

        for i in no2_indices_0:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no2[j]["support"] = torch.cat((ori_labels_no2[j]["support"], torch.tensor([0]).cuda()))
            ori_imgs_no2[j]["support"] = torch.cat(
                (ori_imgs_no2[j]["support"], ori_imgs_no2[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no2_indices_1:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no2[j]["support"] = torch.cat((ori_labels_no2[j]["support"], torch.tensor([1]).cuda()))
            ori_imgs_no2[j]["support"] = torch.cat(
                (ori_imgs_no2[j]["support"], ori_imgs_no2[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no2_indices_2:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no2[j]["support"] = torch.cat((ori_labels_no2[j]["support"], torch.tensor([2]).cuda()))
            ori_imgs_no2[j]["support"] = torch.cat(
                (ori_imgs_no2[j]["support"], ori_imgs_no2[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no2_indices_3:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no2[j]["support"] = torch.cat((ori_labels_no2[j]["support"], torch.tensor([3]).cuda()))
            ori_imgs_no2[j]["support"] = torch.cat(
                (ori_imgs_no2[j]["support"], ori_imgs_no2[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no2_indices_4:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no2[j]["support"] = torch.cat((ori_labels_no2[j]["support"], torch.tensor([4]).cuda()))
            ori_imgs_no2[j]["support"] = torch.cat(
                (ori_imgs_no2[j]["support"], ori_imgs_no2[j]["query"][k].unsqueeze(0)), dim=0)

        all_pre_no3 = torch.stack(all_pre_no3)
        no3_counts_0 = torch.sum(torch.eq(all_pre_no3, 0), dim=0)
        no3_counts_1 = torch.sum(torch.eq(all_pre_no3, 1), dim=0)
        no3_counts_2 = torch.sum(torch.eq(all_pre_no3, 2), dim=0)
        no3_counts_3 = torch.sum(torch.eq(all_pre_no3, 3), dim=0)
        no3_counts_4 = torch.sum(torch.eq(all_pre_no3, 4), dim=0)

        no3_indices_0 = torch.where(no3_counts_0 >= 10)[0]
        no3_indices_1 = torch.where(no3_counts_1 >= 10)[0]
        no3_indices_2 = torch.where(no3_counts_2 >= 10)[0]
        no3_indices_3 = torch.where(no3_counts_3 >= 10)[0]
        no3_indices_4 = torch.where(no3_counts_4 >= 10)[0]

        for i in no3_indices_0:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no3[j]["support"] = torch.cat((ori_labels_no3[j]["support"], torch.tensor([0]).cuda()))
            ori_imgs_no3[j]["support"] = torch.cat(
                (ori_imgs_no3[j]["support"], ori_imgs_no3[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no3_indices_1:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no3[j]["support"] = torch.cat((ori_labels_no3[j]["support"], torch.tensor([1]).cuda()))
            ori_imgs_no3[j]["support"] = torch.cat(
                (ori_imgs_no3[j]["support"], ori_imgs_no3[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no3_indices_2:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no3[j]["support"] = torch.cat((ori_labels_no3[j]["support"], torch.tensor([2]).cuda()))
            ori_imgs_no3[j]["support"] = torch.cat(
                (ori_imgs_no3[j]["support"], ori_imgs_no3[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no3_indices_3:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no3[j]["support"] = torch.cat((ori_labels_no3[j]["support"], torch.tensor([3]).cuda()))
            ori_imgs_no3[j]["support"] = torch.cat(
                (ori_imgs_no3[j]["support"], ori_imgs_no3[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no3_indices_4:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no3[j]["support"] = torch.cat((ori_labels_no3[j]["support"], torch.tensor([4]).cuda()))
            ori_imgs_no3[j]["support"] = torch.cat(
                (ori_imgs_no3[j]["support"], ori_imgs_no3[j]["query"][k].unsqueeze(0)), dim=0)

        all_pre_no4 = torch.stack(all_pre_no4)
        no4_counts_0 = torch.sum(torch.eq(all_pre_no4, 0), dim=0)
        no4_counts_1 = torch.sum(torch.eq(all_pre_no4, 1), dim=0)
        no4_counts_2 = torch.sum(torch.eq(all_pre_no4, 2), dim=0)
        no4_counts_3 = torch.sum(torch.eq(all_pre_no4, 3), dim=0)
        no4_counts_4 = torch.sum(torch.eq(all_pre_no4, 4), dim=0)

        no4_indices_0 = torch.where(no4_counts_0 >= 10)[0]
        no4_indices_1 = torch.where(no4_counts_1 >= 10)[0]
        no4_indices_2 = torch.where(no4_counts_2 >= 10)[0]
        no4_indices_3 = torch.where(no4_counts_3 >= 10)[0]
        no4_indices_4 = torch.where(no4_counts_4 >= 10)[0]

        for i in no4_indices_0:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no4[j]["support"] = torch.cat((ori_labels_no4[j]["support"], torch.tensor([0]).cuda()))
            ori_imgs_no4[j]["support"] = torch.cat(
                (ori_imgs_no4[j]["support"], ori_imgs_no4[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no4_indices_1:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no4[j]["support"] = torch.cat((ori_labels_no4[j]["support"], torch.tensor([1]).cuda()))
            ori_imgs_no4[j]["support"] = torch.cat(
                (ori_imgs_no4[j]["support"], ori_imgs_no4[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no4_indices_2:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no4[j]["support"] = torch.cat((ori_labels_no4[j]["support"], torch.tensor([2]).cuda()))
            ori_imgs_no4[j]["support"] = torch.cat(
                (ori_imgs_no4[j]["support"], ori_imgs_no4[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no4_indices_3:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no4[j]["support"] = torch.cat((ori_labels_no4[j]["support"], torch.tensor([3]).cuda()))
            ori_imgs_no4[j]["support"] = torch.cat(
                (ori_imgs_no4[j]["support"], ori_imgs_no4[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no4_indices_4:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no4[j]["support"] = torch.cat((ori_labels_no4[j]["support"], torch.tensor([4]).cuda()))
            ori_imgs_no4[j]["support"] = torch.cat(
                (ori_imgs_no4[j]["support"], ori_imgs_no4[j]["query"][k].unsqueeze(0)), dim=0)

        all_pre_no5 = torch.stack(all_pre_no5)
        no5_counts_0 = torch.sum(torch.eq(all_pre_no5, 0), dim=0)
        no5_counts_1 = torch.sum(torch.eq(all_pre_no5, 1), dim=0)
        no5_counts_2 = torch.sum(torch.eq(all_pre_no5, 2), dim=0)
        no5_counts_3 = torch.sum(torch.eq(all_pre_no5, 3), dim=0)
        no5_counts_4 = torch.sum(torch.eq(all_pre_no5, 4), dim=0)

        no5_indices_0 = torch.where(no5_counts_0 >= 10)[0]
        no5_indices_1 = torch.where(no5_counts_1 >= 10)[0]
        no5_indices_2 = torch.where(no5_counts_2 >= 10)[0]
        no5_indices_3 = torch.where(no5_counts_3 >= 10)[0]
        no5_indices_4 = torch.where(no5_counts_4 >= 10)[0]

        for i in no5_indices_0:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no5[j]["support"] = torch.cat((ori_labels_no5[j]["support"], torch.tensor([0]).cuda()))
            ori_imgs_no5[j]["support"] = torch.cat(
                (ori_imgs_no5[j]["support"], ori_imgs_no5[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no5_indices_1:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no5[j]["support"] = torch.cat((ori_labels_no5[j]["support"], torch.tensor([1]).cuda()))
            ori_imgs_no5[j]["support"] = torch.cat(
                (ori_imgs_no5[j]["support"], ori_imgs_no5[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no5_indices_2:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no5[j]["support"] = torch.cat((ori_labels_no5[j]["support"], torch.tensor([2]).cuda()))
            ori_imgs_no5[j]["support"] = torch.cat(
                (ori_imgs_no5[j]["support"], ori_imgs_no5[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no5_indices_3:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no5[j]["support"] = torch.cat((ori_labels_no5[j]["support"], torch.tensor([3]).cuda()))
            ori_imgs_no5[j]["support"] = torch.cat(
                (ori_imgs_no5[j]["support"], ori_imgs_no5[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no5_indices_4:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no5[j]["support"] = torch.cat((ori_labels_no5[j]["support"], torch.tensor([4]).cuda()))
            ori_imgs_no5[j]["support"] = torch.cat(
                (ori_imgs_no5[j]["support"], ori_imgs_no5[j]["query"][k].unsqueeze(0)), dim=0)

        all_pre_no6 = torch.stack(all_pre_no6)
        no6_counts_0 = torch.sum(torch.eq(all_pre_no6, 0), dim=0)
        no6_counts_1 = torch.sum(torch.eq(all_pre_no6, 1), dim=0)
        no6_counts_2 = torch.sum(torch.eq(all_pre_no6, 2), dim=0)
        no6_counts_3 = torch.sum(torch.eq(all_pre_no6, 3), dim=0)
        no6_counts_4 = torch.sum(torch.eq(all_pre_no6, 4), dim=0)

        no6_indices_0 = torch.where(no6_counts_0 >= 10)[0]
        no6_indices_1 = torch.where(no6_counts_1 >= 10)[0]
        no6_indices_2 = torch.where(no6_counts_2 >= 10)[0]
        no6_indices_3 = torch.where(no6_counts_3 >= 10)[0]
        no6_indices_4 = torch.where(no6_counts_4 >= 10)[0]

        for i in no6_indices_0:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no6[j]["support"] = torch.cat((ori_labels_no6[j]["support"], torch.tensor([0]).cuda()))
            ori_imgs_no6[j]["support"] = torch.cat(
                (ori_imgs_no6[j]["support"], ori_imgs_no6[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no6_indices_1:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no6[j]["support"] = torch.cat((ori_labels_no6[j]["support"], torch.tensor([1]).cuda()))
            ori_imgs_no6[j]["support"] = torch.cat(
                (ori_imgs_no6[j]["support"], ori_imgs_no6[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no6_indices_2:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no6[j]["support"] = torch.cat((ori_labels_no6[j]["support"], torch.tensor([2]).cuda()))
            ori_imgs_no6[j]["support"] = torch.cat(
                (ori_imgs_no6[j]["support"], ori_imgs_no6[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no6_indices_3:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no6[j]["support"] = torch.cat((ori_labels_no6[j]["support"], torch.tensor([3]).cuda()))
            ori_imgs_no6[j]["support"] = torch.cat(
                (ori_imgs_no6[j]["support"], ori_imgs_no6[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no6_indices_4:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no6[j]["support"] = torch.cat((ori_labels_no6[j]["support"], torch.tensor([4]).cuda()))
            ori_imgs_no6[j]["support"] = torch.cat(
                (ori_imgs_no6[j]["support"], ori_imgs_no6[j]["query"][k].unsqueeze(0)), dim=0)

        all_pre_no7 = torch.stack(all_pre_no7)
        no7_counts_0 = torch.sum(torch.eq(all_pre_no7, 0), dim=0)
        no7_counts_1 = torch.sum(torch.eq(all_pre_no7, 1), dim=0)
        no7_counts_2 = torch.sum(torch.eq(all_pre_no7, 2), dim=0)
        no7_counts_3 = torch.sum(torch.eq(all_pre_no7, 3), dim=0)
        no7_counts_4 = torch.sum(torch.eq(all_pre_no7, 4), dim=0)

        no7_indices_0 = torch.where(no7_counts_0 >= 10)[0]
        no7_indices_1 = torch.where(no7_counts_1 >= 10)[0]
        no7_indices_2 = torch.where(no7_counts_2 >= 10)[0]
        no7_indices_3 = torch.where(no7_counts_3 >= 10)[0]
        no7_indices_4 = torch.where(no7_counts_4 >= 10)[0]

        for i in no7_indices_0:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no7[j]["support"] = torch.cat((ori_labels_no7[j]["support"], torch.tensor([0]).cuda()))
            ori_imgs_no7[j]["support"] = torch.cat(
                (ori_imgs_no7[j]["support"], ori_imgs_no7[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no7_indices_1:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no7[j]["support"] = torch.cat((ori_labels_no7[j]["support"], torch.tensor([1]).cuda()))
            ori_imgs_no7[j]["support"] = torch.cat(
                (ori_imgs_no7[j]["support"], ori_imgs_no7[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no7_indices_2:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no7[j]["support"] = torch.cat((ori_labels_no7[j]["support"], torch.tensor([2]).cuda()))
            ori_imgs_no7[j]["support"] = torch.cat(
                (ori_imgs_no7[j]["support"], ori_imgs_no7[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no7_indices_3:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no7[j]["support"] = torch.cat((ori_labels_no7[j]["support"], torch.tensor([3]).cuda()))
            ori_imgs_no7[j]["support"] = torch.cat(
                (ori_imgs_no7[j]["support"], ori_imgs_no7[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no7_indices_4:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no7[j]["support"] = torch.cat((ori_labels_no7[j]["support"], torch.tensor([4]).cuda()))
            ori_imgs_no7[j]["support"] = torch.cat(
                (ori_imgs_no7[j]["support"], ori_imgs_no7[j]["query"][k].unsqueeze(0)), dim=0)

        all_pre_no8 = torch.stack(all_pre_no8)
        no8_counts_0 = torch.sum(torch.eq(all_pre_no8, 0), dim=0)
        no8_counts_1 = torch.sum(torch.eq(all_pre_no8, 1), dim=0)
        no8_counts_2 = torch.sum(torch.eq(all_pre_no8, 2), dim=0)
        no8_counts_3 = torch.sum(torch.eq(all_pre_no8, 3), dim=0)
        no8_counts_4 = torch.sum(torch.eq(all_pre_no8, 4), dim=0)

        no8_indices_0 = torch.where(no8_counts_0 >= 10)[0]
        no8_indices_1 = torch.where(no8_counts_1 >= 10)[0]
        no8_indices_2 = torch.where(no8_counts_2 >= 10)[0]
        no8_indices_3 = torch.where(no8_counts_3 >= 10)[0]
        no8_indices_4 = torch.where(no8_counts_4 >= 10)[0]

        for i in no8_indices_0:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no8[j]["support"] = torch.cat((ori_labels_no8[j]["support"], torch.tensor([0]).cuda()))
            ori_imgs_no8[j]["support"] = torch.cat(
                (ori_imgs_no8[j]["support"], ori_imgs_no8[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no8_indices_1:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no8[j]["support"] = torch.cat((ori_labels_no8[j]["support"], torch.tensor([1]).cuda()))
            ori_imgs_no8[j]["support"] = torch.cat(
                (ori_imgs_no8[j]["support"], ori_imgs_no8[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no8_indices_2:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no8[j]["support"] = torch.cat((ori_labels_no8[j]["support"], torch.tensor([2]).cuda()))
            ori_imgs_no8[j]["support"] = torch.cat(
                (ori_imgs_no8[j]["support"], ori_imgs_no8[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no8_indices_3:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no8[j]["support"] = torch.cat((ori_labels_no8[j]["support"], torch.tensor([3]).cuda()))
            ori_imgs_no8[j]["support"] = torch.cat(
                (ori_imgs_no8[j]["support"], ori_imgs_no8[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no8_indices_4:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no8[j]["support"] = torch.cat((ori_labels_no8[j]["support"], torch.tensor([4]).cuda()))
            ori_imgs_no8[j]["support"] = torch.cat(
                (ori_imgs_no8[j]["support"], ori_imgs_no8[j]["query"][k].unsqueeze(0)), dim=0)

        all_pre_no9 = torch.stack(all_pre_no9)
        no9_counts_0 = torch.sum(torch.eq(all_pre_no9, 0), dim=0)
        no9_counts_1 = torch.sum(torch.eq(all_pre_no9, 1), dim=0)
        no9_counts_2 = torch.sum(torch.eq(all_pre_no9, 2), dim=0)
        no9_counts_3 = torch.sum(torch.eq(all_pre_no9, 3), dim=0)
        no9_counts_4 = torch.sum(torch.eq(all_pre_no9, 4), dim=0)

        no9_indices_0 = torch.where(no9_counts_0 >= 10)[0]
        no9_indices_1 = torch.where(no9_counts_1 >= 10)[0]
        no9_indices_2 = torch.where(no9_counts_2 >= 10)[0]
        no9_indices_3 = torch.where(no9_counts_3 >= 10)[0]
        no9_indices_4 = torch.where(no9_counts_4 >= 10)[0]

        for i in no9_indices_0:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no9[j]["support"] = torch.cat((ori_labels_no9[j]["support"], torch.tensor([0]).cuda()))
            ori_imgs_no9[j]["support"] = torch.cat(
                (ori_imgs_no9[j]["support"], ori_imgs_no9[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no9_indices_1:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no9[j]["support"] = torch.cat((ori_labels_no9[j]["support"], torch.tensor([1]).cuda()))
            ori_imgs_no9[j]["support"] = torch.cat(
                (ori_imgs_no9[j]["support"], ori_imgs_no9[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no9_indices_2:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no9[j]["support"] = torch.cat((ori_labels_no9[j]["support"], torch.tensor([2]).cuda()))
            ori_imgs_no9[j]["support"] = torch.cat(
                (ori_imgs_no9[j]["support"], ori_imgs_no9[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no9_indices_3:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no9[j]["support"] = torch.cat((ori_labels_no9[j]["support"], torch.tensor([3]).cuda()))
            ori_imgs_no9[j]["support"] = torch.cat(
                (ori_imgs_no9[j]["support"], ori_imgs_no9[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no9_indices_4:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no9[j]["support"] = torch.cat((ori_labels_no9[j]["support"], torch.tensor([4]).cuda()))
            ori_imgs_no9[j]["support"] = torch.cat(
                (ori_imgs_no9[j]["support"], ori_imgs_no9[j]["query"][k].unsqueeze(0)), dim=0)

        all_pre_no10 = torch.stack(all_pre_no10)
        no10_counts_0 = torch.sum(torch.eq(all_pre_no10, 0), dim=0)
        no10_counts_1 = torch.sum(torch.eq(all_pre_no10, 1), dim=0)
        no10_counts_2 = torch.sum(torch.eq(all_pre_no10, 2), dim=0)
        no10_counts_3 = torch.sum(torch.eq(all_pre_no10, 3), dim=0)
        no10_counts_4 = torch.sum(torch.eq(all_pre_no10, 4), dim=0)

        no10_indices_0 = torch.where(no10_counts_0 >= 10)[0]
        no10_indices_1 = torch.where(no10_counts_1 >= 10)[0]
        no10_indices_2 = torch.where(no10_counts_2 >= 10)[0]
        no10_indices_3 = torch.where(no10_counts_3 >= 10)[0]
        no10_indices_4 = torch.where(no10_counts_4 >= 10)[0]

        for i in no10_indices_0:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no10[j]["support"] = torch.cat((ori_labels_no10[j]["support"], torch.tensor([0]).cuda()))
            ori_imgs_no10[j]["support"] = torch.cat(
                (ori_imgs_no10[j]["support"], ori_imgs_no10[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no10_indices_1:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no10[j]["support"] = torch.cat((ori_labels_no10[j]["support"], torch.tensor([1]).cuda()))
            ori_imgs_no10[j]["support"] = torch.cat(
                (ori_imgs_no10[j]["support"], ori_imgs_no10[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no10_indices_2:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no10[j]["support"] = torch.cat((ori_labels_no10[j]["support"], torch.tensor([2]).cuda()))
            ori_imgs_no10[j]["support"] = torch.cat(
                (ori_imgs_no10[j]["support"], ori_imgs_no10[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no10_indices_3:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no10[j]["support"] = torch.cat((ori_labels_no10[j]["support"], torch.tensor([3]).cuda()))
            ori_imgs_no10[j]["support"] = torch.cat(
                (ori_imgs_no10[j]["support"], ori_imgs_no10[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no10_indices_4:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no10[j]["support"] = torch.cat((ori_labels_no10[j]["support"], torch.tensor([4]).cuda()))
            ori_imgs_no10[j]["support"] = torch.cat(
                (ori_imgs_no10[j]["support"], ori_imgs_no10[j]["query"][k].unsqueeze(0)), dim=0)

        all_pre_no11 = torch.stack(all_pre_no11)
        no11_counts_0 = torch.sum(torch.eq(all_pre_no11, 0), dim=0)
        no11_counts_1 = torch.sum(torch.eq(all_pre_no11, 1), dim=0)
        no11_counts_2 = torch.sum(torch.eq(all_pre_no11, 2), dim=0)
        no11_counts_3 = torch.sum(torch.eq(all_pre_no11, 3), dim=0)
        no11_counts_4 = torch.sum(torch.eq(all_pre_no11, 4), dim=0)

        no11_indices_0 = torch.where(no11_counts_0 >= 10)[0]
        no11_indices_1 = torch.where(no11_counts_1 >= 10)[0]
        no11_indices_2 = torch.where(no11_counts_2 >= 10)[0]
        no11_indices_3 = torch.where(no11_counts_3 >= 10)[0]
        no11_indices_4 = torch.where(no11_counts_4 >= 10)[0]

        for i in no11_indices_0:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no11[j]["support"] = torch.cat((ori_labels_no11[j]["support"], torch.tensor([0]).cuda()))
            ori_imgs_no11[j]["support"] = torch.cat(
                (ori_imgs_no11[j]["support"], ori_imgs_no11[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no11_indices_1:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no11[j]["support"] = torch.cat((ori_labels_no11[j]["support"], torch.tensor([1]).cuda()))
            ori_imgs_no11[j]["support"] = torch.cat(
                (ori_imgs_no11[j]["support"], ori_imgs_no11[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no11_indices_2:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no11[j]["support"] = torch.cat((ori_labels_no11[j]["support"], torch.tensor([2]).cuda()))
            ori_imgs_no11[j]["support"] = torch.cat(
                (ori_imgs_no11[j]["support"], ori_imgs_no11[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no11_indices_3:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no11[j]["support"] = torch.cat((ori_labels_no11[j]["support"], torch.tensor([3]).cuda()))
            ori_imgs_no11[j]["support"] = torch.cat(
                (ori_imgs_no11[j]["support"], ori_imgs_no11[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no11_indices_4:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no11[j]["support"] = torch.cat((ori_labels_no11[j]["support"], torch.tensor([4]).cuda()))
            ori_imgs_no11[j]["support"] = torch.cat(
                (ori_imgs_no11[j]["support"], ori_imgs_no11[j]["query"][k].unsqueeze(0)), dim=0)

        all_pre_no12 = torch.stack(all_pre_no12)
        no12_counts_0 = torch.sum(torch.eq(all_pre_no12, 0), dim=0)
        no12_counts_1 = torch.sum(torch.eq(all_pre_no12, 1), dim=0)
        no12_counts_2 = torch.sum(torch.eq(all_pre_no12, 2), dim=0)
        no12_counts_3 = torch.sum(torch.eq(all_pre_no12, 3), dim=0)
        no12_counts_4 = torch.sum(torch.eq(all_pre_no12, 4), dim=0)

        no12_indices_0 = torch.where(no12_counts_0 >= 10)[0]
        no12_indices_1 = torch.where(no12_counts_1 >= 10)[0]
        no12_indices_2 = torch.where(no12_counts_2 >= 10)[0]
        no12_indices_3 = torch.where(no12_counts_3 >= 10)[0]
        no12_indices_4 = torch.where(no12_counts_4 >= 10)[0]

        for i in no12_indices_0:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no12[j]["support"] = torch.cat((ori_labels_no12[j]["support"], torch.tensor([0]).cuda()))
            ori_imgs_no12[j]["support"] = torch.cat(
                (ori_imgs_no12[j]["support"], ori_imgs_no12[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no12_indices_1:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no12[j]["support"] = torch.cat((ori_labels_no12[j]["support"], torch.tensor([1]).cuda()))
            ori_imgs_no12[j]["support"] = torch.cat(
                (ori_imgs_no12[j]["support"], ori_imgs_no12[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no12_indices_2:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no12[j]["support"] = torch.cat((ori_labels_no12[j]["support"], torch.tensor([2]).cuda()))
            ori_imgs_no12[j]["support"] = torch.cat(
                (ori_imgs_no12[j]["support"], ori_imgs_no12[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no12_indices_3:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no12[j]["support"] = torch.cat((ori_labels_no12[j]["support"], torch.tensor([3]).cuda()))
            ori_imgs_no12[j]["support"] = torch.cat(
                (ori_imgs_no12[j]["support"], ori_imgs_no12[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no12_indices_4:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no12[j]["support"] = torch.cat((ori_labels_no12[j]["support"], torch.tensor([4]).cuda()))
            ori_imgs_no12[j]["support"] = torch.cat(
                (ori_imgs_no12[j]["support"], ori_imgs_no12[j]["query"][k].unsqueeze(0)), dim=0)

        all_pre_no13 = torch.stack(all_pre_no13)
        no13_counts_0 = torch.sum(torch.eq(all_pre_no13, 0), dim=0)
        no13_counts_1 = torch.sum(torch.eq(all_pre_no13, 1), dim=0)
        no13_counts_2 = torch.sum(torch.eq(all_pre_no13, 2), dim=0)
        no13_counts_3 = torch.sum(torch.eq(all_pre_no13, 3), dim=0)
        no13_counts_4 = torch.sum(torch.eq(all_pre_no13, 4), dim=0)

        no13_indices_0 = torch.where(no13_counts_0 >= 10)[0]
        no13_indices_1 = torch.where(no13_counts_1 >= 10)[0]
        no13_indices_2 = torch.where(no13_counts_2 >= 10)[0]
        no13_indices_3 = torch.where(no13_counts_3 >= 10)[0]
        no13_indices_4 = torch.where(no13_counts_4 >= 10)[0]

        for i in no13_indices_0:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no13[j]["support"] = torch.cat((ori_labels_no13[j]["support"], torch.tensor([0]).cuda()))
            ori_imgs_no13[j]["support"] = torch.cat(
                (ori_imgs_no13[j]["support"], ori_imgs_no13[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no13_indices_1:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no13[j]["support"] = torch.cat((ori_labels_no13[j]["support"], torch.tensor([1]).cuda()))
            ori_imgs_no13[j]["support"] = torch.cat(
                (ori_imgs_no13[j]["support"], ori_imgs_no13[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no13_indices_2:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no13[j]["support"] = torch.cat((ori_labels_no13[j]["support"], torch.tensor([2]).cuda()))
            ori_imgs_no13[j]["support"] = torch.cat(
                (ori_imgs_no13[j]["support"], ori_imgs_no13[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no13_indices_3:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no13[j]["support"] = torch.cat((ori_labels_no13[j]["support"], torch.tensor([3]).cuda()))
            ori_imgs_no13[j]["support"] = torch.cat(
                (ori_imgs_no13[j]["support"], ori_imgs_no13[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no13_indices_4:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no13[j]["support"] = torch.cat((ori_labels_no13[j]["support"], torch.tensor([4]).cuda()))
            ori_imgs_no13[j]["support"] = torch.cat(
                (ori_imgs_no13[j]["support"], ori_imgs_no13[j]["query"][k].unsqueeze(0)), dim=0)

        all_pre_no14 = torch.stack(all_pre_no14)
        no14_counts_0 = torch.sum(torch.eq(all_pre_no14, 0), dim=0)
        no14_counts_1 = torch.sum(torch.eq(all_pre_no14, 1), dim=0)
        no14_counts_2 = torch.sum(torch.eq(all_pre_no14, 2), dim=0)
        no14_counts_3 = torch.sum(torch.eq(all_pre_no14, 3), dim=0)
        no14_counts_4 = torch.sum(torch.eq(all_pre_no14, 4), dim=0)

        no14_indices_0 = torch.where(no14_counts_0 >= 10)[0]
        no14_indices_1 = torch.where(no14_counts_1 >= 10)[0]
        no14_indices_2 = torch.where(no14_counts_2 >= 10)[0]
        no14_indices_3 = torch.where(no14_counts_3 >= 10)[0]
        no14_indices_4 = torch.where(no14_counts_4 >= 10)[0]

        for i in no14_indices_0:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no14[j]["support"] = torch.cat((ori_labels_no14[j]["support"], torch.tensor([0]).cuda()))
            ori_imgs_no14[j]["support"] = torch.cat(
                (ori_imgs_no14[j]["support"], ori_imgs_no14[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no14_indices_1:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no14[j]["support"] = torch.cat((ori_labels_no14[j]["support"], torch.tensor([1]).cuda()))
            ori_imgs_no14[j]["support"] = torch.cat(
                (ori_imgs_no14[j]["support"], ori_imgs_no14[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no14_indices_2:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no14[j]["support"] = torch.cat((ori_labels_no14[j]["support"], torch.tensor([2]).cuda()))
            ori_imgs_no14[j]["support"] = torch.cat(
                (ori_imgs_no14[j]["support"], ori_imgs_no14[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no14_indices_3:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no14[j]["support"] = torch.cat((ori_labels_no14[j]["support"], torch.tensor([3]).cuda()))
            ori_imgs_no14[j]["support"] = torch.cat(
                (ori_imgs_no14[j]["support"], ori_imgs_no14[j]["query"][k].unsqueeze(0)), dim=0)
        for i in no14_indices_4:
            j = i / 75
            k = i % 75
            j = j.cpu()
            j = j.item()
            j = int(j)
            k = k.cpu()
            k = k.item()
            k = int(k)
            ori_labels_no14[j]["support"] = torch.cat((ori_labels_no14[j]["support"], torch.tensor([4]).cuda()))
            ori_imgs_no14[j]["support"] = torch.cat(
                (ori_imgs_no14[j]["support"], ori_imgs_no14[j]["query"][k].unsqueeze(0)), dim=0)

        q_counts0 = []
        for i, pre in enumerate(
                [all_pre_no0, all_pre_no1, all_pre_no2, all_pre_no3, all_pre_no4, all_pre_no5, all_pre_no6, all_pre_no7,
                 all_pre_no8, all_pre_no9,
                 all_pre_no10, all_pre_no11, all_pre_no12, all_pre_no13, all_pre_no14]):
            q_count = torch.tensor(0)
            q_count.cuda()
            s_0 = torch.sum(torch.eq(pre, 0), dim=0)
            s_1 = torch.sum(torch.eq(pre, 1), dim=0)
            s_2 = torch.sum(torch.eq(pre, 2), dim=0)
            s_3 = torch.sum(torch.eq(pre, 3), dim=0)
            s_4 = torch.sum(torch.eq(pre, 4), dim=0)
            for j in range(s_0.size(0)):
                if s_0[j] >= 10 and all_pre[i, j] != 0:
                    q_count += 1
            for j in range(s_1.size(0)):
                if s_1[j] >= 10 and all_pre[i, j] != 1:
                    q_count += 1
            for j in range(s_2.size(0)):
                if s_2[j] >= 10 and all_pre[i, j] != 2:
                    q_count += 1
            for j in range(s_3.size(0)):
                if s_3[j] >= 10 and all_pre[i, j] != 3:
                    q_count += 1
            for j in range(s_4.size(0)):
                if s_4[j] >= 10 and all_pre[i, j] != 4:
                    q_count += 1
            q_counts0.append(q_count.item())

        score = model.test_forward(ori_imgs_no0, ori_labels_no0, dataset_index)
        score = torch.stack(score)
        score = torch.softmax(score, dim=2)

        score1 = model1.test_forward(ori_imgs_no1, ori_labels_no1, dataset_index)
        score1 = torch.stack(score1)
        score1 = torch.softmax(score1, dim=2)

        score2 = model2.test_forward(ori_imgs_no2, ori_labels_no2, dataset_index)
        score2 = torch.stack(score2)
        score2 = torch.softmax(score2, dim=2)

        score3 = model3.test_forward(ori_imgs_no3, ori_labels_no3, dataset_index)
        score3 = torch.stack(score3)
        score3 = torch.softmax(score3, dim=2)

        score4 = model4.test_forward(ori_imgs_no4, ori_labels_no4, dataset_index)
        score4 = torch.stack(score4)
        score4 = torch.softmax(score4, dim=2)

        score5 = model5.test_forward(ori_imgs_no5, ori_labels_no5, dataset_index)
        score5 = torch.stack(score5)
        score5 = torch.softmax(score5, dim=2)

        score6 = model6.test_forward(ori_imgs_no6, ori_labels_no6, dataset_index)
        score6 = torch.stack(score6)
        score6 = torch.softmax(score6, dim=2)

        score7 = model7.test_forward(ori_imgs_no7, ori_labels_no7, dataset_index)
        score7 = torch.stack(score7)
        score7 = torch.softmax(score7, dim=2)

        score8 = model8.test_forward(ori_imgs_no8, ori_labels_no8, dataset_index)
        score8 = torch.stack(score8)
        score8 = torch.softmax(score8, dim=2)

        score9 = model9.test_forward(ori_imgs_no9, ori_labels_no9, dataset_index)
        score9 = torch.stack(score9)
        score9 = torch.softmax(score9, dim=2)

        score10 = model10.test_forward(ori_imgs_no10, ori_labels_no10, dataset_index)
        score10 = torch.stack(score10)
        score10 = torch.softmax(score10, dim=2)

        score11 = model11.test_forward(ori_imgs_no11, ori_labels_no11, dataset_index)
        score11 = torch.stack(score11)
        score11 = torch.softmax(score11, dim=2)

        score12 = model12.test_forward(ori_imgs_no12, ori_labels_no12, dataset_index)
        score12 = torch.stack(score12)
        score12 = torch.softmax(score12, dim=2)

        score13 = model13.test_forward(ori_imgs_no13, ori_labels_no13, dataset_index)
        score13 = torch.stack(score13)
        score13 = torch.softmax(score13, dim=2)

        score14 = model14.test_forward(ori_imgs_no14, ori_labels_no14, dataset_index)
        score14 = torch.stack(score14)
        score14 = torch.softmax(score14, dim=2)

        score_1 = []
        all_pre_1 = []
        for i, s in enumerate(
                [score, score1, score2, score3, score4, score5, score6, score7, score8, score9, score10, score11,
                 score12, score13, score14]):
            score_1.append(s)
            max_indices_1 = torch.argmax(s, dim=2)
            max_indices_1 = max_indices_1.view(-1)
            all_pre_1.append(max_indices_1)
        all_pre_1 = torch.stack(all_pre_1)
        pre_no10 = all_pre_1[1:, :]
        pre_no11 = torch.cat((all_pre_1[:1, :], all_pre_1[2:, :]), dim=0)
        pre_no12 = torch.cat((all_pre_1[:2, :], all_pre_1[3:, :]), dim=0)
        pre_no13 = torch.cat((all_pre_1[:3, :], all_pre_1[4:, :]), dim=0)
        pre_no14 = torch.cat((all_pre_1[:4, :], all_pre_1[5:, :]), dim=0)
        pre_no15 = torch.cat((all_pre_1[:5, :], all_pre_1[6:, :]), dim=0)
        pre_no16 = torch.cat((all_pre_1[:6, :], all_pre_1[7:, :]), dim=0)
        pre_no17 = torch.cat((all_pre_1[:7, :], all_pre_1[8:, :]), dim=0)
        pre_no18 = torch.cat((all_pre_1[:8, :], all_pre_1[9:, :]), dim=0)
        pre_no19 = torch.cat((all_pre_1[:9, :], all_pre_1[10:, :]), dim=0)
        pre_no110 = torch.cat((all_pre_1[:10, :], all_pre_1[11:, :]), dim=0)
        pre_no111 = torch.cat((all_pre_1[:11, :], all_pre_1[12:, :]), dim=0)
        pre_no112 = torch.cat((all_pre_1[:12, :], all_pre_1[13:, :]), dim=0)
        pre_no113 = torch.cat((all_pre_1[:13, :], all_pre_1[14:, :]), dim=0)
        pre_no114 = all_pre_1[:14, :]

        q_counts1 = []
        for i, pre in enumerate([pre_no10, pre_no11, pre_no12, pre_no13, pre_no14, pre_no15, pre_no16, pre_no17,
                                 pre_no18, pre_no19, pre_no110, pre_no111, pre_no112, pre_no113, pre_no114]):
            q_count = torch.tensor(0)
            q_count.cuda()
            s_0 = torch.sum(torch.eq(pre, 0), dim=0)
            s_1 = torch.sum(torch.eq(pre, 1), dim=0)
            s_2 = torch.sum(torch.eq(pre, 2), dim=0)
            s_3 = torch.sum(torch.eq(pre, 3), dim=0)
            s_4 = torch.sum(torch.eq(pre, 4), dim=0)
            for j in range(s_0.size(0)):
                if s_0[j] >= 10 and all_pre_1[i, j] != 0:
                    q_count += 1
            for j in range(s_1.size(0)):
                if s_1[j] >= 10 and all_pre_1[i, j] != 1:
                    q_count += 1
            for j in range(s_2.size(0)):
                if s_2[j] >= 10 and all_pre_1[i, j] != 2:
                    q_count += 1
            for j in range(s_3.size(0)):
                if s_3[j] >= 10 and all_pre_1[i, j] != 3:
                    q_count += 1
            for j in range(s_4.size(0)):
                if s_4[j] >= 10 and all_pre_1[i, j] != 4:
                    q_count += 1
            q_counts1.append(q_count.item())

        for i in range(15):
            if q_counts1[i] > q_counts0[i]:
                score_1[i] = score_0[i]
                q_counts1[i] = q_counts0[i]

        combined_score = torch.zeros(8, 75, 5).cuda()
        for i in range(15):
            combined_score += score_1[i]

        accs = []
        for i, row in enumerate(combined_score):
            # print(f"Combined Score {i}: {row}")
            accs.append(accuracy(row, labels[i]["query"].cuda())[0])
        # print(accs)
        acc_ci.extend(accs)
        acc = sum(accs) / len(accs)
        print(acc)

        acc_meter.update(acc.item())


        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            logger.info(
                f'Test: [{idx + 1}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                # f'Loss {loss_meter.val:.2f} ({loss_meter.avg:.2f})\t'
                f'Acc@1 {acc_meter.val:.2f} ({acc_meter.avg:.2f})\t'
            )
    acc_ci = torch.stack(acc_ci)

    ci = (1.96 * torch.std(acc_ci) / math.sqrt(acc_ci.shape[0])).item()

    return acc_meter.avg, ci


if __name__ == '__main__':
    args, config, config1, config2, config3, config4, config5, config6, config7, config8, config9, config10, config11, config12, config13, config14 = parse_option()

    torch.cuda.set_device(config.GPU_ID)
    torch.cuda.set_device(config1.GPU_ID)
    torch.cuda.set_device(config2.GPU_ID)
    torch.cuda.set_device(config3.GPU_ID)
    torch.cuda.set_device(config4.GPU_ID)
    torch.cuda.set_device(config5.GPU_ID)
    torch.cuda.set_device(config6.GPU_ID)
    torch.cuda.set_device(config7.GPU_ID)
    torch.cuda.set_device(config8.GPU_ID)
    torch.cuda.set_device(config9.GPU_ID)
    torch.cuda.set_device(config10.GPU_ID)
    torch.cuda.set_device(config11.GPU_ID)
    torch.cuda.set_device(config12.GPU_ID)
    torch.cuda.set_device(config13.GPU_ID)
    torch.cuda.set_device(config14.GPU_ID)

    config.defrost()
    config1.defrost()
    config2.defrost()
    config3.defrost()
    config4.defrost()
    config5.defrost()
    config6.defrost()
    config7.defrost()
    config8.defrost()
    config9.defrost()
    config10.defrost()
    config11.defrost()
    config12.defrost()
    config13.defrost()
    config14.defrost()

    config.freeze()
    config1.freeze()
    config2.freeze()
    config3.freeze()
    config4.freeze()
    config5.freeze()
    config6.freeze()
    config7.freeze()
    config8.freeze()
    config9.freeze()
    config10.freeze()
    config11.freeze()
    config12.freeze()
    config13.freeze()
    config14.freeze()

    setup_seed(config.SEED)
    setup_seed1(config1.SEED)
    setup_seed2(config2.SEED)
    setup_seed3(config3.SEED)
    setup_seed4(config4.SEED)
    setup_seed5(config5.SEED)
    setup_seed6(config6.SEED)
    setup_seed7(config7.SEED)
    setup_seed8(config8.SEED)
    setup_seed9(config9.SEED)
    setup_seed10(config10.SEED)
    setup_seed11(config11.SEED)
    setup_seed12(config12.SEED)
    setup_seed13(config13.SEED)
    setup_seed14(config14.SEED)

    os.makedirs(config.OUTPUT, exist_ok=True)
    os.makedirs(config1.OUTPUT, exist_ok=True)
    os.makedirs(config2.OUTPUT, exist_ok=True)
    os.makedirs(config3.OUTPUT, exist_ok=True)
    os.makedirs(config4.OUTPUT, exist_ok=True)
    os.makedirs(config5.OUTPUT, exist_ok=True)
    os.makedirs(config6.OUTPUT, exist_ok=True)
    os.makedirs(config7.OUTPUT, exist_ok=True)
    os.makedirs(config8.OUTPUT, exist_ok=True)
    os.makedirs(config9.OUTPUT, exist_ok=True)
    os.makedirs(config10.OUTPUT, exist_ok=True)
    os.makedirs(config11.OUTPUT, exist_ok=True)
    os.makedirs(config12.OUTPUT, exist_ok=True)
    os.makedirs(config13.OUTPUT, exist_ok=True)
    os.makedirs(config14.OUTPUT, exist_ok=True)

    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")
    logger1 = create_logger(output_dir=config1.OUTPUT, name=f"{config1.MODEL.NAME}")
    logger2 = create_logger(output_dir=config2.OUTPUT, name=f"{config2.MODEL.NAME}")
    logger3 = create_logger(output_dir=config3.OUTPUT, name=f"{config3.MODEL.NAME}")
    logger4 = create_logger(output_dir=config4.OUTPUT, name=f"{config4.MODEL.NAME}")
    logger5 = create_logger(output_dir=config5.OUTPUT, name=f"{config5.MODEL.NAME}")
    logger6 = create_logger(output_dir=config6.OUTPUT, name=f"{config6.MODEL.NAME}")
    logger7 = create_logger(output_dir=config7.OUTPUT, name=f"{config7.MODEL.NAME}")
    logger8 = create_logger(output_dir=config8.OUTPUT, name=f"{config8.MODEL.NAME}")
    logger9 = create_logger(output_dir=config9.OUTPUT, name=f"{config9.MODEL.NAME}")
    logger10 = create_logger(output_dir=config10.OUTPUT, name=f"{config10.MODEL.NAME}")
    logger11 = create_logger(output_dir=config11.OUTPUT, name=f"{config11.MODEL.NAME}")
    logger12 = create_logger(output_dir=config12.OUTPUT, name=f"{config12.MODEL.NAME}")
    logger13 = create_logger(output_dir=config13.OUTPUT, name=f"{config13.MODEL.NAME}")
    logger14 = create_logger(output_dir=config14.OUTPUT, name=f"{config14.MODEL.NAME}")

    path = os.path.join(config.OUTPUT, "config.json")

    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")



    test(config, config1, config2, config3, config4, config5, config6, config7, config8, config9, config10,config11, config12, config13, config14)


