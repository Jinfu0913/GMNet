from .metrics import averageMeter, runningScore
from .log import get_logger

from .utils import ClassWeight, save_ckpt, load_ckpt, class_to_RGB, \
    compute_speed, setup_seed, group_weight_decay


def get_dataset(cfg):
    assert cfg['dataset'] in ['nyuv2', 'nyuv2_new', 'sunrgbd', 'cityscapes', 'camvid', 'irseg', 'pst900', 'irseg_msv']

    if cfg['dataset'] == 'irseg':
        from .datasets.irseg import IRSeg
        # return IRSeg(cfg, mode='trainval'), IRSeg(cfg, mode='test')
        return IRSeg(cfg, mode='train'), IRSeg(cfg, mode='val'), IRSeg(cfg, mode='test')

    if cfg['dataset'] == 'pst900':
        from .datasets.pst900 import PST900
        return PST900(cfg, mode='train'), PST900(cfg, mode='test')


def get_model(cfg):

    if cfg['model_name'] == 'GMNet':
        from .models.GMNet import GMNet
        return GMNet(n_classes=cfg['n_classes'])
