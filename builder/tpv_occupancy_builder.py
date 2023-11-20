from tpvformer04 import *
try:
    from tpvformer04_orig.tpvformer import TPVFormerOrig
    from tpvformer04_orig.tpv_head import TPVFormerHeadOrig
    from tpvformer04_orig.tpv_aggregator import TPVAggregatorOrig
except:
    pass

from mmseg.models import build_segmentor

def build(model_config):
    model = build_segmentor(model_config)
    model.init_weights()
    return model

