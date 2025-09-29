from .ST.ST import *
from .RD4AD.RD4AD import *
from .PEFM.PEFM import PEFM
from .model import *

from loguru import logger

def get_model_from_args(**kwargs)->CoverModel:
    method = kwargs['method']
    if method == 'ST':
        model = ST(**kwargs)
        debug_str = f"===> method: {method}, backbone: {kwargs['backbone']}"
    elif method == 'RD4AD':
        model = RD4AD(**kwargs)
        debug_str = f"===> method: {method}, backbone: {kwargs['backbone']}"
    elif method == 'PEFM':
        model = PEFM(**kwargs)
        debug_str = f"===> method: {method}, agent_S: {kwargs['agent_S']}, agent_T: {kwargs['agent_T']}, " \
                    f"dual_type: {kwargs['dual_type']}, pe_required: {kwargs['pe_required']}"
    else:
        raise NotImplementedError

    logger.info(debug_str)

    return model