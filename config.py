RD4AD_config = {
    'lr':0.4
}

ST_config = {
    'lr':0.4
}

PEFM_config = {
    'agent_S': 'resnet34',
    'agent_T': 'resnet50',
    'dual_type': 'small',
    'pe_required': True
}

config={'ST':ST_config,
        'RD4AD':RD4AD_config,
        'PEFM':PEFM_config,
        }