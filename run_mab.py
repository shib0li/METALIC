import numpy as np
import pickle
import fire
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm.auto import tqdm, trange

from infras.randutils import *
from infras.misc import *

from infras.configs import ExpConfig


def evaluation(**kwargs):

    config = ExpConfig()
    config.parse(kwargs)

    if config.domain == 'Poisson':
        if config.trunk == 'default':
            from data.xpinn_poisson import XPINN_Client_Poisson as XPINN_Client
        elif config.trunk == 'stage':
            from data.xpinn_poisson_staged import XPINN_Client_Poisson as XPINN_Client
        else:
            raise Exception('error in trunk mode')
        #
        from data.games import GamePoisson as PDEGame
    elif config.domain == 'Heat':
        if config.trunk == 'default':
            from data.xpinn_heat import XPINN_Client_Heat as XPINN_Client
        elif config.trunk == 'stage':
            from data.xpinn_heat_staged import XPINN_Client_Heat as XPINN_Client
        else:
            raise Exception('error in trunk mode')
        #
        from data.games import GameHeat as PDEGame
    elif config.domain == 'Burgers':
        if config.trunk == 'default':
            from data.xpinn_burgers import XPINN_Client_Burgers as XPINN_Client
        elif config.trunk == 'stage':
            from data.xpinn_burgers_staged import XPINN_Client_Burgers as XPINN_Client
        else:
            raise Exception('error in trunk mode')
        #
        from data.games import GameBurgers as PDEGame
    elif config.domain == 'KdV':
        if config.trunk == 'default':
            from data.xpinn_kdv import XPINN_Client_KdV as XPINN_Client
        elif config.trunk == 'stage':
            from data.xpinn_kdv_staged import XPINN_Client_KdV as XPINN_Client
        else:
            raise Exception('error in trunk mode')
        #
        from data.games import GameKdV as PDEGame
    elif config.domain == 'Advec':
        if config.trunk == 'default':
            from data.xpinn_advec import XPINN_Client_Advec as XPINN_Client
        elif config.trunk == 'stage':
            from data.xpinn_advec_staged import XPINN_Client_Advec as XPINN_Client
        else:
            raise Exception('error in trunk mode')
        #
        from data.games import GameAdvec as PDEGame
    elif config.domain == 'Reaction':
        if config.trunk == 'default':
            from data.xpinn_reaction import XPINN_Client_Reaction as XPINN_Client
        elif config.trunk == 'stage':
            from data.xpinn_reaction_staged import XPINN_Client_Reaction as XPINN_Client
        else:
            raise Exception('error in trunk mode')
        #
        from data.games import GameReaction as PDEGame
    else:
        raise Exception('Error in domain name')
    
    if config.mode == 'query':
        from data.bandits import QueryBandit as Bandit
        from core.query_agent import QueryBanditAgent as BanditAgent
        from infras.meters import QueryBanditMeters as BanditMeters
    elif config.mode == 'batch':
        from data.bandits import BatchBandit as Bandit
        from core.batch_agent import BatchBanditAgent as BanditAgent
        from infras.meters import BatchBanditMeters as BanditMeters
    elif config.mode == 'step':
        from data.bandits import StepBandit as Bandit
        from core.step_agent import StepBanditAgent as BanditAgent
        from infras.meters import StepBanditMeters as BanditMeters
    elif config.mode == 'step2':
        from data.bandits import StepBandit as Bandit
        from core.step_agent_v2 import StepBanditAgent_v2 as BanditAgent
        from infras.meters import StepBanditMeters as BanditMeters
    elif config.mode == 'step3':
        from data.bandits import StepBandit as Bandit
        from core.step_agent_v3 import StepBanditAgent_v3 as BanditAgent
        from infras.meters import StepBanditMeters as BanditMeters
    else:
        raise Exception('error mode')
        
    #
    
    
    device = torch.device(config.device)
    domain = config.domain
    trunk = config.trunk
    heuristic = config.heuristic
    
    mode = config.mode
    
    nfree = config.nfree
    nplay = config.nplay
    nroll = config.nroll
    
    log_path = os.path.join(
        '__log__',
        domain,
        trunk,
        mode,
        heuristic
    )
    create_path(log_path)
    
    res_path = os.path.join(
        '__res__',
        domain,
        trunk,
        mode,
        heuristic
    )
    
    create_path(res_path)
    
    logger = get_logger(logpath=os.path.join(log_path, 'log.txt'), displaying=config.verbose)
    
    xpinn_client = XPINN_Client(
        num_adam=config.num_adam,
        int_adam=config.int_adam,
        lr_adam=config.lr_adam,
        num_lbfgs=config.num_lbfgs,
        int_lbfgs=config.int_lbfgs,
        lr_lbfgs=config.lr_lbfgs,
        layers=[2,20,20,1],
        device=device,
        err_cap=100.0,
    )
    
    env = PDEGame(xpinn_client)
    
    bandit = Bandit(env)
    
    
    play_meters = BanditMeters(
        bandit=bandit,
        save_path=os.path.join(res_path, 'play'),
        logger=logger
    )

    roll_meters = BanditMeters(
        bandit=bandit,
        save_path=os.path.join(res_path, 'roll'),
        logger=logger
    )
    
    agent = BanditAgent(
        bandit=bandit, 
        play_meters=play_meters, 
        roll_meters=roll_meters,
        state_dict=None,
        device=device
    )
        
    logger.info('####################')
    logger.info('#       PLAY       #')
    logger.info('####################')
    agent.play(nfree, nplay, heuristic)
    
    
    xpinn_client.int_adam = config.int_adam_test
    xpinn_client.int_lbfgs = config.int_lbfgs_test
    
    logger.info('####################')
    logger.info('#       ROLL       #')
    logger.info('####################')
    agent.roll(nroll, heuristic)


if __name__=='__main__':
    fire.Fire(evaluation)
    
    
    
