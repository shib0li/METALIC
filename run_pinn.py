import numpy as np
import pickle
import fire
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm.auto import tqdm, trange

from infras.randutils import *
from infras.misc import *


from infras.configs import PINNConfig


def evaluation(**kwargs):

    config = PINNConfig()
    config.parse(kwargs)
    
    if config.domain == 'Poisson':
        from data.pinn_poisson import PINN_Client_Poisson as PINN_Client
        from data.games import PINNPoisson as PDEGame
    elif config.domain == 'Heat':
        from data.pinn_heat import PINN_Client_Heat as PINN_Client
        from data.games import PINNHeat as PDEGame
    elif config.domain == 'Burgers':
        from data.pinn_burgers import PINN_Client_Burgers as PINN_Client
        from data.games import PINNBurgers as PDEGame
    elif config.domain == 'KdV':
        from data.pinn_kdv import PINN_Client_KdV as PINN_Client
        from data.games import PINNKdV as PDEGame
    elif config.domain == 'Advec':
        from data.pinn_advec import PINN_Client_Advec as PINN_Client
        from data.games import PINNAdvec as PDEGame
    elif config.domain == 'Reaction':
        from data.pinn_reaction import PINN_Client_Reaction as PINN_Client
        from data.games import PINNReaction as PDEGame
    else:
        raise Exception('Error in domain name')
    #
    
    from data.bandits import PINNBandit as Bandit
    from core.pinn_agent import PINNBanditAgent as BanditAgent
    from infras.meters import PINNBanditMeters as BanditMeters
    
    
    device = torch.device(config.device)
    domain = config.domain
    
    nfree = config.nfree
    nplay = config.nplay
    nroll = config.nroll
    
    log_path = os.path.join(
        '__log__',
        domain,
        'pinn'
    )
    create_path(log_path)
    
    res_path = os.path.join(
        '__res__',
        domain,
        'pinn'
    )
    
    create_path(res_path)
    
    logger = get_logger(logpath=os.path.join(log_path, 'log.txt'), displaying=config.verbose)
    
    pinn_client = PINN_Client(
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
    
    env = PDEGame(pinn_client)
    
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
        
#     logger.info('####################')
#     logger.info('#       PLAY       #')
#     logger.info('####################')
#     agent.play(nfree, nplay)
    
    logger.info('####################')
    logger.info('#       ROLL       #')
    logger.info('####################')
    agent.roll(nroll)


if __name__=='__main__':
    fire.Fire(evaluation)
    
    
    