import numpy as np
import pickle
import fire
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm.auto import tqdm, trange

from infras.randutils import *
from infras.misc import *

from models.gp import GPR

from infras.configs import ExpConfig

# from data.pde_poisson import XPINN_Client
# from data.games import GamePoisson


# from data.bandits import QueryBandit
# from core.query_agent import QueryBanditAgent
# from infras.meters import QueryBanditMeters

def evaluation(**kwargs):

    config = ExpConfig()
    config.parse(kwargs)
    
    if config.domain == 'Poisson':
        from data.xpinn_poisson import XPINN_Client_Poisson as XPINN_Client
        from data.games import GamePoisson as PDEGame
    elif config.domain == 'Heat':
        from data.xpinn_heat import XPINN_Client_Heat as XPINN_Client
        from data.games import GameHeat as PDEGame
    
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
    else:
        raise Exception('error mode')
        
    #
    
    
    device = torch.device(config.device)
    domain = config.domain
    heuristic = config.heuristic
    
    mode = config.mode
    
    nfree = config.nfree
    nplay = config.nplay
    nroll = config.nroll
    
    log_path = os.path.join(
        '__log__',
        domain,
        mode,
        heuristic
    )
    create_path(log_path)
    
    load_path = os.path.join(
        #'__res__',
        '__hist__',
        domain,
        mode,
        heuristic
    )
    
    res_path = os.path.join(
        '__res__',
        domain,
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
    
    hist_pickle = os.path.join(load_path, 'play', 'play{}.pickle'.format(nplay))
    with open(hist_pickle, 'rb') as handle:
        hist_data = pickle.load(handle)
    
    if mode != 'query':
        hist_data['hist_states'] = hist_data['hist_states']
        hist_data['hist_actions'] = hist_data['hist_actionTs']
        hist_data['hist_rewards'] = hist_data['hist_rewardTs']
    
    agent.load_bandit_hist(hist_data)
        
#     logger.info('####################')
#     logger.info('#       PLAY       #')
#     logger.info('####################')
#     agent.play(nfree, nplay, heuristic)
    
    
    xpinn_client.int_adam = config.int_adam_test
    xpinn_client.int_lbfgs = config.int_lbfgs_test
    
    logger.info('####################')
    logger.info('#       ROLL       #')
    logger.info('####################')
    agent.roll(nroll, heuristic)


if __name__=='__main__':
    fire.Fire(evaluation)
    
    
    