import time
import tqdm
import numpy as np
from collections import defaultdict
from typing import Dict, Union, Callable, Optional
import pandas as pd
from esrl.util import *
import torch

from tianshou.data import Collector
from tianshou.policy import BasePolicy
from tianshou.trainer import test_episode, gather_info
from tianshou.utils import tqdm_config, MovAvg, BaseLogger, LazyLogger


def esrl_trainer_v3(
    policy: BasePolicy,
    train_collector: Collector,
    test_collector: Collector,
    max_epoch: int,
    step_per_epoch: int,
    step_per_collect: int,
    episode_per_test: int,
    batch_size: int,
    update_per_step: Union[int, float] = 1,
    train_fn: Optional[Callable[[int, int], None]] = None,
    test_fn: Optional[Callable[[int, Optional[int]], None]] = None,
    stop_fn: Optional[Callable[[float], bool]] = None,
    save_fn: Optional[Callable[[BasePolicy], None]] = None,
    save_checkpoint_fn: Optional[Callable[[int, int, int], None]] = None,
    resume_from_log: bool = False,
    reward_metric: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    logger: BaseLogger = LazyLogger(),
    verbose: bool = True,
    test_in_train: bool = True,
    **kwargs
) -> Dict[str, Union[float, str]]:
    """A wrapper for off-policy trainer procedure.

    The "step" in trainer means an environment step (a.k.a. transition).

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy` class.
    :param Collector train_collector: the collector used for training.
    :param Collector test_collector: the collector used for testing.
    :param int max_epoch: the maximum number of epochs for training. The training
        process might be finished before reaching ``max_epoch`` if ``stop_fn`` is set.
    :param int step_per_epoch: the number of transitions collected per epoch.
    :param int step_per_collect: the number of transitions the collector would collect
        before the network update, i.e., trainer will collect "step_per_collect"
        transitions and do some policy network update repeatly in each epoch.
    :param episode_per_test: the number of episodes for one policy evaluation.
    :param int batch_size: the batch size of sample data, which is going to feed in the
        policy network.
    :param int/float update_per_step: the number of times the policy network would be
        updated per transition after (step_per_collect) transitions are collected,
        e.g., if update_per_step set to 0.3, and step_per_collect is 256, policy will
        be updated round(256 * 0.3 = 76.8) = 77 times after 256 transitions are
        collected by the collector. Default to 1.
    :param function train_fn: a hook called at the beginning of training in each epoch.
        It can be used to perform custom additional operations, with the signature ``f(
        num_epoch: int, step_idx: int) -> None``.
    :param function test_fn: a hook called at the beginning of testing in each epoch.
        It can be used to perform custom additional operations, with the signature ``f(
        num_epoch: int, step_idx: int) -> None``.
    :param function save_fn: a hook called when the undiscounted average mean reward in
        evaluation phase gets better, with the signature ``f(policy: BasePolicy) ->
        None``.
    :param function save_checkpoint_fn: a function to save training process, with the
        signature ``f(epoch: int, env_step: int, gradient_step: int) -> None``; you can
        save whatever you want.
    :param bool resume_from_log: resume env_step/gradient_step and other metadata from
        existing tensorboard log. Default to False.
    :param function stop_fn: a function with signature ``f(mean_rewards: float) ->
        bool``, receives the average undiscounted returns of the testing result,
        returns a boolean which indicates whether reaching the goal.
    :param function reward_metric: a function with signature ``f(rewards: np.ndarray
        with shape (num_episode, agent_num)) -> np.ndarray with shape (num_episode,)``,
        used in multi-agent RL. We need to return a single scalar for each episode's
        result to monitor training in the multi-agent RL setting. This function
        specifies what is the desired metric, e.g., the reward of agent 1 or the
        average reward over all agents.
    :param BaseLogger logger: A logger that logs statistics during
        training/testing/updating. Default to a logger that doesn't log anything.
    :param bool verbose: whether to print the information. Default to True.
    :param bool test_in_train: whether to test in the training phase. Default to True.

    :return: See :func:`~tianshou.trainer.gather_info`.
    """
    start_epoch, env_step, gradient_step = 0, 0, 0
    if resume_from_log:
        start_epoch, env_step, gradient_step = logger.restore_data()
    last_rew, last_len = 0.0, 0
    stat: Dict[str, MovAvg] = defaultdict(MovAvg)
    start_time = time.time()
    train_collector.reset_stat()
    test_collector.reset_stat()
    test_in_train = test_in_train and train_collector.policy == policy
    test_result = test_episode(policy, test_collector, test_fn, start_epoch,
                               episode_per_test, logger, env_step, reward_metric)
    best_epoch = start_epoch
    best_reward, best_reward_std = test_result["rew"], test_result["rew_std"]

    es = kwargs['es']
    pop_size = kwargs['pop_size']
    max_step = kwargs['max_step']
    log_path = kwargs['log_path']
    actor_lr = kwargs['actor_lr']
    episode_per_epoch = kwargs['episode_per_epoch']
    df = pd.DataFrame(columns=["total_steps",
                               "mu_score", "mu_score_std",])
    mean_fitness = -9999

    for epoch in range(1 + start_epoch, 1 + max_epoch):
        if env_step >= max_step:
            break


        params = es.ask(pop_size//2)
        es_fitness = [0] * (pop_size//2)
        rl_fitness = [0] * (pop_size//2)
        for pop_ind in range(pop_size//2):
            set_params(policy.actor, params[pop_ind])
            set_params(policy.actor_old, params[pop_ind])
            policy.train()
            result = train_collector.collect(n_episode=1)
            es_fitness[pop_ind] = int(result['rews'][0])
            env_step += int(result['n/st'])
        prYellow(f'\nEnv Step: {env_step}')
        prGreen(f'ES fitness: {es_fitness}')

        rl_params = np.zeros_like(params)
        for pop_ind in range(pop_size//2):
            set_params(policy.actor, params[pop_ind])
            set_params(policy.actor_old, params[pop_ind])
            policy.actor_optim = torch.optim.Adam(policy.actor.parameters(), actor_lr)
            policy.train()
            actor_step = 0
            actor_score = -9999
            while actor_step < step_per_epoch and actor_score < 1.3 * es_fitness[pop_ind]:
                with tqdm.tqdm(
                    total=episode_per_epoch, desc=f"Actor #{pop_ind}", **tqdm_config
                ) as t:
                    while t.n < t.total:
                        result = {}

                        while 'rew' not in result:
                            result = train_collector.collect(n_step=step_per_collect)
                            env_step += int(result["n/st"])
                            actor_step += int(result["n/st"])
                            logger.log_train_data(result, env_step)
                            data = {
                                "env_step": str(env_step),
                                "n/ep": str(int(t.n)),
                                "n/st": str(actor_step),
                            }
                            for i in range(update_per_step):
                                gradient_step += 1
                                losses = policy.update(batch_size, train_collector.buffer)
                                for k in losses.keys():
                                    stat[k].add(losses[k])
                                    losses[k] = stat[k].get()
                                    data[k] = f"{losses[k]:.3f}"
                                logger.log_update_data(losses, gradient_step)
                                t.set_postfix(**data)
                        t.update(1)

                    if t.n <= t.total:
                        t.update()

                actor_test_result = train_collector.collect(n_episode=1)
                env_step += actor_test_result['n/st']
                actor_score = int(actor_test_result['rews'][0])
                prLightPurple(f'\tactor_test_result: {actor_test_result}')

            rl_params[pop_ind] = get_params(policy.actor)
            rl_fitness[pop_ind] = actor_score
        
        prRed(f'RL fitness: {rl_fitness}')
        es.tell(np.concatenate((rl_params,params)), rl_fitness+es_fitness)

        
        set_params(policy.actor, es.mu)
        set_params(policy.actor_old, es.mu)
        test_result = test_episode(policy, test_collector, test_fn, epoch,
                                   episode_per_test, logger, env_step, reward_metric)
        rew, rew_std = test_result["rew"], test_result["rew_std"]
        df.to_pickle(os.path.join(log_path, 'log.pkl'))
        res = {"total_steps": env_step,
               "mu_score": rew,
               "mu_score_std": rew_std,
               }
        df = df.append(res, ignore_index=True)

        if best_epoch < 0 or best_reward < rew:
            best_epoch, best_reward, best_reward_std = epoch, rew, rew_std
            if save_fn:
                save_fn(policy)
        logger.save_data(epoch, env_step, gradient_step, save_checkpoint_fn)
        if verbose:
            print(f"Epoch #{epoch}: test_reward: {rew:.6f} ± {rew_std:.6f}, best_rew"
                  f"ard: {best_reward:.6f} ± {best_reward_std:.6f} in #{best_epoch}")
        if stop_fn and stop_fn(best_reward):
            break
    return gather_info(start_time, train_collector, test_collector,
                       best_reward, best_reward_std)
