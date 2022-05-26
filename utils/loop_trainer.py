"""
Authors: Ayan Majumdar, Miriam Rateike
Based on an older version of https://github.com/adrianjav/heterogeneous_vaes
"""

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Parameter
from ignite.engine import Engine, Events
from ignite.handlers import TerminateOnNan, ModelCheckpoint
from utils.miscelanea import print_time_epoch_value
from .metrics import NoAverage
from copy import deepcopy
from utils.constants import Cte

global epoch_every_step, time_step_loader, orig_dataset, batch_size, warmup_status, max_epochs


def create_loader():
    """
    Used once we exhaust Phase2 data to resample again.
    @return: dataloader
    """
    global orig_dataset, batch_size
    loader = iter(DataLoader(orig_dataset, batch_size=batch_size, drop_last=False, shuffle=True))
    return loader


def custom_event_filter(engine, event):
    """
    Event filter for ignite engine to trigger at the *start of every time-step*.
    @param engine: Ignite engine
    @param event: the trainer step number as event
    @return: True or False
    """
    if (event + (epoch_every_step - 1)) % epoch_every_step == 0 and not warmup_status:
        return True
    return False


def custom_event_filter_2(engine, event):
    """
    Event filter for ignite engine to trigger at the *end of every time-step*.
    @param engine: Ignite engine
    @param event: the trainer step number as event
    @return: True or False
    """
    if not warmup_status:
        if event % epoch_every_step == 0:
            return True
        return False
    else:
        if event == max_epochs:
            return True
        return False


def separate_data(trainer):
    """
    Separate data in loader into 4 parts.
    @param trainer: trainer object
    @return: split data
    """
    s11 = trainer.state.dataloader.dataset.mask[trainer.state.dataloader.dataset.data[:, -1] == 1, 0] \
        .count_nonzero().item()
    s10 = (~trainer.state.dataloader.dataset.mask[trainer.state.dataloader.dataset.data[:, -1] == 1, 0]) \
        .count_nonzero().item()
    s01 = trainer.state.dataloader.dataset.mask[trainer.state.dataloader.dataset.data[:, -1] == -1, 0] \
        .count_nonzero().item()
    s00 = (~trainer.state.dataloader.dataset.mask[trainer.state.dataloader.dataset.data[:, -1] == -1, 0]) \
        .count_nonzero().item()
    return s11, s10, s01, s00


def time_step_load_batch_data(semisup):
    """
    Function to load new data at new time step
    @param semisup: flag if semisupervised learning or not
    @return: data, mask, params, prob1
    """
    global time_step_loader
    try:
        data_sample, other = next(time_step_loader)
    except StopIteration:
        # We have exhausted the data. Create the Loader again so we can continue.
        time_step_loader = create_loader()
        data_sample, other = next(time_step_loader)

    mask_sample, param_sample = other[1], other[0]
    prob1_sample = None
    if semisup:
        prob1_sample = other[2]

    return data_sample, mask_sample, Parameter(param_sample), prob1_sample


def set_data(trainer, new_data, semisup):
    """
    Function that sets the data of the trainer Engine object to the new data.
    Can be used to update data using some policy.
    :param trainer: Ignite trainer Engine object
    :param new_data: New data that is to be loaded to the Engine
    :param semisup: Is it semi-supervised.
    :return: None
    """
    trainer.state.dataloader.dataset.data = new_data[0]
    trainer.state.dataloader.dataset.mask = new_data[1]
    trainer.state.dataloader.dataset.params = new_data[2]
    if semisup:
        trainer.state.dataloader.dataset.prob1 = new_data[3]
    trainer.state.epoch_length = len(trainer.state.dataloader)


def create_loop_trainer_2(model, dataset, cfg, policy_phase_2):
    """
    Function to create the loop trainer for phase 2 training.
    @param model: VAE model
    @param dataset: dataset object
    @param cfg: configuration dictionary
    @param policy_phase_2: policy type to use in phase 2
    @return: trainer, training evaluator, validation evaluator, test evaluator
    """
    global time_step_loader, epoch_every_step, aggr_data, aggr_mask, aggr_prob1, orig_dataset, batch_size, \
        warmup_status, max_epochs
    aggr_data = None
    aggr_mask = None
    aggr_prob1 = None

    warmup_status = cfg['trainer2']['warmup']
    epoch_every_step = cfg['trainer2']['epochs_per_ts']
    max_epochs = epoch_every_step * cfg['trainer2']['time_steps']  # Total epochs is this now.

    original_dataset = deepcopy(dataset)
    orig_dataset = original_dataset
    batch_size = cfg['trainer2']['samples_per_ts']
    time_step_loader = create_loader()

    trainer, train_evaluator, valid_evaluator, test_evaluator \
        = build_loop_trainer_2(model, dataset, cfg['optimizer']['params']['learning_rate'], cfg['save_dir'])

    metrics = ['-elbo_unsup', 'kl_unsup', '-logprob0_unsup', '-logprob1_unsup', '-recx_unsup', '-elbo_sup',
               'kl_sup', '-logprob_sup', 'clf_acc', 'clf_loss', 'total']
    if model.loss_function == Cte.LOSS_FAIRLOG:
        metrics.insert(-2, 'discr_loss')

    no_average = NoAverage(metrics)
    no_average.attach(trainer, print_time_epoch_value, trainer, vars=metrics,
                      epochs_per_timestep=cfg['trainer2']['epochs_per_ts'], max_timesteps=cfg['trainer2']['time_steps'],
                      print_every=cfg['trainer']['print_every'], evaluation="trainer")

    no_average.attach(train_evaluator, print_time_epoch_value, trainer, vars=metrics,
                      epochs_per_timestep=cfg['trainer2']['epochs_per_ts'], max_timesteps=cfg['trainer2']['time_steps'],
                      print_every=cfg['trainer']['print_every'], evaluation="train_evaluator")

    no_average.attach(valid_evaluator, print_time_epoch_value, trainer, vars=metrics,
                      epochs_per_timestep=cfg['trainer2']['epochs_per_ts'], max_timesteps=cfg['trainer2']['time_steps'],
                      print_every=cfg['trainer']['print_every'], evaluation="valid_evaluator")

    no_average.attach(test_evaluator, print_time_epoch_value, trainer, vars=metrics,
                      epochs_per_timestep=cfg['trainer2']['epochs_per_ts'], max_timesteps=cfg['trainer2']['time_steps'],
                      print_every=cfg['trainer']['print_every'], evaluation="test_evaluator")

    # Per-epoch parameters
    # 1st epoch initial policy application
    # NOTE: Ignite epoch and iterations are 1-based not 0-based!!
    @trainer.on(Events.EPOCH_STARTED(once=1))
    def update_data_by_init_policy(engine):
        if warmup_status:
            data_sample, mask_sample, param_sample, prob1_sample = time_step_load_batch_data(semisup=cfg['semisup'])
            # Initial step mask should be chosen from whatever was set in the beginning.
            if data_sample is None:
                engine.terminate()
            else:
                set_data(engine, (data_sample, mask_sample, param_sample, prob1_sample), semisup=cfg['semisup'])
            s11, s10, s01, s00 = separate_data(trainer)
            print("Policy init at start of Phase2", "... || S=1... 1:", s11, "0:", s10, "|| S=0... 1:", s01, "0:", s00)

    # Call policy training code
    @trainer.on(Events.EPOCH_COMPLETED(event_filter=custom_event_filter_2))
    def train_policy_fz(engine):
        if cfg['model']['params']['learn_policy'] == 'FZ':
            # Need to call separate training only if we are doing policy w.r.t. Z
            data_ = engine.state.dataloader.dataset.data
            mask_ = engine.state.dataloader.dataset.mask
            prob1_ = None
            if cfg['semisup']:
                prob1_ = engine.state.dataloader.dataset.prob1
            policy_phase_2.train_policy(model, data_, mask_, prob1_)
            print("Successfully trained policy model FZ!")

    # Per-K epoch Phase-2 function to update data
    @trainer.on(Events.EPOCH_STARTED(event_filter=custom_event_filter))
    def update_data_by_policy_2(engine):
        # Apply policy
        data_sample, mask_sample, param_sample, prob1_sample = time_step_load_batch_data(semisup=cfg['semisup'])
        if data_sample is None:
            engine.terminate()
        else:
            if cfg['model']['params']['learn_policy'] == 'NA':
                mask_from_policy, prob1_from_policy = policy_phase_2(mask_sample, prob1_sample)
            else:
                mask_from_policy, prob1_from_policy = policy_phase_2(model, data_sample, mask_sample)
            set_data(engine, (data_sample, mask_from_policy, param_sample, prob1_from_policy), semisup=cfg['semisup'])
            s11, s10, s01, s00 = separate_data(trainer)
            print("Applying own policy in Phase2", "... || S=1... 1:", s11, "0:", s10, "|| S=0... 1:", s01, "0:", s00)

    return trainer, train_evaluator, valid_evaluator, test_evaluator


def build_loop_trainer_2(model, dataset, learning_rate, root):
    """
    Builds the Phase 2 loop trainer. Contains the trainer step function.
    @param model: VAE model
    @param dataset: dataset object
    @param learning_rate: optim learning rate
    @param root: root directory, for checkpoint saving
    @return: trainer, train evaluator, val evaluator, test evaluator
    """
    optim = Adam([{'params': model.parameters()}, {'params': dataset.parameters()}], lr=learning_rate)
    global best_loss
    best_loss = None

    def trainer_step(engine, batch):
        model.train()
        optim.zero_grad()
        x, y = batch
        loss = model(x, engine.state, *y[-2:])  # forward
        loss.backward()
        optim.step()
        return loss.item()

    trainer = Engine(trainer_step)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    def train_validation_step(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = batch
            loss = model(x, engine.state, *y[-2:])
            return loss.item()

    def validation_step(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = batch
            loss = model(x, engine.state, y[1])
            return loss.item()

    train_evaluator = Engine(train_validation_step)
    valid_evaluator = Engine(validation_step)
    test_evaluator = Engine(validation_step)

    def score_function(engine):
        return -engine.state.output

    model_checkpoint = ModelCheckpoint(root, 'checkpoint_phase2',
                                       n_saved=1,
                                       score_function=score_function,
                                       score_name="total_loss",
                                       require_empty=False)

    valid_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint,
                                      {"model": model})

    return trainer, train_evaluator, valid_evaluator, test_evaluator
