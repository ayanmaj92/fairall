"""
Authors: Miriam Rateike, Ayan Majumdar
Based on an older version of https://github.com/adrianjav/heterogeneous_vaes
"""

import torch
from torch.optim import Adam
from ignite.engine import Engine, Events
from ignite.handlers import TerminateOnNan, ModelCheckpoint
from utils.miscelanea import print_epoch_value
from .metrics import NoAverage


def create_trainer(model, train_dataset, cfg):
    """
    Creates the phase 1 trainer
    @param model: VAE model
    @param train_dataset: training dataset
    @param cfg: config
    @return: trainer and training, valid, test evaluators
    """
    max_epochs = cfg['trainer1']['epochs']
    trainer, train_evaluator, valid_evaluator, test_evaluator = build_trainer(model, train_dataset,
                                                                              cfg['optimizer']['params'][
                                                                                  'learning_rate'], cfg['save_dir'])
    metrics = ['-elbo', 'kl_z', '-re']

    no_average = NoAverage(metrics)
    no_average.attach(trainer, print_epoch_value, trainer, vars=metrics, max_epochs=cfg['trainer1']['epochs'],
                      print_every=cfg['trainer']['print_every'], evaluation='trainer')
    no_average.attach(train_evaluator, print_epoch_value, trainer, vars=metrics, max_epochs=cfg['trainer1']['epochs'],
                      print_every=cfg['trainer']['print_every'], evaluation='train_evaluator')
    no_average.attach(valid_evaluator, print_epoch_value, trainer, vars=metrics, max_epochs=cfg['trainer1']['epochs'],
                      print_every=cfg['trainer']['print_every'], evaluation='valid_evaluator')
    no_average.attach(test_evaluator, print_epoch_value, trainer, vars=metrics, max_epochs=cfg['trainer1']['epochs'],
                      print_every=cfg['trainer']['print_every'], evaluation='test_evaluator')

    return trainer, train_evaluator, valid_evaluator, test_evaluator


def build_trainer(model, dataset, learning_rate, root):
    """
    Build the phase 1 trainer
    @param model: VAE model
    @param dataset: training dataset
    @param learning_rate: optimizer learning rate
    @param root: root directory (for saving checkpoints)
    @return: trainer, training, valid, test evaluators
    """
    optim = Adam([{'params': model.parameters()}, {'params': dataset.parameters()}], lr=learning_rate)

    def trainer_step(engine, batch):
        model.train()
        optim.zero_grad()
        x, y = batch
        loss = model(x, engine.state, y[-1])  # forward
        loss.backward()
        optim.step()
        return loss.item()

    trainer = Engine(trainer_step)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

    def validation_step(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = batch
            loss = model(x, engine.state, y[-1])
            return loss.item()

    train_evaluator = Engine(validation_step)
    valid_evaluator = Engine(validation_step)
    test_evaluator = Engine(validation_step)

    def score_function(engine):
        return -engine.state.output

    model_checkpoint = ModelCheckpoint(root, 'checkpoint_phase1',
                                       n_saved=1,
                                       score_function=score_function,
                                       score_name="total_loss",
                                       require_empty=False)

    valid_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint,
                                      {"model": model})

    return trainer, train_evaluator, valid_evaluator, test_evaluator
