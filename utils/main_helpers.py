"""
Authors: Ayan Majumdar, Miriam Rateike
"""

import os
import sys
import subprocess
from datetime import datetime
import json
from ignite.engine import Events

import lipstd
import utils.args_parser as argtools
from models import create_model
from utils.datasets import get_dataloader
from utils.loop_trainer import create_loop_trainer_2
from utils.metrics import compute_accuracy
from lipstd.utils import flatten
import utils.plotting as plt
from utils.miscelanea import imputation_error, evaluate_model_generalized
from utils.policies import *
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from utils.trainer import create_trainer
from utils.timer import Timer


def evaluate_all(model, prob_model, loader, valid_loader, test_loader, loss_function, cfg, writer, phase):
    """
    Function that evaluates on all datasets.
    @param model: VAE model
    @param prob_model: probabilistic model
    @param loader: training data loader
    @param valid_loader: validation data loader
    @param test_loader: test data loader
    @param loss_function: loss function used to train [ours (fairall), niki (fairlog)]
    @param cfg: config
    @param writer: writer object of ignite
    @param phase: which phase of learning
    @return: evaluated results of training, validation and test
    """
    model.eval()
    o_train = evaluate(model, prob_model, loader, writer, cfg['save_dir'], phase=phase, type_='Train',
                       loss_function=loss_function)
    o_valid = evaluate(model, prob_model, valid_loader, writer, cfg['save_dir'], phase=phase, type_='Valid',
                       loss_function=loss_function)
    o_test = evaluate(model, prob_model, test_loader, writer, cfg['save_dir'], phase=phase, type_='Test',
                      loss_function=loss_function)
    return o_train, o_valid, o_test


def save_results_json(out_train, out_valid, out_test, save_dir, phase):
    """
    Function to dump all results to JSON. This JSON can be used for analyses.
    @param out_train: training results
    @param out_valid: validation results
    @param out_test: test results
    @param save_dir: save location
    @param phase: which phase of learning
    @return: none
    """
    with open(os.path.join(save_dir, f'output{phase}train.json'), 'w') as f:
        json.dump(out_train, f)
    with open(os.path.join(save_dir, f'output{phase}valid.json'), 'w') as f:
        json.dump(out_valid, f)
    with open(os.path.join(save_dir, f'output{phase}test.json'), 'w') as f:
        json.dump(out_test, f)


def phase_2(p2_model, prob_model, loader, val_loader, test_loader, cf_loader, cfg,
            policy_phase_2, writer, time_dict, save_dir, ran_phase1, is_warmup):
    """
    Function for phase 2 learning. Also used for the intermediate warmup.
    @param p2_model: VAE model object for phase 2
    @param prob_model: probabilistic model
    @param loader: training data loader
    @param val_loader: validation data loader
    @param test_loader: test data loader
    @param cf_loader: counterfactual data loader (None if real-world or not existent)
    @param cfg: config
    @param policy_phase_2: policy to use in phase 2
    @param writer: writer object of ignite
    @param time_dict: dict containing time information (from phase 1)
    @param save_dir: save location
    @param ran_phase1: whether phase1 was run
    @param is_warmup: are we doing warmup for this call or not
    @return: outputs on the data, cfg, time_dict, writer
    """
    # Get the loop trainers.
    loop_trainer, train_evaluator, valid_evaluator, test_evaluator = \
        create_loop_trainer_2(p2_model, loader.dataset, cfg, policy_phase_2)

    epoch_every_step = cfg['trainer2']['epochs_per_ts']

    @loop_trainer.on(Events.EPOCH_COMPLETED(every=epoch_every_step))
    def log_training_results(trainer):
        train_evaluator.run(loader)

    @loop_trainer.on(Events.EPOCH_COMPLETED(every=epoch_every_step))
    def log_validation_results(trainer):
        valid_evaluator.run(val_loader)

    @loop_trainer.on(Events.EPOCH_COMPLETED(every=epoch_every_step))
    def log_test_results(trainer):
        test_evaluator.run(test_loader)

    if not is_warmup:
        timer2 = Timer()
        if ran_phase1:
            time1 = time_dict['phase1_time']
        else:
            time1 = 0

        @loop_trainer.on(Events.STARTED)
        def start_timer(trainer):
            timer2.tic('train')

        @loop_trainer.on(Events.COMPLETED)
        def end_timer(trainer):
            time2 = timer2.toc('train')
            writer.add_scalar(f'Time/phase2_train_time', time2, trainer.state.epoch)
            print("/// TRAINING TIME: ", time2)
            time_dict['phase2_train_time'] = time2
            if 'phase1-ckpt' not in cfg['model']['params']:
                phase1_ckpt = None
            else:
                phase1_ckpt = cfg['model']['params']['phase1-ckpt']
            if phase1_ckpt is None and cfg['trainer1']['training']:
                writer.add_scalar(f'Time/total_train_time', time1 + time2, trainer.state.epoch)
                print("/// Total TRAINING TIME: ", time1 + time2)
                time_dict['total_train_time'] = time1 + time2

    if is_warmup:
        cf_data = None
        flag_util_gnd_truth = False
    else:
        flag_util_gnd_truth = cfg['dataset']['params2']['util_gnd']
        if not cfg['dataset']['params2']['load_cf']:
            cf_data = None
        else:
            cf_data = cf_loader.dataset

    @loop_trainer.on(Events.COMPLETED)
    def evaluate_after_training(trainer):
        global output_train_2, output_valid_2, output_test_2, train_2_metrics, val_2_metrics, test_2_metrics
        output_train_2, output_valid_2, output_test_2 = {}, {}, {}
        output_train_2['metrics'] = evaluate_model_generalized(engine=train_evaluator, model=p2_model,
                                                               prob_model=prob_model,
                                                               dataset=loader.dataset,
                                                               phase_num='Phase2', policy=policy_phase_2,
                                                               costs=costs, loss=loss_function,
                                                               trainer_state=loop_trainer, state='final',
                                                               metric_dct=train_2_metrics,
                                                               epochs_per_step=epoch_every_step)
        output_valid_2['metrics'] = evaluate_model_generalized(engine=valid_evaluator, model=p2_model,
                                                               prob_model=prob_model,
                                                               dataset=val_loader.dataset,
                                                               phase_num='Phase2', policy=policy_phase_2, costs=costs,
                                                               loss=loss_function,
                                                               trainer_state=loop_trainer, state='final',
                                                               metric_dct=val_2_metrics,
                                                               epochs_per_step=epoch_every_step)
        output_test_2['metrics'] = evaluate_model_generalized(engine=test_evaluator, model=p2_model,
                                                              prob_model=prob_model,
                                                              dataset=test_loader.dataset,
                                                              phase_num='Phase2', policy=policy_phase_2, costs=costs,
                                                              loss=loss_function,
                                                              trainer_state=loop_trainer, state='final',
                                                              metric_dct=test_2_metrics,
                                                              epochs_per_step=epoch_every_step,
                                                              cf_dataset=cf_data,
                                                              util_gnd=flag_util_gnd_truth)

    costs = cfg['model']['params']['costs']
    loss_function = cfg['model']['params']['loss_function']
    global train_2_metrics, val_2_metrics, test_2_metrics
    train_2_metrics, val_2_metrics, test_2_metrics = {}, {}, {}

    train_evaluator.add_event_handler(Events.EPOCH_COMPLETED, evaluate_model_generalized, model=p2_model,
                                      prob_model=prob_model, dataset=loader.dataset,
                                      epochs_per_step=cfg['trainer2']['epochs_per_ts'],
                                      metric_dct=train_2_metrics, trainer_state=loop_trainer,
                                      phase_num='Phase2',
                                      policy=policy_phase_2, costs=costs, loss=loss_function, state='epoch')
    valid_evaluator.add_event_handler(Events.EPOCH_COMPLETED, evaluate_model_generalized,
                                      trainer_state=loop_trainer, model=p2_model,
                                      prob_model=prob_model,
                                      dataset=val_loader.dataset,
                                      epochs_per_step=cfg['trainer2']['epochs_per_ts'],
                                      metric_dct=val_2_metrics, phase_num='Phase2', policy=policy_phase_2,
                                      costs=costs, loss=loss_function, state='epoch')
    test_evaluator.add_event_handler(Events.EPOCH_COMPLETED, evaluate_model_generalized,
                                     trainer_state=loop_trainer, model=p2_model,
                                     prob_model=prob_model,
                                     dataset=test_loader.dataset,
                                     epochs_per_step=cfg['trainer2']['epochs_per_ts'],
                                     metric_dct=test_2_metrics, phase_num='Phase2', policy=policy_phase_2,
                                     costs=costs, loss=loss_function, state='epoch',
                                     cf_dataset=cf_data, util_gnd=flag_util_gnd_truth)

    if is_warmup:
        print("Phase Warmup Begins")
    else:
        print("Phase 2 Begins")
    epoch_every_timestep = cfg['trainer2']['epochs_per_ts']
    max_epochs_phase_2 = epoch_every_timestep * cfg['trainer2']['time_steps']  # Total epochs is this now.
    # Run model
    train(loop_trainer, loader, max_epochs_phase_2)
    if is_warmup:
        argtools.save_yaml(cfg, file_path=os.path.join(save_dir, 'hparams_warmup.yaml'))
    else:
        argtools.save_yaml(cfg, file_path=os.path.join(save_dir, 'hparams_phase2.yaml'))
    return output_train_2, output_valid_2, output_test_2, cfg, time_dict, writer


def phase_1(p1_model, p1_prob_model, p1_loader, p1_valid_loader, p1_test_loader,
            cfg, writer, time_dict, save_dir):
    """
    Function for phase 1 learning.
    @param p1_model: VAE model
    @param p1_prob_model: probabilistic model
    @param p1_loader: train data loader
    @param p1_valid_loader: validation data loader
    @param p1_test_loader: test data loader
    @param cfg: config
    @param writer: writer object
    @param time_dict: dict for time information
    @param save_dir: save location
    @return: outputs on the data, cfg, time_dict, writer
    """

    p1_trainer, p1_train_evaluator, \
        p1_valid_evaluator, p1_test_evaluator = create_trainer(p1_model, p1_loader.dataset, cfg)
    print_info(p1_model, p1_loader)

    @p1_trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        p1_train_evaluator.run(p1_loader)

    @p1_trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        p1_valid_evaluator.run(p1_valid_loader)

    @p1_trainer.on(Events.EPOCH_COMPLETED)
    def log_test_results(trainer):
        p1_test_evaluator.run(p1_test_loader)

    timer = Timer()

    @p1_trainer.on(Events.STARTED)
    def start_timer(trainer):
        timer.tic('train')

    @p1_trainer.on(Events.COMPLETED)
    def end_timer(trainer):
        global time
        time = timer.toc('train')
        writer.add_scalar(f'Time/phase1_train_time', time, trainer.state.epoch)
        print("//// Time", time)
        time_dict['phase1_time'] = time

    @p1_trainer.on(Events.COMPLETED)
    def evaluate_after_training(trainer):
        global output_train, output_valid, output_test, train1_metrics, val1_metrics, test1_metrics
        output_train, output_valid, output_test = {}, {}, {}
        output_train['metrics'] = evaluate_model_generalized(engine=p1_train_evaluator, model=p1_model,
                                                             prob_model=p1_prob_model, dataset=p1_loader.dataset,
                                                             phase_num='Phase1', policy=None, costs=costs,
                                                             loss=loss_function, trainer_state=p1_trainer,
                                                             state='final', metric_dct=train1_metrics,
                                                             epochs_per_step=None)
        output_valid['metrics'] = evaluate_model_generalized(engine=p1_valid_evaluator, model=p1_model,
                                                             prob_model=p1_prob_model,
                                                             dataset=p1_valid_loader.dataset,
                                                             phase_num='Phase1', policy=None, costs=costs,
                                                             loss=loss_function, trainer_state=p1_trainer,
                                                             state='final', metric_dct=val1_metrics,
                                                             epochs_per_step=None)
        output_test['metrics'] = evaluate_model_generalized(engine=p1_test_evaluator, model=p1_model,
                                                            prob_model=p1_prob_model,
                                                            dataset=p1_test_loader.dataset,
                                                            phase_num='Phase1', policy=None, costs=costs,
                                                            loss=loss_function, trainer_state=p1_trainer,
                                                            state='final', metric_dct=test1_metrics,
                                                            epochs_per_step=None)

    costs = cfg['model']['params']['costs']
    loss_function = cfg['model']['params']['loss_function']

    global train1_metrics, val1_metrics, test1_metrics
    train1_metrics, val1_metrics, test1_metrics = {}, {}, {}

    p1_train_evaluator.add_event_handler(Events.EPOCH_COMPLETED, evaluate_model_generalized, model=p1_model,
                                         prob_model=p1_prob_model, metric_dct=train1_metrics, epochs_per_step=None,
                                         dataset=p1_loader.dataset, trainer_state=p1_trainer,
                                         phase_num='Phase1', policy=None, costs=costs, loss=loss_function,
                                         state='train')
    p1_valid_evaluator.add_event_handler(Events.EPOCH_COMPLETED, evaluate_model_generalized, model=p1_model,
                                         prob_model=p1_prob_model, metric_dct=val1_metrics, epochs_per_step=None,
                                         dataset=p1_valid_loader.dataset, trainer_state=p1_trainer,
                                         phase_num='Phase1', policy=None, costs=costs, loss=loss_function,
                                         state='train')
    p1_test_evaluator.add_event_handler(Events.EPOCH_COMPLETED, evaluate_model_generalized, model=p1_model,
                                        prob_model=p1_prob_model, metric_dct=test1_metrics, epochs_per_step=None,
                                        dataset=p1_test_loader.dataset, trainer_state=p1_trainer,
                                        phase_num='Phase1', policy=None, costs=costs, loss=loss_function,
                                        state='train')

    print("Start training Phase 1 from scratch")
    train(p1_trainer, p1_loader, cfg['trainer1']['epochs'])
    argtools.save_yaml(cfg, file_path=os.path.join(save_dir, 'hparams_phase1.yaml'))
    cfg['model']['params']['phase1-hparams'] = os.path.join(save_dir, 'hparams_phase1.yaml')
    return output_train, output_valid, output_test, cfg, time_dict, writer


def get_policy(learn_policy, cfg):
    """
    Function to return the policy we want to use in phase 2
    @param learn_policy: which policy (our paper's method is FZ, with FZ method as DEC)
    @param cfg: config
    @return: policy
    """
    if learn_policy == 'QXS':
        policy_phase_2 = PolicyQxs(config=cfg)
    elif learn_policy == 'PZS':
        policy_phase_2 = PolicyPzs(config=cfg)
    elif learn_policy == 'FZ':
        # In warmup, num epochs should be num warmup_ts
        policy_phase_2 = PolicyFz(config=cfg, epochs=cfg['trainer2']['warmup_ts'])
    elif learn_policy == 'NA':
        policy_phase_2 = PolicyNA()
    else:
        raise NotImplementedError
    return policy_phase_2


def transfer_model(phase1_ckpt, p1_prob_model, p2_model, warmup_prob_model, cfg, ran_phase1, req_semisup, conditional):
    """
    Helper function to transfer weights from pre-trained phase 1 model to phase 2 model.
    @param phase1_ckpt: phase 1 model checkpoint
    @param p1_prob_model: phase 1 prob model
    @param p2_model: phase 2 VAE model
    @param warmup_prob_model: warmup phase prob model
    @param cfg: config dict
    @param ran_phase1: flag to check if we ran phase 1 in this run or not.
    @param req_semisup: flag to check if we are to do semisup learning.
    @param conditional: flag to check if we will model conditional.
    @return: phase 2 model object with transferred weights.
    """
    if (phase1_ckpt is not None) or ran_phase1:
        # Only if Phase 1 training happens will we do transfer learning! Otherwise we directly load random model!
        # Load best phase1 model
        if ran_phase1 is True:
            print("Loading best model from Phase 1 ....")
            path = cfg['save_dir']
            files = []
            for i in os.listdir(path):
                if os.path.isfile(os.path.join(path, i)) and 'checkpoint_phase1' in i:
                    files.append(i)
            ckpt_path = f'{cfg["save_dir"]}/{files[0]}'
        else:
            ckpt_path = phase1_ckpt
        print("Phase 1 ckpt:", ckpt_path)
        checkpoint = torch.load(ckpt_path)
        print("Loaded model from Phase 1!")
        cfg1 = argtools.parse_args(cfg['model']['params']['phase1-hparams'])
        if p1_prob_model is None:
            p1_prob_model = lipstd.get_likelihood(cfg1['probabilistic']['probabilistic_model1']).eval()
        if req_semisup:
            p1_model = create_model('cvae', p1_prob_model, conditional, cfg1, phase=1)
        else:
            p1_model = create_model(cfg['model']['name'], p1_prob_model, conditional, cfg1, phase=1)

        p1_model.load_state_dict(checkpoint)

        # These are the weights that will need to be transferred from the CVAE!
        p1_model_keys = [k for k in list(p1_model.state_dict().keys()) if 'encoder' in k or 'decoder' in k]
        print("Weights to transfer")
        print(p1_model_keys)
        # This encoder weight and bias will need to be manually transferred!
        enc_weight = p1_model_keys[0]
        # This decoder weight and bias will need to be manually transferred!
        dec_weight, dec_bias = p1_model_keys[-2], p1_model_keys[-1]
        # So, these are the weights that we can directly transfer because no dim. size problems!
        p1_model_keys = p1_model_keys[1:-2]
        with torch.no_grad():
            for k in p1_model_keys:
                p2_model.state_dict()['cvae.' + k][:, ] = p1_model.state_dict()[k]
            # Now we need special care!
            # 1. Decoder last layer that goes to the reconstruction of data. Changes from p(X) to p(X, U).
            diff_prob_size = warmup_prob_model.num_params - p1_prob_model.num_params
            p2_model.state_dict()['cvae.' + dec_weight][:-diff_prob_size, :] = p1_model.state_dict()[dec_weight]
            p2_model.state_dict()['cvae.' + dec_bias][:-diff_prob_size] = p1_model.state_dict()[dec_bias]
            # 2. Encoder first layer that goes from data to hidden. Changes from (X, S) to (X, U, S).
            p2_model.state_dict()['cvae.' + enc_weight][:, :-2] = p1_model.state_dict()[enc_weight][:, :-1]
            p2_model.state_dict()['cvae.' + enc_weight][:, -1] = p1_model.state_dict()[enc_weight][:, -1]
        print("Transferred Phase 1 model to Phase 2!")
    return p2_model


def init_loaders(cfg, prob_model, phase, warmup):
    """
    Function to initialise data loaders for a phase
    @param cfg: config
    @param prob_model: probabilistic model
    @param phase: which phase of learning 1 or 2
    @param warmup: is this warmup call?
    @return: train, valid, test, (counterfactual if there) loaders
    """
    loader = get_dataloader(cfg, prob_model, phase=phase, test=False, val=False, warmup=warmup)
    # warmup is False for valid and test data
    valid_loader = get_dataloader(cfg, prob_model, phase=phase, test=False, val=True, warmup=False)
    test_loader = get_dataloader(cfg, prob_model, phase=phase, test=True, val=False, warmup=False)
    if cfg['dataset']['params2']['load_cf'] and phase == 2:
        cf_loader = get_dataloader(cfg, prob_model, phase=phase, test=True, val=False, warmup=False, cf=True)
        return loader, valid_loader, test_loader, cf_loader
    else:
        return loader, valid_loader, test_loader


def check_predict_from_data(loader, test_loader, writer, pred_var, phase):
    """
    Helper function to predict either sensitive S or utility U from data features
    @param loader: train data loader
    @param test_loader: test data loader
    @param writer: writer object
    @param pred_var: predict 'S' (sensitive) or 'U' (utility)
    @param phase: which phase 1 or 2
    @return: updated writer
    """
    X_tr = loader.dataset.data.numpy()[:, :-1]
    X_ts = test_loader.dataset.data.numpy()[:, :-1]
    s = loader.dataset.sens  # needs to be cond index
    s_test = test_loader.dataset.sens  # needs to be cond index
    if pred_var == 'S':
        S_tr = s.numpy().ravel()
        S_ts = s_test.numpy().ravel()
        p_var, p_var_test = S_tr, S_ts
    elif pred_var == 'U':
        U_tr = loader.dataset.data.numpy()[:, -2]
        U_ts = test_loader.dataset.data.numpy()[:, -2]
        p_var, p_var_test = U_tr, U_ts
        X_tr = np.hstack((X_tr, s[:, np.newaxis]))
        X_ts = np.hstack((X_ts, s_test[:, np.newaxis]))
    else:
        raise NotImplementedError
    clf = LogisticRegression(class_weight='balanced').fit(X_tr, p_var)
    score = clf.score(X_ts, p_var_test)
    if pred_var == 'S':
        print(f"Phase {phase} predictability of s from x ACC:", score)
        writer.add_scalar(f'Data/S-from-X ACC', score, phase)
    elif pred_var == 'U':
        print(f"Phase {phase} predictability of u from X,S ACC:", score)
        writer.add_scalar(f'Data/U-from-XS ACC', score, 2)
    return writer


def set_latent_size(cfg, prob_model):
    if cfg['model']['params']['latent_size'] is None:
        if cfg['model']['name'] in ['vae', 'cvae', 'sscvae']:
            len_ = len(prob_model)
            cfg['model']['params']['latent_size'] = max(1, int(len_ * 0.75 + 0.5))
            print("Latent size:", cfg['model']['params']['latent_size'])
        else:
            raise NotImplementedError
    return cfg


def process_conditional(cfg):
    if cfg['trainer1']['training']:
        condition_index1 = len(cfg['probabilistic']['probabilistic_model1']) - 1
        del cfg['probabilistic']['probabilistic_model1'][condition_index1]
        cfg['probabilistic']['categoricals1'].remove(condition_index1)
    if cfg['trainer2']['training']:
        condition_index2 = len(cfg['probabilistic']['probabilistic_model2']) - 1
        del cfg['probabilistic']['probabilistic_model2'][condition_index2]
        cfg['probabilistic']['categoricals2'].remove(condition_index2)
    return cfg


def process_parser_args(parser):
    """
    Add necessary arguments to arg parser, process and generate config dict.
    @param parser: parser object
    @return: cfg dict, args
    """
    parser.add_argument('--dataset_file', default='_params/dataset_SCB_11.yaml', type=str,
                        help='path to configuration file for the dataset')
    parser.add_argument('--model_file', default='_params/model_fairall.yaml', type=str,
                        help='path to configuration file for the dataset')
    parser.add_argument('--trainer_file', default='_params/trainer.yaml', type=str,
                        help='path to configuration file for the training')
    parser.add_argument('-d', '--dataset_dict', action=argtools.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...",
                        help='manually define dataset configurations as string: KEY1=VALUE1+KEY2=VALUE2+...')
    parser.add_argument('-m', '--model_dict', action=argtools.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...",
                        help='manually define model configurations as string: KEY1=VALUE1+KEY2=VALUE2+...')
    parser.add_argument('-o', '--optim_dict', action=argtools.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...",
                        help='manually define optimizer configurations as string: KEY1=VALUE1+KEY2=VALUE2+...')
    parser.add_argument('-t', '--trainer_dict', action=argtools.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...",
                        help='manually define trainer configurations as string: KEY1=VALUE1+KEY2=VALUE2+...')
    parser.add_argument('-s', '--seed', default=1, type=int, help='set random seed')
    parser.add_argument('-r', '--result_dir', default='', type=str, help='directory for storing results')

    args = parser.parse_args()

    cfg = argtools.parse_args(args.dataset_file)
    cfg.update(argtools.parse_args(args.model_file))
    cfg.update(argtools.parse_args(args.trainer_file))

    if len(args.result_dir) > 0:
        cfg['root_dir'] = args.result_dir
    if int(args.seed) >= 0:
        cfg['seed'] = int(args.seed)

    if args.dataset_dict is not None:
        cfg['dataset']['params2'].update(args.dataset_dict)
    if args.model_dict is not None:
        cfg['model']['params'].update(args.model_dict)
    if args.optim_dict is not None:
        cfg['optimizer']['params'].update(args.optim_dict)
    if args.trainer_dict is not None:
        cfg['trainer1'].update(args.trainer_dict)

    cfg['dataset']['params'] = cfg['dataset']['params1'].copy()
    cfg['dataset']['params'].update(cfg['dataset']['params2'])
    # print(args.dataset_dict)
    # print(cfg)
    return cfg, args


def validate(cfg):
    """
    Validate and correct config dict
    @param cfg: config dict
    @return: validated config dict
    """
    time_stamp = datetime.today()
    cfg['timestamp'] = time_stamp.strftime('%Y-%m-%d-%H:%M:%S')

    dataset = cfg['dataset']['name']
    if 'load_cf' not in cfg['dataset']['params2']:
        cfg['dataset']['params2']['load_cf'] = False
    if 'util_gnd' not in cfg['dataset']['params2']:
        cfg['dataset']['params2']['util_gnd'] = False
    if 'SCB' not in dataset:
        assert cfg['dataset']['params2']['load_cf'] is False, "Cannot load counterfactual for non-synthetic!"

    assert cfg['model']['params']['loss_function'] in [Cte.LOSS_FAIRALL, Cte.LOSS_FAIRLAB, Cte.LOSS_FAIRLOG], \
        'dont recognize this loss'

    assert cfg['model']['params']['learn_policy'] in [Cte.POL_FZ, Cte.POL_PZS, Cte.POL_NA, Cte.POL_QXS], \
        'dont recognize this learn_poicy'

    assert cfg['model']['params']['fz_method'] in [Cte.FZ_LAB, Cte.FZ_CLF, Cte.FZ_DEC, None], \
        'dont recognize this fz_method'

    if cfg['model']['params']['loss_function'] in [Cte.LOSS_FAIRLAB]:
        assert cfg['model']['params']['learn_policy'] in [Cte.POL_FZ, Cte.POL_PZS,
                                                          Cte.POL_NA], 'cannot use classifier policy'
        if cfg['model']['params']['learn_policy'] == Cte.POL_FZ:
            assert cfg['model']['params'][
                       'fz_method'] == Cte.FZ_LAB, 'can only learn from labelled data when doing ips_loss'

    if cfg['model']['params']['loss_function'] in [Cte.LOSS_FAIRLOG]:
        assert cfg['model']['params']['learn_policy'] in [Cte.POL_QXS,
                                                          Cte.POL_NA], 'cannot use decoder or latent space policy'

    if cfg['model']['params']['loss_function'] == Cte.LOSS_FAIRLOG:
        cfg['trainer1']['training'] = False

    phase1_training = cfg['trainer1']['training']
    phase2_training = cfg['trainer2']['training']

    if cfg['model']['params']['loss_function'] != Cte.LOSS_FAIRLOG:
        # If not doing Niki loss, then forcefully set lambda to 0 because no constraints for fairness!
        cfg['model']['params']['lambda'] = 0.0

    assert not (phase1_training is False and phase2_training is False), \
        "Both phase1 and phase2 training cannot be False!"

    cfg['probabilistic'] = {}
    cfg['probabilistic']['probabilistic_model1'] = None
    cfg['probabilistic']['categoricals1'] = None

    if phase1_training:
        print("Generate prob_model for Phase 1")
        arguments1 = ['./read_types.sh', f'datasets/{dataset}/data1_types.csv']
        proc = subprocess.Popen(arguments1, stdout=subprocess.PIPE)
        out = eval(proc.communicate()[0].decode('ascii'))
        cfg['probabilistic']['probabilistic_model1'] = out['probabilistic model']
        cfg['probabilistic']['categoricals1'] = out['categoricals']

    cfg['probabilistic']['probabilistic_model2'] = cfg['probabilistic']['probabilistic_model1']
    cfg['probabilistic']['categoricals2'] = cfg['probabilistic']['categoricals1']

    if phase2_training:
        print("Generate prob_model for Phase 2")
        arguments2 = ['./read_types.sh', f'datasets/{dataset}/data2_types.csv']
        proc = subprocess.Popen(arguments2, stdout=subprocess.PIPE)
        out = eval(proc.communicate()[0].decode('ascii'))
        cfg['probabilistic']['probabilistic_model2'] = out['probabilistic model']
        cfg['probabilistic']['categoricals2'] = out['categoricals']

    return cfg


def print_data_info(prob_model, data):
    """
    Prints info about the data and the probabilistic model on it
    @param prob_model: probabilistic model object
    @param data: data
    """
    prob_model.eval()
    print()
    print('#' * 20)
    print('Original data')
    x = data
    pos = 0
    for i, dist_i in enumerate(prob_model):
        print(f'range of [{i}={dist_i}]: {x[:, pos: pos + dist_i.domain_size].min()} '
              f'{x[:, pos: pos + dist_i.domain_size].max()}')
        pos += dist_i.domain_size

    prob_model.train()
    print()
    print(f'weights = {[x.item() for x in flatten(prob_model.scale)]}')
    print()
    print('Scaled data')

    x = prob_model >> data
    pos = 0
    for i, dist_i in enumerate(prob_model):
        print(f'range of [{i}={dist_i}]: {x[:, pos: pos + dist_i.domain_size].min()} '
              f'{x[:, pos: pos + dist_i.domain_size].max()}')
        pos += dist_i.domain_size
    print('#' * 20)
    print()


def print_info(model, loader):
    print('Dataset:', loader.dataset)
    print('Model:', model)


def train(trainer, loader, max_epochs):
    """
    Runs the ignite trainer
    @param trainer: trainer object
    @param loader: data loader
    @param max_epochs: maximum epochs
    """
    try:
        trainer.run(loader, max_epochs=max_epochs)
    except KeyboardInterrupt:
        from ignite.engine import Events
        trainer.fire_event(Events.COMPLETED)
        trainer.terminate()
        print(f'Training interrupted by keyboard.', file=sys.stderr)
        print('', file=sys.stderr)


@torch.no_grad()
def evaluate(model, prob_model, loader, writer, path, phase, type_, loss_function):
    """
    Calls some basic evaluation
    @param model: VAE model
    @param prob_model: probabilistic model
    @param loader: data loader
    @param writer: writer object
    @param path: root path dir
    @param phase: whether '1' or '2'
    @param type_: train, validation or test?
    @param loss_function: which loss was used?
    @return: results dictionary
    """
    model.eval()
    results = {}
    mask = getattr(loader.dataset, 'mask', None)
    mask_ones = torch.ones_like(mask)

    data = loader.dataset.data

    # Classifier's evaluation. Error and DP fairness.
    if phase in ['2'] and 'SSCVAE' in str(model):
        # Store CLF ERR U in eval.
        clf_probs = sigmoid(model.classify(data[:, :-1], data[:, -1], mask_ones))
        pred_u = torch.bernoulli(clf_probs).float()
        s = data[:, -1]
        pred_0, pred_1 = pred_u[s[:] == -1], pred_u[s[:] == 1]
        unfairness_dp = torch.abs(pred_0.mean() - pred_1.mean())
        clf_acc = compute_accuracy(data[:, -2], pred_u.flatten(), mask_ones, None)
        clf_err = 1.0 - clf_acc
        writer.add_scalar(phase + '_Eval_' + type_ + '/CLF_Err U', clf_err)
        results[type_ + '_CLF_ERR_U'] = clf_err.item()
        results[type_ + '_CLF_DP'] = unfairness_dp.item()

    # VAE model evaluation of reconstruction of X and U (if phase 2).
    # Also tests predictability of S and U from Z.
    if loss_function in [Cte.LOSS_FAIRLAB, Cte.LOSS_FAIRALL]:
        generated_data, z, _ = model.reconstruct(data, mask_ones)
        observed_error = imputation_error(prob_model, generated_data, data, mask_ones)
        is_discrete = lambda d: 'categorical' in str(d) or 'bernoulli' in str(d)
        # 3. Loop and write the errors to writer
        for idx, (err, dist) in enumerate(zip(observed_error, prob_model)):
            if ('SSCVAE' in str(model)) and (idx == (len(prob_model) - 1)):
                # If model is SSCVAE and the last dim, then this is the U.
                writer.add_scalar(phase + '_Eval_' + type_ + '/Err U', err)
                results[type_ + '_Err_U'] = err.item()
            else:
                if not is_discrete(dist):
                    writer.add_scalar(phase + '_Eval_' + type_ + f'/NRMSE X{idx}', err)
                    results[type_ + f'_NRMSE_X{idx}'] = err.item()
                else:
                    writer.add_scalar(phase + '_Eval_' + type_ + f'/ERR X{idx}', err)
                    results[type_ + f'_ERR_X{idx}'] = err.item()

        if type_ == 'Test':
            plt.plot_together([data, generated_data], prob_model, title='', legend=['original', 'generated'],
                              path=f'{path}/marginal' + '_phase_' + phase)
        z = z.numpy()
        # Predictability of s from z.
        _s = data[:, -1].unsqueeze(-1).int().numpy().ravel()
        z_tr, z_ts, s_tr, s_ts = train_test_split(z, _s, test_size=0.2, random_state=42)
        clf = LogisticRegression(class_weight='balanced').fit(z_tr, s_tr)
        score = clf.score(z_ts, s_ts)
        print(type_ + " Predictability of s from latent z ACC:", score)
        writer.add_scalar(phase + '_Eval_' + type_ + '/S-from-Z ERR', 1 - score)
        results[type_ + '_S-from-Z_ACC'] = score

        if type_ == 'Test':
            with open(f'{path}/pred-s-from-z-err' + '-ph-' + phase + '.txt', 'w') as f:
                f.write("err: " + str(1 - score))

        # Predictability of u from z.
        _u = data[:, -2].unsqueeze(-1).int().numpy().ravel()
        z_tr, z_ts, u_tr, u_ts = train_test_split(z, _u, test_size=0.2, random_state=43)
        clf = LogisticRegression(class_weight='balanced').fit(z_tr, u_tr)
        score = clf.score(z_ts, u_ts)
        print(type_ + " Predictability of u from latent z:", score)
        print(type_ + " ERROR predicting of u from latent z:", 1 - score)
        writer.add_scalar(phase + '_Eval_' + type_ + '/U-from-Z ERR', 1 - score)
        results[type_ + '_U-from-Z ERR'] = 1 - score

        if type_ == 'Test':
            with open(f'{path}/pred-u-from-z-err' + '-ph-' + phase + '.txt', 'w') as f:
                f.write("err " + str(1 - score))
        # Visualize z in 2-dim.
        if type_ == 'Test':
            if z.shape[1] > 2:
                print("To plot z, performing PCA...")
                pca = PCA(n_components=2)
                z = pca.fit_transform(z)
                print("PCA explained variance:", pca.explained_variance_ratio_)
                txt = "_pca"
            else:
                txt = ""
            plt.plot_z(data, z, title='latent z', path=f'{path}/latent{txt}' + '_phase_' + phase)

    return results
