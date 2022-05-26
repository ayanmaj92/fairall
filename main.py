"""
Authors: Ayan Majumdar, Miriam Rateike
"""

import argparse
from torch.utils.tensorboard import SummaryWriter
from utils.miscelanea import fix_seed
from utils.main_helpers import *
import numpy as np
import warnings
from datetime import date

torch.set_default_dtype(torch.float64)
# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
# Initialise timer dict
time_dict = {}

# Create arg parser, add arguments, parse from input.
parser = argparse.ArgumentParser(description=__doc__)
cfg, args = process_parser_args(parser)

# Validate arguments and create probabilistic model
cfg = validate(cfg)
# Creating directory
today = date.today()
today_date = today.strftime("%y%m%d")
save_dir = argtools.mkdir(os.path.join(cfg['root_dir'], today_date,
                                       argtools.get_experiment_folder(cfg),
                                       str(cfg['seed'])))
# Create writer.
writer = SummaryWriter(log_dir=f'{save_dir}')
# Fix random seed.
fix_seed(cfg['seed'])
# Set save directory to cfg.
cfg['save_dir'] = save_dir
# For now, supporting only CSV file reading.
cfg['file_type'] = 'csv'
print("Save directory:", save_dir)

# MAIN FUNCTION STARTS

# Load probabilistic model
conditional = False
if cfg['model']['name'] in ['cvae', 'sscvae']:
    conditional = cfg['model']['params']['conditional']
    if conditional:
        # Conditional not part of data we are learning to reconstruct. So reformat the model.
        cfg = process_conditional(cfg)
# Set flag variable for semi-supervised (Phase 2) learning.
if cfg['model']['name'] in ['sscvae']:
    req_semisup = True
else:
    req_semisup = False

#######################
# PHASE 1 PRETRAINING #
#######################
ran_phase1 = False
p1_prob_model = None

if cfg['trainer1']['training']:
    p1_prob_model = lipstd.get_likelihood(cfg['probabilistic']['probabilistic_model1']).eval()
    cfg = set_latent_size(cfg, p1_prob_model)
    cfg['semisup'] = False
    cfg['probabilistic']['categoricals'] = cfg['probabilistic']['categoricals1']
    # Initialise data loaders
    p1_loader, p1_valid_loader, p1_test_loader = init_loaders(cfg, p1_prob_model, phase=1, warmup=False)
    p1_prob_model.train()
    # Checking predictability of sensitive S from the data itself.
    if conditional:
        writer = check_predict_from_data(loader=p1_loader, test_loader=p1_test_loader,
                                         writer=writer, pred_var='S', phase=1)

    # Fill elements of self tensor with value 'nan' where mask is True. (where its true, its missing)
    if conditional:
        torch.masked_fill(p1_loader.dataset.data[:, :-1], ~p1_loader.dataset.mask, float('nan'))
    else:
        torch.masked_fill(p1_loader.dataset.data, ~p1_loader.dataset.mask, float('nan'))
    # Create the CVAE model.
    if req_semisup:
        p1_model = create_model('cvae', p1_prob_model, conditional, cfg, phase=1)
    else:
        p1_model = create_model(cfg['model']['name'], p1_prob_model, conditional, cfg, phase=1)

    # Run phase 1 or load the checkpoint (if exists)
    phase1_ckpt = cfg['model']['params']['phase1-ckpt']
    if phase1_ckpt is None:
        costs = cfg['model']['params']['costs']
        loss_function = cfg['model']['params']['loss_function']
        output_train, output_valid, output_test, cfg, time_dict, writer = \
            phase_1(p1_model, p1_prob_model, p1_loader, p1_valid_loader, p1_test_loader, cfg, writer, time_dict,
                    save_dir)
        ran_phase1 = True
    else:
        print("Loading Phase 1 trained unsupervised model from:", phase1_ckpt)
        ckpt = torch.load(phase1_ckpt)
        p1_model.load_state_dict(ckpt)
        costs = cfg['model']['params']['costs']
        loss_function = cfg['model']['params']['loss_function']
        print("Phase 1 Model loaded successfully!")

    model_parameters = filter(lambda p: p.requires_grad, p1_model.parameters())
    params = int(sum([np.prod(p.size()) for p in model_parameters]))

    # Call evaluate for all types of data.
    o_train, o_valid, o_test = evaluate_all(p1_model, p1_prob_model, p1_loader, p1_valid_loader, p1_test_loader,
                                            loss_function, cfg, writer, phase='1')

    # Save results
    if ran_phase1:
        output_train.update({"eval": o_train})
        output_valid.update({"eval": o_valid})
        output_test.update({"eval": o_test})
    else:
        output_train = {"eval": o_train}
        output_valid = {"eval": o_valid}
        output_test = {"eval": o_test}
    # Save the config as well
    output_train.update(argtools.flatten_cfg(cfg))
    output_valid.update(argtools.flatten_cfg(cfg))
    output_test.update(argtools.flatten_cfg(cfg))
    # Dump to JSON. All analysis can be done by reading the JSON files.
    save_results_json(output_train, output_valid, output_test, save_dir, '1')
    print("Phase 1 is done!")
    print(f"All results saved to: {save_dir}")
else:
    print("No Phase 1 training done. Going directly to Phase 2 training.")

######################################################
# PHASE 2 WARMUP (ALSO HAS TRANSFER LEARNING FROM 1) #
######################################################
EPOCH_PER_TS = cfg['trainer2']['epochs_per_ts']  # Number of epochs per time step
TOTAL_TS = cfg['trainer2']['time_steps']  # Number of time steps
SAMPLES_PER_TS = cfg['trainer2']['samples_per_ts']

if cfg['trainer2']['training']:
    print("Let us warm up before we start phase 2...")
    cfg['trainer2']['warmup'] = True

if cfg['trainer2']['training']:
    warmup_prob_model = lipstd.get_likelihood(cfg['probabilistic']['probabilistic_model2']).eval()

    # For phase 2 we should set some settings.
    cfg['semisup'] = req_semisup
    cfg['probabilistic']['categoricals'] = cfg['probabilistic']['categoricals2']
    cfg['trainer2']['epochs_per_ts'] = 1
    cfg['trainer2']['time_steps'] = cfg['trainer2']['warmup_ts']
    cfg['trainer2']['samples_per_ts'] = cfg['trainer2']['warmup_samples']

    # Initialise the data loaders
    warmup_loader, warmup_val_loader, warmup_test_loader = init_loaders(cfg, warmup_prob_model, phase=1, warmup=True)
    warmup_prob_model.train()

    # Create the model object for phase 2
    p2_model = create_model(cfg['model']['name'], warmup_prob_model, conditional, cfg, phase=2)

    # This is the bridge where we transfer!
    phase1_ckpt = cfg['model']['params']['phase1-ckpt']
    p2_model = transfer_model(phase1_ckpt, p1_prob_model, p2_model, warmup_prob_model,
                              cfg, ran_phase1, req_semisup, conditional)

    # Continuing
    learn_policy = cfg['model']['params']['learn_policy']
    policy_phase_2 = get_policy(learn_policy, cfg)

    # Call the main call function to run everything.
    output_train_w, output_valid_w, output_test_w, cfg, time_dict, writer = \
        phase_2(p2_model, warmup_prob_model, warmup_loader, warmup_val_loader, warmup_test_loader, None, cfg,
                policy_phase_2, writer, time_dict, save_dir, ran_phase1, is_warmup=True)

    p2_model.eval()
    costs = cfg['model']['params']['costs']
    loss_function = cfg['model']['params']['loss_function']
    # Run evaluations
    o_train, o_valid, o_test = evaluate_all(p2_model, warmup_prob_model, warmup_loader, warmup_val_loader,
                                            warmup_test_loader, loss_function, cfg, writer, phase='w')
    # Save results
    try:
        output_train_w.update({"eval": o_train})
    except:
        output_train_w = {"eval": o_train}
    output_train_w.update({'setup': argtools.flatten_cfg(cfg)})
    try:
        output_valid_w.update({"eval": o_valid})
    except:
        output_valid_w = {"eval": o_valid}
    output_valid_w.update({'setup': argtools.flatten_cfg(cfg)})
    try:
        output_test_w.update({"eval": o_test})
    except:
        output_test_w = {"eval": o_test}
    output_test_w.update({'setup': argtools.flatten_cfg(cfg)})
    # Dump to JSON. All analysis can be done by reading the JSON files.
    save_results_json(output_train_w, output_valid_w, output_test_w, save_dir, 'w')
    print(f"All results saved to: {save_dir}")
    print("Warmup done. Model loaded successfully after warmup (before Phase 2).")

###########################
# PHASE 2 DECISION MAKING #
###########################

if cfg['trainer2']['training'] and not cfg['trainer2']['only_warmup']:
    print("Starting decision making phase 2...")
    cfg['trainer2']['warmup'] = False
    p2_prob_model = lipstd.get_likelihood(cfg['probabilistic']['probabilistic_model2']).eval()
    # For phase 2 we should go back to old setting of semisup.
    cfg['semisup'] = req_semisup
    cfg['probabilistic']['categoricals'] = cfg['probabilistic']['categoricals2']
    # Resetting certain params for actual phase 2
    cfg['trainer2']['epochs_per_ts'] = EPOCH_PER_TS
    cfg['trainer2']['time_steps'] = TOTAL_TS
    cfg['trainer2']['samples_per_ts'] = SAMPLES_PER_TS

    # Set epochs to train policy the same as number of epochs per time-step (default 1).
    if learn_policy == 'FZ':
        policy_phase_2.epochs = EPOCH_PER_TS

    # Load counterfactual data?
    if cfg['dataset']['params2']['load_cf']:
        p2_loader, p2_val_loader, p2_test_loader, p2_cf_loader = init_loaders(cfg, p2_prob_model, phase=2, warmup=False)
    else:
        p2_loader, p2_val_loader, p2_test_loader = init_loaders(cfg, p2_prob_model, phase=2, warmup=False)
        p2_cf_loader = None

    p2_prob_model.train()
    cfg = set_latent_size(cfg, p2_prob_model)

    # Basic checks of the data. Predicting sensitive S and utility U from features X directly.
    if conditional:
        writer = check_predict_from_data(loader=p2_loader, test_loader=p2_test_loader,
                                         writer=writer, pred_var='S', phase=2)
        writer = check_predict_from_data(loader=p2_loader, test_loader=p2_test_loader,
                                         writer=writer, pred_var='U', phase=2)

    # Update the models at phase 2.
    if cfg["model"]["params"]["loss_function"] != Cte.LOSS_FAIRLOG:
        # CVAE needs updating only if we are not doing Niki.
        p2_model.cvae.prob_model = p2_prob_model
        p2_model.cvae.num_params = p2_prob_model.num_params
        p2_model.cvae.input_scaler = p2_prob_model.InputScaler
    p2_model.prob_model = p2_prob_model
    p2_model.num_params = p2_prob_model.num_params
    p2_model.input_scaler = p2_prob_model.InputScaler

    # Are we evaluating also to the hidden ground truth?
    flag_util_gnd_truth = cfg['dataset']['params2']['util_gnd']

    # Now we create trainer and train!
    p2_model.train()
    epoch_every_step = cfg['trainer2']['epochs_per_ts']
    if cfg['trainer2']['time_steps'] > 0:
        output_train_2, output_valid_2, output_test_2, cfg, time_dict, writer = \
            phase_2(p2_model=p2_model, prob_model=p2_prob_model, loader=p2_loader,
                    val_loader=p2_val_loader, test_loader=p2_test_loader,
                    cf_loader=p2_cf_loader, cfg=cfg, policy_phase_2=policy_phase_2,
                    writer=writer, time_dict=time_dict, save_dir=save_dir,
                    ran_phase1=ran_phase1,
                    is_warmup=False)

        p2_model.eval()
        path = cfg['save_dir']
        files = []
        # Get the ckpt exact location
        for i in os.listdir(path):
            if os.path.isfile(os.path.join(path, i)) and 'checkpoint_phase2' in i:
                files.append(i)
        ckpt_path = f'{cfg["save_dir"]}/{files[0]}'
        costs = cfg['model']['params']['costs']
        loss_function = cfg['model']['params']['loss_function']

        # Run evaluations
        o_train, o_valid, o_test = evaluate_all(p2_model, p2_prob_model, p2_loader, p2_val_loader,
                                                p2_test_loader, loss_function, cfg, writer, phase='2')

        model_parameters = filter(lambda p: p.requires_grad, p2_model.parameters())
        params = int(sum([np.prod(p.size()) for p in model_parameters]))

        # Save results.
        try:
            output_train_2.update({"eval": o_train})
        except:
            output_train_2 = {"eval": o_train}

        output_train_2.update({'setup': argtools.flatten_cfg(cfg)})
        output_train_2['setup'].update({'ckpt_file': ckpt_path,
                                        'num_parameters': params})
        try:
            output_valid_2.update({"eval": o_valid})
        except:
            output_valid_2 = {"eval": o_valid}
        output_valid_2.update({'setup': argtools.flatten_cfg(cfg)})
        output_valid_2['setup'].update({'ckpt_file': ckpt_path,
                                        'num_parameters': params})
        try:
            output_test_2.update({"eval": o_test})
        except:
            output_test_2 = {"eval": o_test}
        output_test_2.update({'setup': argtools.flatten_cfg(cfg)})
        output_test_2['setup'].update({'ckpt_file': ckpt_path,
                                       'num_parameters': params})
        # Dump to JSON. All analysis can be done by reading the JSON files.
        save_results_json(output_train_2, output_valid_2, output_test_2, save_dir, '2')
        print('Phase 2 training completed!')
        print(f"All results saved to: {save_dir}")
else:
    # Did nothing.
    print("No Phase 2 training. Done!")

writer.close()
with open(os.path.join(save_dir, 'time.json'), 'w') as f:
    json.dump(time_dict, f)
