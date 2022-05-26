"""
Authors: Ayan Majumdar, Miriam Rateike
Based on an older version of https://github.com/adrianjav/heterogeneous_vaes
"""

import torch
import torch.distributions as dists
from torch.nn.functional import sigmoid
from utils.metrics import compute_accuracy, compute_utility
from utils.constants import Cte


def fix_seed(seed) -> None:
    if seed is not None:
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def to_one_hot(x, size):
    x_one_hot = x.new_zeros(x.size(0), size)
    x_one_hot.scatter_(1, x.unsqueeze(-1).long(), 1).float()
    return x_one_hot


def get_distribution_by_name(name):
    return {'normal': dists.Normal, 'gamma': dists.Gamma, 'bernoulli': dists.Bernoulli,
            'categorical': dists.Categorical, 'lognormal': dists.LogNormal,
            'poisson': dists.Poisson, 'exponential': dists.Exponential}[name]


def print_metric_msg(fn, engine, msg, metrics):
    """
    Print out metrics
    """
    for name in metrics:
        value = fn(engine.state.metrics[name])
        msg += ' {} {:.5f}' if isinstance(value, float) else ' {} {}'

        if isinstance(value, torch.Tensor):
            value = value.tolist()
        msg = msg.format(name, value)
    print(msg)


def print_epoch_value(engine, metrics, trainer, max_epochs, print_every, evaluation):
    fn = lambda x: x
    if trainer.state.epoch % print_every != 0:
        return
    msg = f'Epoch {trainer.state.epoch} of {max_epochs} in {evaluation}:'
    print_metric_msg(fn, engine, msg, metrics)


def print_time_epoch_value(engine, metrics, trainer, print_every, max_timesteps, epochs_per_timestep, evaluation):
    fn = lambda x: x
    if trainer.state.epoch % print_every != 0:
        return
    time_step = (trainer.state.epoch + (epochs_per_timestep - 1)) // epochs_per_timestep
    epoch = (trainer.state.epoch + (epochs_per_timestep - 1)) % epochs_per_timestep + 1
    msg = f'Time Step {time_step} of {max_timesteps}: Epoch {epoch} of {epochs_per_timestep} ({trainer.state.epoch}) ' \
          f'{evaluation}:'
    print_metric_msg(fn, engine, msg, metrics)


def write_losses(engine, metrics, writer, phase, fn=lambda x: x):
    """
    This writes the VAE losses
    """
    for name in metrics:
        value = fn(engine.state.metrics[name])  # do we need to divide by engine.state.epoch
        writer.add_scalar(f'{phase}_Losses/{name}', value, engine.state.epoch)


@torch.no_grad()
def evaluate_model_generalized(engine, model, prob_model, dataset, phase_num, policy, costs, loss, state, trainer_state,
                               metric_dct, epochs_per_step, cf_dataset=None, util_gnd=False):
    """
    Evaluation and collection of all metrics
    @param engine: Ignite engine object
    @param model: VAE model
    @param prob_model: probabilistic model
    @param dataset: dataset object
    @param phase_num: phase '1' or '2'
    @param policy: policy used in phase 2
    @param costs: decision cost c
    @param loss: which loss function is used (ours, niki, etc.)
    @param state: state to return (final) or not
    @param trainer_state: ignite trainer state
    @param metric_dct: dictionary storing metrics
    @param epochs_per_step: num epochs per time-step in phase 2
    @param cf_dataset: counterfactual dataset (for eval)
    @param util_gnd: utility w.r.t. ground truth (hidden)
    @return: metric dictionary if 'final' state
    """
    # Collect everything from engine.state.metrics to our metric_dct
    epoch = 1 if trainer_state is None else trainer_state.state.epoch
    if '2' in phase_num:
        epoch = int(epoch / epochs_per_step)
    for k in engine.state.metrics.keys():
        try:
            metric_dct[k].update({epoch: engine.state.metrics[k]})
        except:
            metric_dct.update({k: {epoch: engine.state.metrics[k]}})
    prob_model.eval()
    model.eval()

    is_discrete = lambda d: 'categorical' in str(d) or 'bernoulli' in str(d)  # or 'poisson' in str(d)

    use_data = dataset.data
    use_mask = dataset.mask

    if cf_dataset is not None:
        cf_use_data = cf_dataset.data
        cf_use_mask = cf_dataset.mask

    # NOTE: Policy Q(U|X,S) and P(U|Z,S) are already logged.
    if '2' in phase_num:
        # Our assumption that the last column is S and second last U
        u_true = use_data[:, -2]
        yf_true = dataset.yf
        if yf_true is not None:
            yf_true = yf_true.flatten()

    # Evaluate on fully supervised dataset. Important assumption, otherwise we need to IPS the scores!
    mask_ones = torch.ones_like(use_data[:, :-1])  # NxD
    prob = None  # fully observed dataset

    if loss in [Cte.LOSS_FAIRALL, Cte.LOSS_FAIRLAB]:
        # DECODER NORMAL PERFORMANCE
        # 1. Get the reconstructed data. Make one pass and evaluate
        generated_data, z, dec_prob1 = model.reconstruct(use_data, mask_ones)  # NxD
        # 2. Compute error using Adrian's functions with prob_model
        observed_error = imputation_error(prob_model, generated_data, use_data, mask_ones)
        # 3. Loop and write the errors to writer
        if hasattr(engine.state, 'registers'):  # and state.epoch % self.print_every == 1:
            with torch.no_grad():
                for idx, (err, dist) in enumerate(zip(observed_error, prob_model)):
                    # Error of reconstructing utility U
                    if 'SSCVAE' in str(model) and idx == len(prob_model) - 1:
                        # If model is SSCVAE and the last dim, then this is the U.
                        engine.state.registers.update({'Rec_ERR_U': err.item()})
                        try:
                            metric_dct['Rec_ERR_U'].update({epoch: err.item()})
                        except:
                            metric_dct.update({'Rec_ERR_U': {epoch: err.item()}})
                    else:
                        # Reconstruction error for continuous X
                        if not is_discrete(dist):
                            engine.state.registers.update({f'Rec_NRMSE_X{idx}': err.item()})
                            try:
                                metric_dct[f'Rec_NRMSE_X{idx}'].update({epoch: err.item()})
                            except:
                                metric_dct.update({f'Rec_NRMSE_X{idx}': {epoch: err.item()}})
                        # Reconstruction error for other X.
                        else:
                            engine.state.registers.update({f'Rec_ERR_X{idx}': err.item()})
                            try:
                                metric_dct[f'Rec_ERR_X{idx}'].update({epoch: err.item()})
                            except:
                                metric_dct.update({f'Rec_ERR_X{idx}': {epoch: err.item()}})

        # DECODER FOR DECISION MAKING
        if '2' in phase_num:
            if "Policy U ~ P(z,S)" in str(policy):
                ret_mask, prob_pol = policy(model, use_data, use_mask, is_sup=False)
                dec_pred = ret_mask[:, 0].float()

                dec_utility = compute_utility(u_true, dec_pred.flatten(), mask_ones, costs, prob)
                dec_acc = compute_accuracy(u_true, dec_pred.flatten(), mask_ones, prob)
                if util_gnd:
                    pol_utility_gnd = compute_utility(yf_true, dec_pred.flatten(), mask_ones, costs, prob)
                    pol_acc_gnd = compute_accuracy(yf_true, dec_pred.flatten(), mask_ones, prob)
                pol_acc = dec_acc
                pol_utility = dec_utility

                pol_preds = dec_pred
                if cf_dataset is not None:
                    cf_ret_mask, cf_prob_pol = policy(model, cf_use_data, cf_use_mask, is_sup=False)
                    cf_pol_preds = cf_ret_mask[:, 0].float()
            else:
                # TODO: Maybe should remove this dummy part later. Will not use in analysis later!
                dec_pred = generated_data[:, -1]
                dec_utility = compute_utility(u_true, dec_pred.flatten(), mask_ones, costs, prob)
                dec_acc = compute_accuracy(u_true, dec_pred.flatten(), mask_ones, prob)

            if hasattr(engine.state, 'registers'):
                with torch.no_grad():
                    engine.state.registers.update({'Dec_ERR_U': 1 - dec_acc.item()})
                    engine.state.registers.update({'Dec_utility': dec_utility.item()})
                    # Prediction error and utility for decoder policy.
                    try:
                        metric_dct['Dec_ERR_U'].update({epoch: 1 - dec_acc.item()})
                    except:
                        metric_dct.update({'Dec_ERR_U': {epoch: 1 - dec_acc.item()}})
                    try:
                        metric_dct['Dec_utility'].update({epoch: dec_utility.item()})
                    except:
                        metric_dct.update({'Dec_utility': {epoch: dec_utility.item()}})

    # Classifier
    # 4. Classifier Accuracy: only for OUR model (FairAll) and NIKIs (Fair/UnfairLog)
    if 'SSCVAE' in str(model) and ('2' in phase_num):
        # Feed XU, S.
        if loss in [Cte.LOSS_FAIRALL, Cte.LOSS_FAIRLOG]:
            prob_pred = sigmoid(model.classify(use_data[:, :-1], use_data[:, -1], mask_ones))
            # accuracy
            if "Policy U ~ Q(X,S)" in str(policy):
                ret_mask, prob_pol = policy(model, use_data, use_mask)
                prob_pred = prob_pol
                pred_u = ret_mask[:, 0].float()
            else:
                pred_u = prob_pred.clone()
                pred_u[pred_u >= costs] = 1
                pred_u[pred_u < costs] = 0

            clf_acc = compute_accuracy(u_true, pred_u.flatten(), mask_ones, prob)
            clf_utility = compute_utility(u_true, pred_u.flatten(), mask_ones, costs, prob)

            # Classifier for decision making
            if "Policy U ~ Q(X,S)" in str(policy):
                pol_acc = clf_acc
                pol_utility = clf_utility
                pol_preds = pred_u
                if util_gnd:
                    pol_acc_gnd = compute_accuracy(yf_true, pred_u.flatten(), mask_ones, prob)
                    pol_utility_gnd = compute_utility(yf_true, pred_u.flatten(), mask_ones, costs, prob)
                if cf_dataset is not None:
                    cf_ret_mask, cf_prob_pol = policy(model, cf_use_data, cf_use_mask)
                    cf_pol_preds = cf_ret_mask[:, 0].float()

            if hasattr(engine.state, 'registers'):
                # Prediction error and utility for classifier policy.
                with torch.no_grad():
                    engine.state.registers.update({'Clf_ERR_U': 1 - clf_acc.item()})
                    try:
                        metric_dct['Clf_ERR_U'].update({epoch: 1 - clf_acc.item()})
                    except:
                        metric_dct.update({'Clf_ERR_U': {epoch: 1 - clf_acc.item()}})
                    engine.state.registers.update({'Clf_utility': clf_utility.item()})
                    try:
                        metric_dct['Clf_utility'].update({epoch: clf_utility.item()})
                    except:
                        metric_dct.update({'Clf_utility': {epoch: clf_utility.item()}})

    # FZ: Latent Z for decision making.
    # 5. Also report policy F(Z) if phase 2.
    # NOTE: Policy Q(U|X,S) and P(U|Z,S) are already logged.
    if ('2' in phase_num) and ('U ~ F(z)' in str(policy)) and (loss in [Cte.LOSS_FAIRALL, Cte.LOSS_FAIRLAB]):
        return_mask, prob_pol = policy(model, use_data, use_mask)
        true_lab = use_data[:, -2]
        pred_lab = return_mask[:, 0].float()

        pol_acc = compute_accuracy(true_lab, pred_lab.flatten(), mask_ones, prob)
        pol_utility = compute_utility(u_true, pred_lab.flatten(), mask_ones, costs, prob)
        if util_gnd:
            pol_acc_gnd = compute_accuracy(yf_true, pred_lab.flatten(), mask_ones, prob)
            pol_utility_gnd = compute_utility(yf_true, pred_lab.flatten(), mask_ones, costs, prob)
        pol_preds = pred_lab
        if cf_dataset is not None:
            cf_ret_mask, cf_prob_pol = policy(model, cf_use_data, cf_use_mask)
            cf_pol_preds = cf_ret_mask[:, 0].float()

        if hasattr(engine.state, 'registers'):
            # Prediction error and utility for FZ (latent Z) policy.
            with torch.no_grad():
                engine.state.registers.update({'Policy_FZ_ERR': 1 - pol_acc.item()})
                try:
                    metric_dct['Policy_FZ_ERR'].update({epoch: 1 - pol_acc.item()})
                except:
                    metric_dct.update({'Policy_FZ_ERR': {epoch: 1 - pol_acc.item()}})
                engine.state.registers.update({'Policy_FZ_utility': pol_utility.item()})
                try:
                    metric_dct['Policy_FZ_utility'].update({epoch: pol_utility.item()})
                except:
                    metric_dct.update({'Policy_FZ_utility': {epoch: pol_utility.item()}})

    if '2' in phase_num:
        if hasattr(engine.state, 'registers'):
            # This basically will contain the main policy being used! We use this for analysis.
            # All above logging is for debugging and internal studies.
            with torch.no_grad():
                engine.state.registers.update({'Policy_ERR': 1 - pol_acc.item()})
                try:
                    metric_dct['Policy_ERR'].update({epoch: 1 - pol_acc.item()})
                except:
                    metric_dct.update({'Policy_ERR': {epoch: 1 - pol_acc.item()}})
                engine.state.registers.update({'Policy_utility': pol_utility.item()})
                try:
                    metric_dct['Policy_utility'].update({epoch: pol_utility.item()})
                except:
                    metric_dct.update({'Policy_utility': {epoch: pol_utility.item()}})

                # If synthetic data, we also store utility and error w.r.t. ground truth (hidden)
                if util_gnd:
                    engine.state.registers.update({'Policy_Gnd_ERR': 1 - pol_acc_gnd.item()})
                    try:
                        metric_dct['Policy_Gnd_ERR'].update({epoch: 1 - pol_acc_gnd.item()})
                    except:
                        metric_dct.update({'Policy_Gnd_ERR': {epoch: 1 - pol_acc_gnd.item()}})
                    engine.state.registers.update({'Policy_Gnd_utility': pol_utility_gnd.item()})
                    try:
                        metric_dct['Policy_Gnd_utility'].update({epoch: pol_utility_gnd.item()})
                    except:
                        metric_dct.update({'Policy_Gnd_utility': {epoch: pol_utility_gnd.item()}})

        pred_labels = pol_preds
        pred_labels_s1, pred_labels_s0 = pred_labels[use_data[:, -1] == 1], pred_labels[use_data[:, -1] == -1]

        # Compute and store demographic parity unfairness.
        discrimination_abs = torch.abs(torch.mean(pred_labels_s1) - torch.mean(pred_labels_s0)).item()

        try:
            metric_dct['DP_unfairness'].update({epoch: discrimination_abs})
        except:
            metric_dct.update({'DP_unfairness': {epoch: discrimination_abs}})

        # If counterfactual data is present, also compute counterfactual unfairness.
        if cf_dataset is not None:
            cf_pred_labels = cf_pol_preds
            cf_unfairness = torch.mean(torch.abs(pred_labels - cf_pred_labels)).item()
            try:
                metric_dct['CF_unfairness'].update({epoch: cf_unfairness})
            except:
                metric_dct.update({'CF_unfairness': {epoch: cf_unfairness}})

    # Store effective utility (accumulated utility) from training process
    if '2' in phase_num and not (dataset.val or dataset.test):
        pred_labels = pol_preds
        temp_utility = compute_utility(u_true, pred_labels.flatten(), mask_ones, costs, prob).item()
        if util_gnd:
            temp_utility_gnd = compute_utility(yf_true, pred_labels.flatten(), mask_ones, costs, prob).item()
        if 'Effective_util' not in metric_dct:
            effective_utility = temp_utility
        else:
            effective_utility = (metric_dct['Effective_util'][epoch - 1] * (epoch - 1) + temp_utility) / epoch
        try:
            metric_dct['Effective_util'].update({epoch: effective_utility})
        except:
            metric_dct.update({'Effective_util': {epoch: effective_utility}})

        if util_gnd:
            if 'Effective_util_gnd' not in metric_dct:
                effective_utility_gnd = temp_utility_gnd
            else:
                # Epochs are 1-based so easier!
                effective_utility_gnd = (metric_dct['Effective_util_gnd'][epoch - 1] * (
                        epoch - 1) + temp_utility_gnd) / epoch
            try:
                metric_dct['Effective_util_gnd'].update({epoch: effective_utility_gnd})
            except:
                metric_dct.update({'Effective_util_gnd': {epoch: effective_utility_gnd}})

        # Logging Effective discrimination as well (accumulated unfairness measure across training)
        pred_labels_s1, pred_labels_s0 = pred_labels[use_data[:, -1] == 1], pred_labels[use_data[:, -1] == -1]
        temp_discrimination = torch.abs(torch.mean(pred_labels_s1) - torch.mean(pred_labels_s0)).item()
        if 'Effective_DPU' not in metric_dct:
            effective_discrimination = temp_discrimination
        else:
            effective_discrimination = (metric_dct['Effective_DPU'][epoch - 1] * (epoch - 1) +
                                        temp_discrimination) / epoch
        try:
            metric_dct['Effective_DPU'].update({epoch: effective_discrimination})
        except:
            metric_dct.update({'Effective_DPU': {epoch: effective_discrimination}})

    if state == 'final':
        return metric_dct


def nrmse(pred, target, mask):  # for numerical variables
    norm_term = torch.max(target) - torch.min(target)
    new_pred = torch.masked_select(pred, mask.bool())
    new_target = torch.masked_select(target, mask.bool())

    return torch.sqrt(torch.nn.functional.mse_loss(new_pred, new_target)) / norm_term


def error_rate(pred, target, mask):  # for categorical variables - this is 1-accuracy!
    return torch.sum((pred != target).double() * mask) / mask.sum()


def displacement(pred, target, mask, size):  # for ordinal variables
    diff = (target - pred).abs() * mask / size
    return diff.sum() / mask.sum()


def imputation_error(prob_model, pred, target, mask):
    mask = mask.double()

    pos = 0
    errors = []
    for i, dist in enumerate(prob_model):
        if 'bernoulli' in str(dist):  # nominal
            errors.append(error_rate(pred[:, pos], target[:, pos], mask[:, pos]))
        elif 'categorical' in str(dist):  # nominal
            if dist.domain_size > 1:
                pred_i = torch.argmax(pred[:, pos: pos + dist.domain_size], dim=-1)
                target_i = torch.argmax(target[:, pos: pos + dist.domain_size], dim=-1)
            else:
                pred_i, target_i = pred[:, pos], target[:, pos]
            errors.append(error_rate(pred_i, target_i, mask[:, pos]))
        else:  # numerical
            errors.append(nrmse(pred[:, pos], target[:, pos], mask[:, pos]))
        pos += dist.domain_size

    return errors


@torch.no_grad()
def calculate_ll(engine, model, prob_model, dataset, writer):
    prob_model.eval()

    epoch = engine.state.epoch
    mean = lambda x: sum(x).item() / len(x)

    # observed_mask = getattr(dataset, 'mask_original', torch.ones_like(dataset.data))[:, :-1] #take off s
    observed_mask = getattr(dataset, 'mask_original', torch.ones_like(dataset.data))  # take off s

    if str(model)[-3:] == 'VAE':  # I can't import models due to circular dependencies
        observed_log_prob = model.log_likelihood(dataset.data, None, getattr(dataset, 'mask'))
    else:
        observed_log_prob = model.log_likelihood(dataset.data, dataset.params, None)

    observed_log_prob = (observed_log_prob * observed_mask).mean(dim=0)

    is_discrete = lambda d: 'categorical' in str(d) or 'bernoulli' in str(d)
    nominal_error = [e for e, d in zip(observed_log_prob, prob_model) if is_discrete(d)]
    nominal_error = mean(nominal_error) if len(nominal_error) > 0 else 0.
    numerical_error = [e for e, d in zip(observed_log_prob, prob_model) if not is_discrete(d)]
    numerical_error = mean(numerical_error) if len(numerical_error) > 0 else 0.

    writer.add_scalar('Loglikelihood/nominal', nominal_error, engine.state.epoch)
    writer.add_scalar('Loglikelihood/numerical', numerical_error, engine.state.epoch)
    writer.add_scalar('Loglikelihood/total', mean(observed_log_prob), engine.state.epoch)

    if epoch % (engine.state.max_epochs / 4) == 0:
        print(f'[{int(epoch / engine.state.max_epochs * 100.)}%] observed log-likelihood:')
        for i, error in enumerate(observed_log_prob):
            print(f'[dim={i}] {error}')
        print('nominal  :', nominal_error)
        print('numerical:', numerical_error)
        print('total    :', mean(observed_log_prob))
        print('')
