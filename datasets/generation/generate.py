import argparse
import os
import sys
from pathlib import Path
import yaml
import csv
# import h5py
import numpy as np
import pandas as pd
# import yaml
from util import get_utility

root_path = os.path.abspath(os.path.join(".."))
if root_path not in sys.path:
    sys.path.append(root_path)

from distributions_syn import NoConfounders, NonSensitiveConfounder, SensitiveConfounderBias, \
SensitiveConfounderVar, SensitiveConfounderVarBias
# from distributions_real import COMPASDistribution, AdultCreditDistribution, GermanCreditDistribution, \
#     UncalibratedScore, FICODistribution
from initial_policies import OptimalPolicy, LogPolicy, BiasedPolicy, SigmoidPolicy, RandomPolicy

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-dis', '--distribution', type=str, required=True,
                        help='distribution choice:' 
                             '[NC] No Confounders, [NSC] Non-Sensitive Confounder, '
                             '[SCB] Non-Sensitive and Sensitive Confounder Bias,'
                             '[SCV] Non-Sensitive and Sensitive Confounder Variance, '
                             '[SC2] Non-Sensitive and Sensitive Confounder Bias and Variance,')
                             # '[COMPAS] Compas dataset (x_dim = 5),'
                             # '[GECR] German Credit dataset (x_dim = 41),'
                             # '[ADCR] Adult Credit dataset (x_dim = 71)')

    parser.add_argument('-pol', '--policy', required=True, nargs='+',
                        help='policy choice: '
                             '[LENI] Lenient based on one feature, '
                             '[HARSH] Harsh based on one feature, '
                             '[BPLENI] Biased Lenient based on one feature and S, '
                             '[BPHARSH] Biased Harsh based on one feature and S, '
                             '[OPT] Stochastically Optimal, '
                             '[BP] Biased stochastically optimal, '
                             '[FUL] Full Logistic Regression using X and S as input, '
                             '[UNA] Unaware Logistic Regression using X but not S as input'
                             '[RAN] Randomly labelled,'
                             '[LAB] Fully labelled')

    parser.add_argument('-n1', '--train_n1', type=int, required=True,
                        help='total amount of training data points to be created in phase 1')
    parser.add_argument('-n2', '--train_n2', type=int, required=True,
                        help='total amount of training data points to be created in phase 2')

    parser.add_argument('-nt', '--test_n', type=int, required=False,default=0,
                        help="total amount of testing data points to be created (optional: default=0, no training "
                             "set is created), same for phase 1 and phase 2")


    parser.add_argument('-p', '--path', type=str, required=True,
                        help='path, where to save data generated')

    parser.add_argument('-op', '--optimal_prob', type=float, required=False, default=0.7,
                        help='for OPT policy: probability of taking a correct decision, default=0.7')

    parser.add_argument('-nlog', '--train_log_n', type=int, required=False, default=100,
                        help='for FUL and UNA policies: amount of training data to be generated and '
                             'trained on, default=500')

    parser.add_argument('-fp', '--fraction_protected', type=float, required=False, default=0.5,
                        help='fraction protected: proportion of people with attribute s=1, default=0.3')

    parser.add_argument('-seed', '--seed', type=int, required=False, default= 123,
                        help='random seed, default=123')

    parser.add_argument('-tp', '--test_percentage', type=float, required=False, default=0.1,
                        help='percentage of test dataset for train-test splitting of data, default=0.1')

    args = parser.parse_args()



    def policy_dataset(pol_name, data, seed, dist=None, dist_name=None, nlog=None, opt_prob=None):
        # Todo: input dist
        if pol_name == "OPT":
            pol = OptimalPolicy(opt_prob, seed)
        elif pol_name == 'BP':
            pol = BiasedPolicy(opt_prob, seed)
        elif pol_name == 'FUL':
            pol = LogPolicy('Full', seed)
        elif pol_name == 'UNA':
            pol = LogPolicy('Unaware', seed)
        elif pol_name =='LENI':
            pol = SigmoidPolicy('lenient', dist_name, seed, biased=False)
        elif pol_name == 'HARSH':
            pol = SigmoidPolicy('harsh', dist_name, seed, biased=False)
        elif pol_name == 'BPLENI':
            pol = SigmoidPolicy('lenient', dist_name, seed, biased=True)
        elif pol_name == 'BPHARSH':
            pol = SigmoidPolicy('harsh', dist_name, seed, biased=True)
        elif pol_name == 'RAN':
            pol = RandomPolicy(seed)
        else:
            print("I dont know policy ", pol)
            raise NotImplementedError


        if type(pol) is LogPolicy:
            # generate training set for policy
            dt = dist.sample_train_dataset(nlog, seed*24)
            pol.fit(dt)
        print(f"data {data}")
        prob, util, fpr = pol.predict(data)
        se_prob = pd.Series(prob, name=pol_name)

        return se_prob, np.round(util,4), np.round(fpr,4)


    for phase in [1,2]:
        print("phase", phase)
        ntrain1 = args.train_n1
        ntrain2 = args.train_n2
        ntest = args.test_n
        dist_name = args.distribution

        if args.policy[0] in ['all']:
            policy_list = ['LENI', 'HARSH', 'BP']
        else:
            policy_list = args.policy # list

        opt_prob = args.optimal_prob
        base_save_path = "{}_{}".format(args.path, args.seed)
        nlog = args.train_log_n
        fraction_protect = args.fraction_protected

        test_percentage = args.test_percentage
        seed = args.seed*phase
        seed2 = seed*5*phase

        opt_prob_str = str(opt_prob).replace("0.", "")
        # policy_str = args.policy.replace(" ", "")


        # Todo: Potentially to with try except
        if dist_name == "NC":
            dist = NoConfounders(fraction_protect)
        elif dist_name == "NSC":
            dist = NonSensitiveConfounder(fraction_protect)
        elif dist_name == "SCB":
            dist = SensitiveConfounderBias(fraction_protect)
        elif dist_name == "SCV":
            dist = SensitiveConfounderVar(fraction_protect)
        elif dist_name == "SC2":
            dist = SensitiveConfounderVarBias(fraction_protect)
        elif dist_name == "UNCL":
            dist = UncalibratedScore(fraction_protect)
        elif dist_name == "1D":
            dist = DummyDistribution1D(fraction_protect)
        else:
            print('ERROR: DIST', dist_name)
            raise NotImplementedError


        print('dist_name', dist_name)
        dataset_type = ['train', 'valid']
        # train_samples[ntrain1, ntrain2]
        # samples = [ntrain, ntest]
        for i in range(2):
            if i == 1:
                samples = ntest
            else:
                if phase == 1:
                    samples = ntrain1
                else:
                    samples = ntrain2

            dataset_training = dist.sample_train_dataset(samples, phase*(i+2)*seed)
            # print('Vfae Trainingset', d)
            df = dataset_training


            train_dict = {}

            print('/// --- Statistics of {} dataset for phase {} --- ///'.format(dataset_type[i], phase))
            y = dataset_training["Y"]
            y_fair = dataset_training["Y_F"]
            rand = pd.Series(0.5*np.ones(len(dataset_training["Y"])), index=dataset_training["Y"].index)
            _, util_opt_fair = get_utility(y_fair, y)
            _, util_opt_unfair = get_utility(y, y)
            _, util_random = get_utility(rand, y)
            _, util_opt_0 = get_utility(np.zeros(len(y)), y)
            _, util_opt_1 = get_utility(np.ones(len(y)), y)

            print('// Optimal Fair utility (OFU)', util_opt_fair)
            print('// Optimal Unfair utility (OUU)', util_opt_unfair)
            print('// Random utility (RAU)', util_random)
            print('// Trivial decide 1 always (DA1)', util_opt_1)
            print('// Trivial decide 0 always (DA0)', util_opt_0)

            train_dict['samples'] = samples
            train_dict['opt_prob'] = opt_prob
            train_dict['nlog'] = nlog
            train_dict['fraction_protect'] = fraction_protect

            train_dict['OUU'] = util_opt_unfair
            train_dict['OFU'] = util_opt_fair
            train_dict['RAU'] = util_random
            train_dict['DA0'] = util_opt_0
            train_dict['DA1'] = util_opt_1

            if dist_name == "SCB":
                prob_FYA = dataset_training['prob_FYA']
                _, util_prob_FYA_F = get_utility(prob_FYA, y_fair)
                print('/// Optimal Fair utility probabilistic (OUPROB_F)', util_prob_FYA_F)
                train_dict['OUPROB_F'] = util_prob_FYA_F
                _, util_prob_FYA = get_utility(prob_FYA, y)
                print('/// Optimal Unfair utility probabilistic (OUPROB)', util_prob_FYA)
                train_dict['OUPROB'] = util_prob_FYA

            info_all = ""
            info = 'no-info'
            if phase == 2:
                for policy in policy_list:
                    print("///////////////////////", policy)
                    if policy in ['LENI', 'HARSH', 'BPLENI', 'BPHARSH', 'OPT', 'BP', 'UNA', 'FUL', 'RAN']:
                        # info needed for printing and saving
                        if policy in ['OPT', 'BP']:
                            info = policy + str(int(100 * opt_prob))
                            pol_prob, util, fpr = policy_dataset(policy, dataset_training, phase*i*seed, dist_name, opt_prob=opt_prob)

                        elif policy in ['UNA', 'FUL', 'RAN']:
                            info = policy + str(nlog)
                            print(f"phase {phase}")
                            print(f"data {dataset_training}")
                            pol_prob, util, fpr = policy_dataset(policy, dataset_training, phase*i*seed, dist=dist, nlog=nlog)

                        elif policy in ['LENI', 'HARSH', 'BPLENI', 'BPHARSH']:
                            info = policy
                            pol_prob, util, fpr = policy_dataset(policy, dataset_training, phase*i*seed, dist_name=dist_name)

                        else:
                            raise NotImplementedError # labeling with

                    else:
                        print('ERROR: POL name {} not recognized'.format(policy))

                    print('// policy', info, 'av. utility', util, 'fpr', fpr)
                    train_dict[info] = util
                    train_dict[info+'_fpr'] = fpr

                    info_all = info_all + info + "_"

                    df = pd.concat([df, pol_prob], axis=1)
            print('/// ------------------------- ///')
            print('columns', df)
            print('train labelled', df.columns)


            #  if name does not exist, create file, otherwise append (mode='a')
            # this can be helpfulf, if we want to create a few different distributions or policies
            # data_path = "{}/{}".format(base_save_path, policy)
            # Path(data_path).mkdir(parents=True, exist_ok=True)

            data_path = base_save_path
            Path(data_path).mkdir(parents=True, exist_ok=True)

            fp = int(fraction_protect * 100)

            if dataset_type[i] == 'train':
                # path_train = "{}/data{}.csv".format(base_save_path, phase)
                path_train_all = "{}/data{}.csv".format(base_save_path, phase)
                # path_train_all = "{}/data{}_all.csv".format(base_save_path, phase)
                path_train_types = "{}/data{}_types.csv".format(base_save_path, phase)
                txfile_name = "{}/data{}_util.txt".format(base_save_path, phase)
            else:
                # path_train = "{}/{}{}.csv".format(base_save_path,dataset_type[i], phase)
                path_train_all = "{}/{}{}.csv".format(base_save_path, dataset_type[i], phase)
                # path_train_all = "{}/{}{}_all.csv".format(base_save_path, dataset_type[i], phase)
                path_train_types = "{}/{}{}_types.csv".format(base_save_path, dataset_type[i], phase)
                txfile_name = "{}/{}{}_util.txt".format(base_save_path, dataset_type[i], phase)

            # save utils to txt file
            with open(txfile_name, 'w') as f:
                print(train_dict, file=f)

            # save data to csv and types to csv
            if phase == 1:
                # df[["GPA", "LSAT", "S"]].to_csv(path_train, index=False)
                df.to_csv(path_train_all, index=False)
                types_dict = {'type': ['real', 'real', 'cat'], 'dim': [1, 1, 2], 'nclass': [pd.NA,pd.NA,2]}
            else:
                # df[["GPA", "LSAT", "Y", "S", *policy_list]].to_csv(path_train, index=False)
                df.to_csv(path_train_all, index=False)
                types_dict = {'type': ['real', 'real', 'cat', 'cat'], 'dim': [1, 1, 2, 2], 'nclass': [pd.NA,pd.NA,2,2]}
            #save datafiles to pd
            # types_dict = {'col1': [1, 2], 'col2': [3, 4]}
            df_types = pd.DataFrame(data=types_dict)
            df_types.to_csv(path_train_types, index=False)

            # file = h5py.File(path_train, 'w')
            # df.to_hdf(path_train, dist_name, mode='a')
            print("SAVED training to ", path_train_all)

            data_hypams_dict = [{'data': args.distribution,
                                 'policies': args.policy,
                                 'training_size_phase1': args.train_n1,
                                 'training_size_phase2': args.train_n2,
                                 'testing_size': args.test_n,
                                 'fraction_protected': args.fraction_protected,
                                 'seed': args.seed,
                                 'muG':dist.muG, 'sigG':dist.sigG,
                                 'muL': dist.muL, 'sigL': dist.sigL,
                                 'muF': dist.muF, 'sigF': dist.sigF, 'threshold':dist.threshold,
                                 'wSF': dist.wSF, 'wSG': dist.wSG, 'wSL': dist.wSL,
                                 'muK0': dist.muK0,'muK1': dist.muK1, 'sigK': dist.sigK,
                                 'wKG': dist.wKG, 'wKL': dist.wKL, 'wKF': dist.wKF
                                 }]

            #todo: check that this works
            with open('{}/data_hyperparams.yaml'.format(base_save_path), 'w') as file:
                documents = yaml.dump(data_hypams_dict, file)

        if ntest > 0:
            test_dict = {}
            # print('Test set for Vfae')
            # Todo: y_fair
            df_test = dist.sample_test_dataset(ntest, 7*seed*phase)
            # d = {'S': s, 'GPA': xG, 'LSAT': xL, 'Y': y, 'Y_F': y_fair, 'prob_FYA':prob_FYA, 'K': k, 'FYA': fya}
            # d_stats = dist.sample_stats_dataset(ntest, 17 * seed)  # S, GPA, LSAT, Y, Y_F, K, FYA
            y_fair = df_test['Y_F']
            y = df_test["Y"]
            rand = pd.Series(0.5*np.ones(len(df_test["Y"])), index=df_test["Y"].index)

            # print('d', d)

            _, util_opt_fair = get_utility(y_fair, y)
            _, util_opt_unfair = get_utility(y, y)
            _, util_random = get_utility(rand, y)
            _, util_opt_0 = get_utility(np.zeros(len(y)), y)
            _, util_opt_1 = get_utility(np.ones(len(y)), y)


            print(f'/// --- Statistics of TEST dataset for phase {phase}--- ///')
            print('/// Optimal Fair utility (OFU)', util_opt_fair)
            print('/// Optimal Unfair utility (OUU)', util_opt_unfair)
            print('/// Random utility (RAU)', util_random)
            print('/// Trivial decide 1 always (D1A)', util_opt_1)
            print('/// Trivial decide 0 always (D0A)', util_opt_0)
            print('/// ------------------------- ///')


            test_dict['OUU'] = util_opt_unfair
            test_dict['OFU'] = util_opt_fair
            train_dict['RAU'] = util_random
            test_dict['DA0'] = util_opt_0
            test_dict['DA1'] = util_opt_1

            if dist_name == "SCB":
                prob_FYA = df_test['prob_FYA']
                _, util_prob_FYA_F = get_utility(prob_FYA, y_fair)
                print('/// Optimal Fair utility probabilistic (OUPROB_F)', util_prob_FYA_F)
                test_dict['OUPROB_F'] = util_prob_FYA_F
                _, util_prob_FYA = get_utility(prob_FYA, y)
                print('/// Optimal Unfair utility probabilistic (OUPROB)', util_prob_FYA)
                test_dict['OUPROB'] = util_prob_FYA

            key_name = dist_name
            fp = int(fraction_protect*100)

            data_path = base_save_path
            Path(data_path).mkdir(parents=True, exist_ok=True)

            fp = int(fraction_protect * 100)
            file_name = f"{dist_name}_{seed}"

            # path_test = f"{base_save_path}/test{phase}.csv"
            path_test_all = f"{base_save_path}/test{phase}.csv"
            # file = h5py.File(path_test, 'w')

            if not os.path.exists(data_path):
                os.makedirs(data_path)


            if phase == 1:
                # df_test[["GPA", "LSAT", "S"]].to_csv(path_test, index=False)
                df_test.to_csv(path_test_all, index=False)
            else:
                # df_test[["GPA", "LSAT", "Y", "S"]].to_csv(path_test, index=False)
                df_test.to_csv(path_test_all, index=False)

            # file_name_stats = "{}_{}".format(file_name, "stats.h5")
            # path_stats = "{}/{}".format(data_path, file_name_stats)
            # file = h5py.File(path_stats, 'w')
            # d_stats.to_hdf(path_stats, key_name, mode='a')

            # readit = pd.read_hdf(path_test, key_name)
            # print('readit', readit)
            # 1/0

            txfile_name = f"{base_save_path}/test{phase}_util.txt"
            print('txfile_name',txfile_name)

            with open(txfile_name, 'w') as f:
                print(test_dict, file=f)







