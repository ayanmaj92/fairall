"""
Authors: Miriam Rateike, Ayan Majumdar
"""


class Cte:
    """
    Common constants used in the project
    """
    # initial policies
    HARSH = 'HARSH'
    LENI = 'LENI'
    HARSHC = 'HARSHC'
    LENIC = 'LENIC'
    RAN = 'RAN'

    # Datasets (add new dataset below)
    SCB_11 = 'SCB_11'
    compas = 'compas'
    credit = 'credit'
    meps = 'meps'

    # Models
    OURS = 'sscvae'
    CVAE = 'cvae'
    VAE = 'vae'
    IPSCVAE = 'ipscvae'
    NIKICLF = 'nikiclf'

    # Loss function
    LOSS_FAIRALL = 'loss_fairall'
    LOSS_FAIRLAB = 'loss_fairlab'
    LOSS_FAIRLOG = 'loss_fairlog'

    # Niki model
    MODEL_LR = 'lr'
    MODEL_NN = 'nn'

    # learn policies
    POL_QXS = 'QXS'
    POL_PZS = 'PZS'
    POL_FZ = 'FZ'
    POL_NA = 'NA'

    # FZ options
    FZ_LAB = 'LAB'
    FZ_DEC = 'DEC'
    FZ_CLF = 'CLF'

    # LAYERES

    # Optimizers
    ADAM = 'adam'

    # Scheduler
    STEP_LR = 'step_lr'
    EXP_LR = 'exp_lr'

    # Initializer
    XAVIER = 'xav'

    # Activation
    TANH = 'tanh'
    RELU = 'relu'
    RELU6 = 'relu6'
    SOFTPLUS = 'softplus'
    RRELU = 'rrelu'
    LRELU = 'lrelu'
    ELU = 'elu'
    SELU = 'selu'
    SIGMOID = 'sigmoid'
    GLU = 'glu'
    IDENTITY = 'identity'

    # DATA SPLIT
    TEST = 'test'
    VALID = 'valid'
    TRAIN = 'train'

    # Distribution
    BETA = 'beta'
    CONTINOUS_BERN = 'cb'
    BERNOULLI = 'ber'

    GAUSSIAN = 'normal'
    CATEGORICAL = 'cat'
    EXPONENTIAL = 'exp'
    DELTA = 'delta'
