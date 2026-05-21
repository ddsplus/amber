Dpath = '../../Dataset'
datasets = {
    'assist2009' : 'assist2009',
    'assist2012' : 'assist2012',
    'assist2017' : 'assist2017',
    'assistednet': 'assistednet',
}


# question number of each dataset
numbers = {
    'assist2009' : 16891,
    'assist2012' : 37125,
    'assist2017' : 3162,
    'assistednet': 10795,
}

skill = {
    'assist2009' : 101,
    'assist2012' : 188,
    'assist2017' : 102,
    'assistednet': 1676,
}


DATASET = datasets['assist2017']
NUM_OF_QUESTIONS = numbers['assist2017']
H = '2017'

MAX_STEP = 50
max_step = 50
BATCH_SIZE = 512
LR = 0.005
teacher_learning_rate = 1e-6
EPOCH = 20
EMB = 256
HIDDEN = 128
loss_weight = 5.00E-04
kd_loss = 5.00E-06
LAYERS = 1

num_meta_batches = 1
gradient_accumulation_steps = 1

assume_s_step_size = 1e-3

max_grad_norm = 1

step2_frequency = 1
