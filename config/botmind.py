out_dir = 'out_botmind'
dataset = 'botmind' 
device = 'mps' #since i am currently running on a m2
dtype = 'float32'
compile = False

n_layer = 4
n_head = 4
n_embd = 256
block_size = 256
dropout = 0.1

batch_size = 64
gradient_accumulation_steps = 1
learning_rate = 1e-3
max_iters = 4000
lr_decay_iters = 4000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

eval_interval = 250
eval_iters = 50
log_interval = 10
always_save_checkpoint = False
wandb_log = False