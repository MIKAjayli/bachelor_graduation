"""Single-episode diagnostic evaluation for SG-DP3."""
import sys
import os

sys.path.append('./')
sys.path.append('./policy')
sys.path.append('./description/utils')
import importlib
import yaml
import numpy as np

# Config
policy_name = 'Pi0_Dp3'
task_name = 'beat_block_hammer'
task_config = 'demo_clean'
ckpt_setting = 'demo_clean'
expert_data_num = 50
seed = 42

usr_args = {
    'task_name': task_name,
    'task_config': task_config,
    'ckpt_setting': ckpt_setting,
    'expert_data_num': expert_data_num,
    'seed': seed,
    'policy_name': policy_name,
    'instruction_type': 'unseen',
    'checkpoint_num': 3000,
    'use_light_vlm': True,
    'left_arm_dim': 7,
    'right_arm_dim': 7,
}

# Load model
policy_module = importlib.import_module(policy_name)
get_model = getattr(policy_module, 'get_model')
eval_fn = getattr(policy_module, 'eval')
reset_fn = getattr(policy_module, 'reset_model')

print('Loading model...')
model = get_model(usr_args)

# Load env config
with open(f'./task_config/{task_config}.yml', 'r') as f:
    args = yaml.load(f.read(), Loader=yaml.FullLoader)
args['task_name'] = task_name
args['task_config'] = task_config
args['ckpt_setting'] = ckpt_setting
args['eval_mode'] = True
args['eval_video_log'] = False
args['render_freq'] = 0

# Create env
envs_module = importlib.import_module(f'envs.{task_name}')
TASK_ENV = envs_module.beat_block_hammer()

# Run one episode
TASK_ENV.setup_demo(now_ep_num=0, seed=4300000, is_test=True, **args)
TASK_ENV.play_once()
TASK_ENV.close_env()

# Re-setup for eval
TASK_ENV.setup_demo(now_ep_num=0, seed=4300000, is_test=True, **args)
reset_fn(model)

step = 0
np.set_printoptions(precision=3, suppress=True, linewidth=200)

while step < 100:
    observation = TASK_ENV.get_obs()
    ap = observation['joint_action']['vector']
    
    # Print state every 5 steps
    if step % 5 == 0:
        print(f'\n=== STEP {step} ===')
        print(f'  STATE: left_grip={ap[6]:.3f}, right_grip={ap[13]:.3f}, '
              f'right_j0={ap[7]:.3f}, right_j1={ap[8]:.3f}')
    
    eval_fn(TASK_ENV, model, observation)
    step += 1
    
    if TASK_ENV.eval_success:
        print(f'\nSUCCESS at step {step}')
        break

if not TASK_ENV.eval_success:
    print(f'\nFAIL after {step} steps')

TASK_ENV.close_env()
