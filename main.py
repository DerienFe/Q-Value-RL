from environment.gridworld import Grid
from qvalue import qvalue_iter
import json
import argparse

def li_to_tu(blckd_states):
    return [tuple(ele) for ele in blckd_states]

def run(json_path):
    cfgs = json.load(open(json_path))
    world_cfgs = cfgs['env']
    val_fn_cfgs = cfgs['val_fn']
    world = Grid(
        x_range=world_cfgs['x_range'],
        y_range=world_cfgs['y_range'],
        pos_reward_states=li_to_tu(world_cfgs['pos_rwd_state']),
        neg_reward_states=li_to_tu(world_cfgs['neg_rwd_states']),
        pos_reward_vals=world_cfgs['pos_rwd_vals'],
        neg_reward_vals=world_cfgs['neg_rwd_vals'],
        blocked_states=li_to_tu(world_cfgs['blocked_states'])
    )
    q = qvalue_iter(
        world,
        noise=val_fn_cfgs['noise'],
        gamma=val_fn_cfgs['gamma'],
        h=val_fn_cfgs['h'],
        verbose=val_fn_cfgs['verbose']
    )
    return q, world

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, default='./main.json',
                        help='Give path to config json; Default - "./main.json"')
    args = parser.parse_args()
    q, world = run(args.json_path)

    print('Final V Values: ')
    world.display_world_all_q_vals(q)
    print('\n')
    print('Final Policy Values: ')
    world.display_world_pi_vals(q)
