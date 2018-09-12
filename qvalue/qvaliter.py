from collections import defaultdict

def get_max_q_for_state(q_star, state):
    if len(q_star[state].values()) == 0:
        return 0.
    return max(q_star[state].values())

def get_q_vals(world,
               q_star,
               state,
               actions,
               noise,
               gamma):
    q = {action:0. for action in actions}
    for action in actions:
        is_end, n_state = world.move_given_action(state, action)
        if is_end:
            q[action] = n_state
        else:
            lr_states = world.move_lr_given_action(state, action)
            q_n_state = (1. - noise) * (world.get_reward(state, action, n_state) +
                get_max_q_for_state(q_star, n_state) * gamma)
            q_l_state = (noise / 2) * (world.get_reward(state, action, lr_states[0]) +
                get_max_q_for_state(q_star, lr_states[0]) * gamma)
            q_r_state = (noise / 2) * (world.get_reward(state, action, lr_states[1]) +
                get_max_q_for_state(q_star, lr_states[1]) * gamma)
            q[action] = q_n_state + q_l_state + q_r_state

    return q

def qvalue_iter(world,
                noise=0.2,
                gamma=0.99,
                h=100,
                q_star_init=None,
                verbose=5):
    if q_star_init is not None:
        q_star = q_star_init
    else:
        q_star = defaultdict(lambda: {})

    for k in range(1, h+1):
        for state in world.states:
            possible_actions = world.actions_available(state)
            q_star[state] = get_q_vals(world, q_star,
                state, possible_actions, noise, gamma)
        if k % verbose == 0:
            print('Horizon {}/{}'.format(k, h), '-'*30)
            world.display_world_q_vals(q_star)
            print('\n')
    return q_star
