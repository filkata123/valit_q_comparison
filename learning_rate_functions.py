def inverse_time_decay(episode, alpha_0, decay_rate = 0.001, omega = 1):
    return alpha_0 / ((1 + decay_rate * episode) ** omega)

def polynomial_decay(num_steps, omega = 1):
    return 1 / (num_steps + 1) ** omega

def polynomial_decay_normalized(num_steps, num_states, omega = 1):
    return 1 / (num_steps/num_states + 1) ** omega

def visit_count_decay(visits, omega = 1):
    return 1 / (visits**omega)