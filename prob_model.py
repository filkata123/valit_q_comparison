def probability_model(num_neigbors, prob_success = 0.9):
    prob_stay = 0
    prob_other= 0
    if num_neigbors > 1: 
        prob_stay = (1 - prob_success) / 2
        prob_other = (1 - prob_success - prob_stay) / (num_neigbors - 1)
    else:
        prob_stay = 1 - prob_success
        prob_other = 0.0
    return prob_success, prob_stay, prob_other
