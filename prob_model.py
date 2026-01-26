# ## distribution of non-successful action choices is not uniform
def probability_model(num_neighbours, prob_success = 0.9):
    prob_stay = 0
    prob_other= 0
    if num_neighbours > 1: 
        prob_stay = (1 - prob_success) / 2
        prob_other = (1 - prob_success - prob_stay) / (num_neighbours - 1)
    else:
        prob_stay = 1 - prob_success
        prob_other = 0.0
    return prob_success, prob_stay, prob_other

## Uniform distribution for non-chosen actions
# def probability_model(num_neighbours, prob_success = 0.9):
#     prob_stay = 0
#     prob_other= 0
#     if num_neighbours > 1:
#         prob_other = (1 - prob_success) / (num_neighbours) # we don't take into account the chosen action as a neighbour but we count the stay action so we divide by num_neighbours
#         prob_stay = prob_other
#     else:
#         prob_stay = 1 - prob_success
#         prob_other = 0.0
#     return prob_success, prob_stay, prob_other

# Transition probability can be defined as
# H = -[p_success*ln(p_success) + p_stay*ln(p_stay) + (num_neighborus - 1) * p_other*ln(p_other)]
