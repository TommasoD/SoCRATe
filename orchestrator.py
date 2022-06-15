import numpy as np
import pandas as pd
import json

# Parameters
from assignment import item_assignment
from choice_model import choice_model
from initialisation import normal_initialise_availability, utility_initialise
from loss import top_n, loss_computation, top_n_availabilities, update_hist_loss
from objectives_initialise import price_initialise
from plot import plot_system
from rerank import re_rank
from synthetic_ds import create_synthetic_data, synthetic_utility, synthetic_recs, synthetic_price
from user_sorting import user_sorting
from weights import update_weights
from utils import write_to_file, preprocessing, train_recommender, get_recommendations

T = 15  # number of iterations of SoCRATe
N = 5  # number of items actually recommended
M = 150  # number of items returned by the single-user recommender
K = 3  # number of adopted items
alpha_1 = 0.5  # weight of objective functions item-dependant
alpha_2 = 0.5  # weight of utility
beta = 0.1  # parameter for weight update
gamma = 0.1  # parameter for weight update
delta = 0.1  # parameter for weight update
R = {}
OptR_hat = {}

# Options
choice_model_option = "top_k"  # options: 'top_k', 'random', 'utility'
sorting_option = 'historical'  # options: 'no_sort', 'random', 'loss', 'historical'
compensation_strategy = 'item'  # options: 'item', 'user', 'hybrid'
synthetic = False
mean_availability = 30  # optimal value for T=15 is 10 for synth, 150 for az-music, 30 for az-movie
print("Choice model option: " + choice_model_option)
print("Sorting strategy: " + str(sorting_option))
print("Compensation strategy: " + str(compensation_strategy))

# Dataset (CHANGE PATH AND ABBREVIATION ACCORDING TO THE TARGET DATASET)
data_file_path = 'datasets/reviewsAmazonMoviesRecommenderSystem.csv'
metadata_file_path = 'datasets/metadataAmazonMoviesRecommenderSystem.csv'
abbreviation = 'az-movie'  # synth, az-music, az-movie

test_name = "T{}-A{}-C{}-S{}".format(T, compensation_strategy, choice_model_option, sorting_option)
if synthetic:
    test_name = 'synth-' + test_name
else:
    test_name = abbreviation + '-' + test_name

# Use synthetic dataset
if synthetic:
    items, users = create_synthetic_data(n_items=1500, n_users=100)
    objective_functions_on_item = synthetic_price(items)
    recommendations = synthetic_recs(items, users)
    utility_matrix = synthetic_utility(recommendations, items, users)
    write_to_file(recommendations, "obj_functions/recommendations_synth.json")
    write_to_file(objective_functions_on_item, "obj_functions/objectives_synth.json")
    write_to_file(utility_matrix, "obj_functions/utility_matrix_synth.json")
else:
    # Data preprocessing
    data_train = pd.read_csv(data_file_path).groupby(
        ['CUST_ID', 'ARTICLE_ID'])['RATING'].mean().reset_index(name='FREQUENCY')
    items, users, user_indexes, ratings = preprocessing(data_train)

    # Create and train a new single-user recommender
    knn_recommender = train_recommender(ratings)

    # Initialise objective functions + utility + recommendations
    print("Loading objective functions...")
    with open("obj_functions/objectives_{}.json".format(abbreviation)) as read_file:
        objective_functions_on_item = json.load(read_file)
    with open("obj_functions/utility_matrix_{}.json".format(abbreviation)) as read_file:
        utility_matrix = json.load(read_file)
    print("Loading recommendations...")
    with open("obj_functions/recommendations_{}.json".format(abbreviation)) as read_file:
        recommendations = json.load(read_file)

    # UNCOMMENT IF OBJECTIVES ARE NOT ALREADY COMPUTED
    # objective_functions_on_item = price_initialise(metadata_file_path)
    # write_to_file(objective_functions_on_item, "obj_functions/synth_objectives_{}.json".format(abbreviation))
    # utility_matrix = utility_initialise(users, user_indexes, knn_recommender, items)
    # write_to_file(utility_matrix, "obj_functions/synth_utility_matrix_{}.json".format(abbreviation))
    # recommendations = get_recommendations(users, user_indexes, ratings, len(items), knn_recommender, items)
    # write_to_file(recommendations, "obj_functions/synth_recommendations_{}.json".format(abbreviation))

print('# Users:', len(users))
print('# Items:', len(items))

# Users ID for plotting results
length = len(users)
plot_users = [users[0], users[length//3], users[length//3*2], users[length-1]]

# Initialise weights + loss
keys = users
value = [alpha_1, alpha_2]
weights = {key: list(value) for key in keys}

value = [0, 0]
losses = {key: list(value) for key in keys}
historical_loss = {key: 0 for key in keys}

folder_name = "system_output/" + test_name + "/"

# Assign availability to each item
availabilities = dict(zip(items, np.zeros(len(items), dtype=int)))
normal_initialise_availability(availabilities, mean_availability)
write_to_file(availabilities, folder_name + "availabilities0.json")
write_to_file(weights, folder_name + "weights0.json")

for t in range(0, T):
    print('Starting iteration', t+1)  # for each iteration except the first one, execute choice model and weight update
    if t:
        # Compute adopted items
        S = choice_model(R, K, recommendations, availabilities, choice_model_option)

        # write_to_file(S, folder_name + "adopted_items" + str(t) + ".json")
        write_to_file(availabilities, folder_name + "availabilities" + str(t) + ".json")

        # Update weights
        weights = update_weights(weights, S, R, OptR_hat, objective_functions_on_item,
                                 utility_matrix, beta, gamma, delta, N)
        write_to_file(weights, folder_name + "weights" + str(t) + ".json")

    # Take first M recommendations
    top_m_recommendations = top_n_availabilities(recommendations, M, availabilities)  # defined in loss.py

    OptR = re_rank(top_m_recommendations, users, weights, objective_functions_on_item, utility_matrix)
    write_to_file(OptR, folder_name + "OptR" + str(t) + ".json")

    if sorting_option != 'no_sort':
        users = user_sorting(losses, historical_loss, users, weights, sorting_option)
    write_to_file(users, folder_name + "sorted_users" + str(t) + ".json")

    R = item_assignment(availabilities, OptR, users, N, compensation_strategy)
    # write_to_file(R, folder_name + "R" + str(t) + ".json")

    OptR_hat = top_n(OptR, N)  # get top N from OptR (function is defined in loss.py)
    # write_to_file(OptR_hat, folder_name + "OptR_hat" + str(t) + ".json")

    losses = loss_computation(R, OptR_hat, users, objective_functions_on_item, utility_matrix)
    historical_loss = update_hist_loss(historical_loss, losses, weights)
    write_to_file(losses, folder_name + "losses" + str(t) + ".json")
    write_to_file(historical_loss, folder_name + "hist_loss" + str(t) + ".json")

# Save plot analysis
# plot_system(T, plot_users, utility_matrix, objective_functions_on_item, test_name)
