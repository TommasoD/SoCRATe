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
N = 5  # number of items recommended to the user
M = 150  # number of items returned by the single-user recommender (does only change performance of the system)
K = 3  # number of adopted items
alpha_1 = 0.5  # weight of objective functions item-dependant
alpha_2 = 0.5  # weight of utility
beta = 0.1  # parameter for weight update
gamma = 0.1  # parameter for weight update
delta = 0.1  # parameter for weight update
R = {}
OptR_hat = {}

# General Options
mean_availability = 10
assignment_strategy = 'user'  # options: 'item', 'user'
choice_model_option = 'top_k'  # options: 'top_k', 'random', 'utility'
sorting_option = 'no_sort'  # options: 'no_sort', 'random', 'loss', 'historical'
time_granularity = 'group'  # options: 'fixed', 'group'
print("Choice model option: " + choice_model_option)
print("Sorting strategy: " + sorting_option)
print("Assignment strategy: " + assignment_strategy)
print("Time granularity: " + time_granularity)

# Dataset (DEFINE IT HERE)
dataset_abbr = 'crowd'  # Dataset options: az-music, az-movie, crowd, synth
data_file_path = ''
metadata_file_path = ''

# Dataset-related options
if dataset_abbr == "az-music":
    data_file_path = 'datasets/reviewsAmazonMusicRecommenderSystem.csv'
    metadata_file_path = 'datasets/metadataAmazonMusicRecommenderSystem.csv'
    M = 450  # tuned to the nature of the dataset
    mean_availability = 150  # optimal value for T=15
elif dataset_abbr == "az-movie":
    data_file_path = 'datasets/reviewsAmazonMoviesRecommenderSystem.csv'
    metadata_file_path = 'datasets/metadataAmazonMoviesRecommenderSystem.csv'
    M = 150
    mean_availability = 30  # optimal value for T=15
elif dataset_abbr == "crowd":
    data_file_path = 'datasets/reviewsCrowdsourcing.csv'
    metadata_file_path = 'datasets/metadataCrowdsourcing.csv'
    M = 200
    mean_availability = 5  # optimal value for T=15

# SYSTEM LOGIC STARTS FROM HERE

test_name = "T{}-A{}-C{}-S{}".format(T, assignment_strategy, choice_model_option, sorting_option)
if dataset_abbr == 'synth':
    test_name = 'synth-' + test_name
    print("Dataset: synthetic")
else:
    test_name = dataset_abbr + '-' + test_name
    print("Dataset: " + dataset_abbr)

if time_granularity == "group":
    test_name = 'group-' + test_name

if dataset_abbr == 'synth':  # use synthetic dataset
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
    with open("obj_functions/objectives_{}.json".format(dataset_abbr)) as read_file:
        objective_functions_on_item = json.load(read_file)
    with open("obj_functions/utility_matrix_{}.json".format(dataset_abbr)) as read_file:
        utility_matrix = json.load(read_file)
    print("Loading recommendations...")
    with open("obj_functions/recommendations_{}.json".format(dataset_abbr)) as read_file:
        recommendations = json.load(read_file)

    # # UNCOMMENT IF OBJECTIVES, UTILITY MATRIX AND RECOMMENDATIONS ARE NOT YET COMPUTED
    # objective_functions_on_item = price_initialise(metadata_file_path)
    # write_to_file(objective_functions_on_item, "obj_functions/objectives_{}.json".format(dataset_abbr))
    # utility_matrix = utility_initialise(users, user_indexes, knn_recommender, items)
    # write_to_file(utility_matrix, "obj_functions/utility_matrix_{}.json".format(dataset_abbr))
    # recommendations = get_recommendations(users, user_indexes, ratings, len(items), knn_recommender, items)
    # write_to_file(recommendations, "obj_functions/recommendations_{}.json".format(dataset_abbr))

print('\n# Users:', len(users))
print('# Items:', len(items), "\n")

# Users ID for plotting results
length = len(users)
plot_users = [users[0], users[length // 3], users[length // 3 * 2], users[length - 1]]

# Separate users into groups for user-group time granularity
# In a real-case scenario users are split according to their history (more tasks = faster consumer)
user_groups = [users[:length // 3], users[length // 3:length // 3 * 2], users[length // 3 * 2:]]
group_order = [0, 1, 2, 0, 1, 0, 0, 1, 2, 0, 1, 0, 0, 2, 1]

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

# --------- ITERATIONS START ---------

for t in range(0, T):
    print('Starting iteration',
          t + 1)  # for each iteration except the first one, execute choice model and weight update
    if t:
        # Compute adopted items
        S = choice_model(R, K, recommendations, availabilities, choice_model_option, utility_matrix)

        # write_to_file(S, folder_name + "adopted_items" + str(t) + ".json")
        write_to_file(availabilities, folder_name + "availabilities" + str(t) + ".json")

        # Update weights
        weights = update_weights(weights, S, R, OptR_hat, objective_functions_on_item,
                                 utility_matrix, beta, gamma, delta, N)
        write_to_file(weights, folder_name + "weights" + str(t) + ".json")

    # Take first M recommendations
    top_m_recommendations = top_n_availabilities(recommendations, M, availabilities)  # defined in loss.py

    # From here split into two directions: fixed time granularity and user-group
    if time_granularity == 'group':
        g = group_order[t]
        OptR = re_rank(top_m_recommendations, user_groups[g], weights, objective_functions_on_item, utility_matrix)
        write_to_file(OptR, folder_name + "OptR" + str(t) + ".json")

        if sorting_option != 'no_sort':
            users = user_sorting(losses, historical_loss, users, weights, sorting_option)
            user_groups[g] = user_sorting(losses, historical_loss, user_groups[g], weights, sorting_option)
        write_to_file(users, folder_name + "sorted_users" + str(t) + ".json")

        R = item_assignment(availabilities, OptR, user_groups[g], N, assignment_strategy)
        OptR_hat = top_n(OptR, N)  # get top N from OptR (function is defined in loss.py)
        losses = loss_computation(R, OptR_hat, users, objective_functions_on_item, utility_matrix)
    else:
        OptR = re_rank(top_m_recommendations, users, weights, objective_functions_on_item, utility_matrix)
        write_to_file(OptR, folder_name + "OptR" + str(t) + ".json")

        if sorting_option != 'no_sort':
            users = user_sorting(losses, historical_loss, users, weights, sorting_option)
        write_to_file(users, folder_name + "sorted_users" + str(t) + ".json")

        R = item_assignment(availabilities, OptR, users, N, assignment_strategy)
        OptR_hat = top_n(OptR, N)  # get top N from OptR (function is defined in loss.py)
        losses = loss_computation(R, OptR_hat, users, objective_functions_on_item, utility_matrix)

    historical_loss = update_hist_loss(historical_loss, losses, weights)
    write_to_file(losses, folder_name + "losses" + str(t) + ".json")
    write_to_file(historical_loss, folder_name + "hist_loss" + str(t) + ".json")

# Save plot analysis
plot_system(T, plot_users, utility_matrix, objective_functions_on_item, test_name)
