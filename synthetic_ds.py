import random
from random import sample
import numpy.random as rd


def create_synthetic_data(n_items, n_users):
    random.seed(0)
    user_array = []
    for i in range(n_users):
        new_user = "u" + str(i)
        user_array.append(new_user)

    item_array = []
    items_id = sample(range(1, 10000), n_items)
    for item in items_id:
        new_item = "#00" + str(item)
        item_array.append(new_item)
    return item_array, user_array


def synthetic_price(items):
    rd.seed(0)
    price_list = rd.random_sample(size=(len(items))) / 10
    prices = dict(zip(items, price_list))
    return prices


def synthetic_recs(items, users):
    keys = users
    value = items
    recommendations = {key: list(value) for key in keys}
    return recommendations


def synthetic_utility(recommendations, items, users):
    utility_matrix_u = {}
    for user in users:
        u_recommendations = recommendations[user]
        item_score = {}

        for item in u_recommendations:
            score = (len(u_recommendations) - u_recommendations.index(item)) / len(u_recommendations)
            item_score[item] = score

        utility_matrix_u[user] = item_score

    utility_matrix_i = {}
    for i in items:
        user_score = {}
        for u in users:
            user_score[u] = utility_matrix_u[u][i]
        utility_matrix_i[i] = user_score

    return utility_matrix_i
