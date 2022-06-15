import copy
import random


def choice_model(r, k, recommendations, availabilities, option):  # options: 'top_k', 'random', 'utility'
    s = copy.deepcopy(r)

    for user in s.keys():
        if option == 'top_k':
            s[user] = s[user][0:k]
        elif option == 'random':
            s[user] = random.sample(s[user], k=k)
        else:
            print('no choice')

        for item in r[user]:
            if item in s[user]:  # remove adopted items for each user
                recommendations[user].remove(item)
            else:  # restore availability of the item
                availabilities[item] = availabilities[item] + 1

    # # manage availabilities (NOW IT'S DONE ELSEWHERE)
    # for item in availabilities.keys():
    #     if availabilities[item] == 0:
    #         for user in recommendations.keys():
    #             if item in recommendations[user]:
    #                 recommendations[user].remove(item)

    return s
