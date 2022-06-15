import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json


def plot_system(t, users, utility, price, name):

    print("Plotting...")

    matplotlib.rc('xtick', labelsize=16)
    matplotlib.rc('ytick', labelsize=16)

    folder_path = 'system_output/' + name + '/'
    specific_user1 = users[0]
    specific_user2 = users[1]
    specific_user3 = users[2]
    specific_user4 = users[3]

    max_loss = []
    min_loss = []
    mean_loss = []
    std_loss = []
    historical_loss1 = []
    historical_loss2 = []
    historical_loss3 = []
    historical_loss4 = []
    u_loss1 = []
    u_loss2 = []
    u_loss3 = []
    u_loss4 = []
    happiness = []
    total_availability = []

    for t in range(t):

        with open(folder_path + 'losses' + str(t) + '.json') as json_file:
            losses = json.load(json_file)
        with open(folder_path + 'hist_loss' + str(t) + '.json') as json_file:
            hist_loss = json.load(json_file)

        # aggregate loss on objective functions
        for u in losses:
            losses[u] = sum(losses[u])

        max_loss.append(max(list(hist_loss.values())))
        min_loss.append(min(list(hist_loss.values())))
        mean_loss.append(np.mean(list(hist_loss.values())))
        std_loss.append(np.std(list(hist_loss.values())))
        u_loss1.append(losses[specific_user1])
        u_loss2.append(losses[specific_user2])
        u_loss3.append(losses[specific_user3])
        u_loss4.append(losses[specific_user4])
        historical_loss1.append(hist_loss[specific_user1])
        historical_loss2.append(hist_loss[specific_user2])
        historical_loss3.append(hist_loss[specific_user3])
        historical_loss4.append(hist_loss[specific_user4])

    fig, axs = plt.subplots(12, figsize=(8, 25))

    axs[0].set_title("max user loss", fontsize=18)
    axs[0].plot(max_loss)

    axs[1].set_title("min user loss", fontsize=18)
    axs[1].plot(min_loss)

    axs[2].set_title("avg user loss", fontsize=18)
    axs[2].plot(mean_loss)

    axs[3].set_title("std_dev user loss", fontsize=18)
    axs[3].plot(std_loss)

    axs[4].set_title(specific_user1 + "'s loss", fontsize=18)
    axs[4].plot(u_loss1)
    # axs[4].set_ylim(bottom=-0.05, top=0.6)
    axs[5].set_title(specific_user2 + "'s loss", fontsize=18)
    axs[5].plot(u_loss2)
    # axs[5].set_ylim(bottom=-0.05, top=0.6)
    axs[6].set_title(specific_user3 + "'s loss", fontsize=18)
    axs[6].plot(u_loss3)
    # axs[6].set_ylim(bottom=-0.05, top=0.6)
    axs[7].set_title(specific_user4 + "'s loss", fontsize=18)
    axs[7].plot(u_loss4)
    # axs[7].set_ylim(bottom=-0.05, top=0.6)

    axs[8].set_title(specific_user1 + "'s historical loss", fontsize=18)
    axs[8].plot(historical_loss1)
    # axs[8].set_ylim(bottom=-1, top=5)
    axs[9].set_title(specific_user2 + "'s historical loss", fontsize=18)
    axs[9].plot(historical_loss2)
    # axs[9].set_ylim(bottom=-1, top=5)
    axs[10].set_title(specific_user3 + "'s historical loss", fontsize=18)
    axs[10].plot(historical_loss3)
    # axs[10].set_ylim(bottom=-1, top=5)
    axs[11].set_title(specific_user4 + "'s historical loss", fontsize=18)
    axs[11].plot(historical_loss4)
    # axs[11].set_ylim(bottom=-1, top=5)

    # # COMPUTE PERCENTAGE OF AVAILABLE ITEMS
    # with open(folder_path + 'availabilities' + str(t) + '.json') as json_file:
    #     availabilities = json.load(json_file)
    # availabilities.append(
    #     100 - (((list(availabilities.values()).count(0)) / len(list(availabilities.values()))) * 100))
    # total_availability.append(sum(list(availabilities.values())))

    # # COMPUTE HAPPINESS
    # with open(folder_path + 'R' + str(t) + '.json') as json_file:
    #     R = json.load(json_file)
    # all_happiness = []
    # for u in R.keys():
    #     happiness = []
    #     for i in R[u]:
    #         happiness.append((price[i] + utility[i][u]) / 2)
    #     all_happiness.append(np.mean(happiness))
    # happiness.append(np.mean(all_happiness))

    # axs[5].set_title("avg user happiness", fontsize=18)
    # axs[5].plot(happiness)
    #
    # axs[6].set_title("% of items available", fontsize=18)
    # axs[6].plot(availabilities)
    #
    # axs[7].set_title("quantity of items available", fontsize=18)
    # axs[7].plot(total_availability)

    fig.tight_layout()
    plt.savefig("plots/plot-{}.pdf".format(name))
