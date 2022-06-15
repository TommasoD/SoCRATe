# SoCRATe
A Recommendation System with Limited Availability Items by Davide Azzalini, Fabio Azzalini, Chiara Criscuolo, Tommaso Dolci, Davide Martinenghi, Sihem Amer-Yahia.

### System Description
This repository contains the code for SoCRATe, allowing to simulate compensation strategies on real-world and synthetic datasets, in a limited availability of items scenario.
At the moment, two datasets are available to be analysed, *Amazon Movies and TV* and *Amazon Digital Music*, in addition to a synthetic dataset with a custom number of users and items.

File `orchestrator` is the main of the system, 
To select the desired dataset, you need to specify the correct file paths in lines 41 and 42, and select the abbreviation used to identify the system output.
The user can select different options. Among them:
- `choice_model_option` can be `top_k` (rank-based), `random`, `utility`;
- `sorting_option` can be `no_sort` (baseline option), `random`, `loss`, `historical` (default value);
- `compensation_strategy` can be `item` (round-robin), `user` (preference-drive);
- `synthetic` flag can be set to `True` to analyse a synthetic dataset, with number of items and users to be specified at line 53.

Additionally, the user can select the mean number of availability for the items in the simulation, the number of system iterations and more.

Simulation output is saved in `system_output` under a folder named after the chosen simulation options (e.g. T15-Aitem-Crandom-Shistorical for a 15 iterations simulation, with assignment of items according to round-robin compensation strategy (`item`), choice model `random`, sorting option `historical`).

Line 147 can be uncommented to save basic plots on the system performance, showing loss and cumulative loss for certain specific users.
Plots are saved in `plots`, where some examples of performance are already present.

### First Time Simulation
The system computes the optimal user recommendations and the utility matrix at the beginning of the execution. This process takes some time, and the output files are too large to be uploaded here.
Therefore, we give you the possibility to download the pre-computed files, which must be copied into the folder `obj_functions`, from here:

> link

Alternatively, you can compute them directly in the system, by uncommenting lines 80-85. Resulting files are automatically saved under the folder `obj_functions`.
