"""!
"""


from random import randint, choice
import pandas as pd
import pickle
import numpy as np

STARTING_COORDS = [0,6]
OOB_COORDS_RIGHT = [[9,6], [10,6], [11,6], [10,5], [11,5], [11,4]]
OOB_COORDS_LEFT = [[0,0], [1,0], [0,1]]
OOB_COORDS_OCEAN = [[2,0], [3,0], [4,0], [5,0], [6,0]]
LANDED_COORDS = [[7,0], [8,0], [9,0], [10,0], [11,0]]

BASE_ACTION_INVERSE_MAPPING = {
    0: "NOTHING",
    1: "GLIDE",
    2: "DIVE"
}

PILOT_ACTION_INVERSE_MAPPING = {
    0: "NOTHING",
    1: "+X",
    2: "-X",
    3: "+Y",
    4: "-Y"
}


def clean_dataframe(df):
    transform_dict = {
        'base_action': {"NOTHING": 0, "GLIDE": 1, "DIVE": 2},
        'pilot_action': {"NOTHING": 0, "+X": 1, "-X": 2, "+Y": 3, "-Y": 4}
    }

    df = df.replace(transform_dict)
    df = df.astype(int)
    return df

def land_the_plane(pilot_licence=False, clf=None):
    """!
    """

    # Dice options: 1 down, 1 across, 1, 2, 3, 4
    # We will treat a roll of 5 as 1 across, 6 as 1 down

    columns = ['x_coord', 'y_coord', 'roll_1', 'roll_2', 'base_action', 'pilot_action']
    df = pd.DataFrame(columns=columns)

    landed = False
    crashed = False

    current_coords = STARTING_COORDS.copy()

    while not (landed or crashed):

        coords_this_iter = current_coords.copy()

        pilot_action = ""
        roll_1 = randint(1,6)
        roll_2 = randint(1,6)

        # print("Starting Coords", current_coords)
        # print("Rolled", roll_1, roll_2)

        base_options = ["GLIDE", "DIVE"]
        base_action = "NOTHING"

        if pilot_licence:
            pilot_options = ["+X", "-X", "+Y", "-Y", "NOTHING"]
        else:
            pilot_options = ["NOTHING"]

        # If either roll forces an action, then the base options
        # is Nothing, you don't have a choice, your action is forced.
        # BUT, if you have a pilot licence, you can still choose a pilot
        # action.
        if roll_1 in [5,6] or roll_2 in [5,6]:
            if pilot_licence:
                if clf != None:
                    test_df = pd.DataFrame(columns=columns)
                    for base_option in ["NOTHING"]:
                        for pilot_option in pilot_options:
                            test_df.loc[len(test_df)] = [coords_this_iter[0],coords_this_iter[1],roll_1,roll_2,base_option,pilot_option]
                    test_df = clean_dataframe(test_df)
                    chance_of_success = clf.predict_proba(test_df)
                    # print(chance_of_success)

                    cos_list = chance_of_success.tolist()
                    max_val = 0
                    best_option_idx = 0
                    for idx, entry in enumerate(cos_list):
                        val = entry[1]
                        if val > max_val:
                            max_val = val
                            best_option_idx = idx
                    # print(max_val, best_option_idx)
                    best_row = test_df.iloc[[best_option_idx]]
                    pilot_action = PILOT_ACTION_INVERSE_MAPPING[best_row['pilot_action'].values[0]]
                else:
                    # Take one at random if no model
                    pilot_action = choice(pilot_options)
            else:
                pilot_action = "NOTHING"

        if roll_1 == 5:
            if pilot_action == "+X":
                current_coords[0] += 2
            elif pilot_action == "-X":
                # +1-1 = 0. Do nothing
                pass
            else:
                current_coords[0] += 1
            
            if roll_2 == 5:
                current_coords[0] += 1
            elif roll_2 == 6:
                if pilot_action == "+Y":
                    pass
                elif pilot_action == "-Y":
                    current_coords[1] -= 2
                else:
                    current_coords[1] -= 1
            else:
                if pilot_action == "+Y":
                    current_coords[1] -= (roll_2-1)
                elif pilot_action == "-Y":
                    current_coords[1] -= (roll_2+1)
                else:
                    current_coords[1] -= roll_2
        elif roll_1 == 6:
            if pilot_action == "+Y":
                pass
            elif pilot_action == "-Y":
                current_coords[1] -= 2
            else:
                current_coords[1] -= 1
            
            if roll_2 == 6:
                current_coords[1] -= 1
            elif roll_2 == 5:
                if pilot_action == "+X":
                    current_coords[0] += 2
                elif pilot_action == "-X":
                    pass
                else:
                    current_coords[0] += 1
            else:
                if pilot_action == "+X":
                    current_coords[0] += (roll_2+1)
                elif pilot_action == "-X":
                    current_coords[0] += (roll_2-1)
                else:
                    current_coords[0] += roll_2
        elif roll_2 == 5:
            if pilot_action == "+X":
                current_coords[0] += 2
            elif pilot_action == "-X":
                # +1-1 = 0. Do nothing
                pass
            else:
                current_coords[0] += 1
            
            if roll_1 == 5:
                current_coords[0] += 1
            elif roll_1 == 6:
                if pilot_action == "+Y":
                    pass
                elif pilot_action == "-Y":
                    current_coords[1] -= 2
                else:
                    current_coords[1] -= 1
            else:
                if pilot_action == "+Y":
                    current_coords[1] -= (roll_1-1)
                elif pilot_action == "-Y":
                    current_coords[1] -= (roll_1+1)
                else:
                    current_coords[1] -= roll_1
        elif roll_2 == 6:
            if pilot_action == "+Y":
                pass
            elif pilot_action == "-Y":
                current_coords[1] -= 2
            else:
                current_coords[1] -= 1
            
            if roll_1 == 6:
                current_coords[1] -= 1
            elif roll_1 == 5:
                if pilot_action == "+X":
                    current_coords[0] += 2
                elif pilot_action == "-X":
                    pass
                else:
                    current_coords[0] += 1
            else:
                if pilot_action == "+X":
                    current_coords[0] += (roll_1+1)
                elif pilot_action == "-X":
                    current_coords[0] += (roll_1-1)
                else:
                    current_coords[0] += roll_1
        else:
            if clf != None:
                test_df = pd.DataFrame(columns=columns)
                for base_option in base_options:
                    for pilot_option in pilot_options:
                        test_df.loc[len(test_df)] = [coords_this_iter[0],coords_this_iter[1],roll_1,roll_2,base_option,pilot_option]
                test_df = clean_dataframe(test_df)
                chance_of_success = clf.predict_proba(test_df)
                # print(chance_of_success)

                cos_list = chance_of_success.tolist()
                max_val = 0
                best_option_idx = 0
                for idx, entry in enumerate(cos_list):
                    val = entry[1]
                    if val > max_val:
                        max_val = val
                        best_option_idx = idx
                # print(max_val, best_option_idx)
                best_row = test_df.iloc[[best_option_idx]]
                base_action = BASE_ACTION_INVERSE_MAPPING[best_row['base_action'].values[0]]
                pilot_action = PILOT_ACTION_INVERSE_MAPPING[best_row['pilot_action'].values[0]]

                # print(base_action, pilot_action)
            else:
                # If not testing a model, then its totally random
                base_action = choice(base_options)
                pilot_action = choice(pilot_options)

            rolls = [roll_1, roll_2]
            max_roll = max(rolls)
            min_roll = min(rolls)

            if base_action == "GLIDE":
                if pilot_action == "+X":
                    current_coords[0] += (max_roll+1)
                    current_coords[1] -= min_roll
                elif pilot_action == "-X":
                    current_coords[0] += (max_roll-1)
                    current_coords[1] -= min_roll
                else:
                    current_coords[0] += max_roll
                    if pilot_action == "+Y":
                        current_coords[1] -= (min_roll-1)
                    elif pilot_action == "-Y":
                        current_coords[1] -= (min_roll+1)
                    else:
                        current_coords[1] -= min_roll
            elif base_action == "DIVE":
                if pilot_action == "+Y":
                    current_coords[1] -= (max_roll-1)
                    current_coords[0] += min_roll
                elif pilot_action == "-Y":
                    current_coords[1] -= (max_roll+1)
                    current_coords[0] += min_roll
                else:
                    current_coords[1] -= max_roll
                    if pilot_action == "+X":
                        current_coords[0] += (min_roll+1)
                    elif pilot_action == "-X":
                        current_coords[0] += (min_roll-1)
                    else:
                        current_coords[0] += min_roll
            else:
                print("HOW ARE YOU HERE?")

            # print(base_action)

        df.loc[len(df)] = [coords_this_iter[0],
                           coords_this_iter[1],
                           roll_1,
                           roll_2,
                           base_action,
                           pilot_action,
                        ]

        # Check for crash / land
        if current_coords in LANDED_COORDS:
            landed = True
        elif current_coords in OOB_COORDS_RIGHT:
            crashed = True
        elif current_coords in OOB_COORDS_LEFT:
            crashed = True
        elif current_coords in OOB_COORDS_OCEAN:
            crashed = True
        elif current_coords[0] > 11:
            crashed = True
        elif current_coords[1] < 0:
            crashed = True
        
        # print("New Coords", current_coords)

    
    if landed:
        # print("Landed!")
        x_land = current_coords[0]
        roll_final = randint(1,6)
        if (roll_final == 1) or (roll_final == 5):
            if pilot_licence:
                res = "WIN"
            else:
                if x_land < 11:
                    res = "WIN"
                else:
                    res = "LOSS"
        elif roll_final == 2:
            if pilot_licence and x_land < 11:
                res = "WIN"
            else:
                if x_land < 10:
                    res = "WIN"
                else:
                    res = "LOSS"
        elif roll_final == 3:
            if pilot_licence and x_land < 10:
                res = "WIN"
            else:
                if x_land < 9:
                    res = "WIN"
                else:
                    res = "LOSS"
        elif roll_final == 4:
            if pilot_licence and x_land < 9:
                res = "WIN"
            else:
                if x_land < 8:
                    res = "WIN"
                else:
                    res = "LOSS"
        elif roll_final == 6:
            if pilot_licence:
                res = "WIN"
            else:
                res = "LOSS"
        else:
            print("HOW DID YOU GET HERE?")
    else:
        res = "LOSS"

    if res == "WIN":
        df['outcome'] = 1
    else:
        df['outcome'] = 0

    return res, df


def generate_dataset(landing_attempts=1000, clf_str=None):
    columns = ['x_coord', 'y_coord', 'roll_1', 'roll_2', 'base_action', 'pilot_action', 'outcome']
    
    try:
        df_master = pd.read_csv('master.csv')
        # print(df_master)
        print(df_master['outcome'].value_counts())
    except:
        print("file doesn't exist")
        df_master = pd.DataFrame(columns=columns)

    if clf_str != None:
        clf = pickle.load(open(clf_str, 'rb'))
    else:
        clf = None

    for i in range(1000):
        _, df = land_the_plane(pilot_licence=True, clf=clf)
        df_master = pd.concat([df_master,df], axis=0)
    
    df_master = df_master.reset_index(drop=True)

    df_master.to_csv("master.csv", index=False)

    return df_master

def ml_stuff(df, iter):
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from imblearn.under_sampling import RandomUnderSampler
    import pickle

    df = clean_dataframe(df)

    X = df[['x_coord', 'y_coord', 'roll_1', 'roll_2', 'base_action', 'pilot_action']]
    y = df[['outcome']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    under_sampler = RandomUnderSampler(random_state=42)
    X_res, y_res = under_sampler.fit_resample(X_train, y_train)

    clf = MLPClassifier(random_state=1, max_iter=1000).fit(X_res, y_res)

    model_score = clf.score(X_test, y_test)
    print(model_score)
    
    pickle.dump(clf, open('mlp_{}_{}.pickle'.format(round(model_score*100), iter), 'wb'))
    
if __name__ == "__main__":

    # number_of_models_to_make = 10
    # for iter in range(number_of_models_to_make):
    #     # df = generate_dataset(landing_attempts=1000, clf_str=None)
    #     df = generate_dataset(landing_attempts=10000, clf_str="mlp_with_licence.pickle")
    #     ml_stuff(df, iter)

    wins = 0
    losses = 0
    clf = pickle.load(open("mlp_no_licence.pickle", 'rb'))
    for i in range(5000):
        res, df = land_the_plane(False, clf)
        if res == "WIN":
            wins += 1
        else:
            losses += 1
    print("NO LICENCE... Wins: {}, Losses: {}. Win Ratio: {}".format(wins, losses, wins/losses))

    wins = 0
    losses = 0
    clf = pickle.load(open("mlp_with_licence.pickle", 'rb'))
    for i in range(5000):
        res, df = land_the_plane(True, clf)
        if res == "WIN":
            wins += 1
        else:
            losses += 1
    print("WITH LICENCE... Wins: {}, Losses: {}. Win Ratio: {}".format(wins, losses, wins/losses))