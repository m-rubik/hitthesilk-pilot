# hitthesilk-pilot
:grey_exclamation: This project uses machine learning [Multi-layer Perceptron Classifiers](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) models to learn how to land the plane in the [_Hit the Silk!_](https://escapeplanboardgames.com/hit-the-silk/) board game. 

The objective is to evaluate how powerful the in-game "Pilot Licence" item is by comparing the probability of successfully landing the plane with and without it.

TL;DR (*spoiler*) -- The Pilot Licence quadruples the chance to successfully land the plane. The strength of the item is mostly in its ability to reduce the falling speed of the plane.

# Background
_Hit the Silk!_ [[link](https://escapeplanboardgames.com/hit-the-silk/)] is a semi-cooperative game in which player's find themselves in a heist-gone-wrong scenario in which everything was going to plan until the pilot seized a gun from the lockbox, opened the emergency door and fired three rounds into the engine before bailing with his parachute and a spare.

The team's objective is to collectively secure a target sum of cash. Individually, the goal is to stay alive by securing a parachute and jumping before the plane crashes.

As a last resort, players who did not jump from the plane (i.e, those that did not _Hit the Silk!_) may attempt to land the plane.

Landing the plane is conceptually simple. The player controls a plane token on a simple grid system:

![base](/media/1.svg)

The objective is:
- Navigate the plane token from the "START" space down and across the grid over to the one of the 5 spaces on the landing strip
- :heavy_check_mark: Once on the landing strip, hit the brakes and come to a standstill on any of the landing strip spaces; a successful landing!
- :x: If the plane ever moves off the right end of any row, it crashes into a mountain; landing fails
- :x: If the plane ever moves down into the water, it crashes into the water; landing fails
- :x: If the plane ever moves down below the landing strip surface, it crashes into the ground; landing fails

This navigation is controlled through rounds of dice rolls. Each round, the player rolls 2 dice and moves the plane token accordingly. This continues until the plane crashes or is landed.

The rules for the dice rolls are as described in the game's manual:

![base](/media/rolling_rules.PNG)

The player must, when possible, try to make the best decision on how to apply each dice roll (which direction to take on each die, if the option is available) to have the best chance of landing the plane.

### Special rolling round -- hitting the brakes
Once on the landing strip, the player must hit the brakes and come to a stop.
> "Roll one dice to determine the final move across the landing strip. If you roll a down indicator, you lose. If you pass beyond the final space into the mountain, you crash and lose. If you come to a standstill on the landing strip, you have survived."

## The Twist -- Pilot Licence
There is an item in the game called the Pilot Licence that players can use during their dice rolls to land the plane. Its ability reads:

> the pilot licence allows you to deduct a value of 1 from one of the dice, on each turn.

This gives players a bit more agency over their fate, allowing them to have a small modification on the randomness of their dice rolls. But how powerful is this item...?

# Inspiration
When I played this game with friends, I found (albeit in the limited number of games we played) that we were quite consistently able to land the plane with the Pilot Licence.

This got me thinking, **just how easy is it to land the plane**, with and without the Pilot Licence? Is the Pilot Licence an un-balanced item?


# Analysis
Landing the plane is not all up to fate. Though a player's success relies heavily on the randomness of the dice rolls aligning with their desired values at different stages during the landing, the player still, sometimes, has some amount of control over how to move their token and secure the landing. The Pilot Licence adds an extra bit of flavour to the game by giving the player an extra bit of control during the landing process.

## The Environment
The landing environment (grid) was replicated in code to allow for running thousands of simulated landings.

To play in this environment, I created 2 seperate MLPClassifiers as our test dummies.

The "regular" model does not have a Pilot Licence, the "pilot" model does.

The regular model's action space is very limited. Depending on what it rolls, it really only has 3 options.
1. Nothing! This happens when dice rolls are identical, or when both die force a particular direction, etc.
2. Dive! This means to use the bigger number of the two rolls to move down (vertical dive).
3. Glide! This means to use the bigger number of the two rolls to move across (horizontal glide).

The pilot model gets the base action space, but it also gets the choice of an *additional*, special, Pilot action which can be:
1. Nothing!
2. Glide more (+1 horizontal)
3. Glide less (-1 horizontal)
4. Dive more (+1 vertical drop)
5. Dive less (-1 vertical drop)

Using the grid coordinate system, I add in the barriers that cause crashing, and I add a special section for handling the final braking sequence.

## Learning to land a plane
The models start by taking actions at random. They continue doing so until they either crash or land. If they crash, for each action they took at each coordinate along the way, they mark the overall outcome of that run as either "crashed" or "landed". This continues for hundreds of thousands of iterations to build up a training dataset, on which they train, and learn how each action they took at each coordinate, given the choice they had with the dice roll (again, fate still has a big say here) affected the end result. Now that they have been trained on what they've previously done, they will go through the landing sequence again, but this time at each phases during the dice rolling, they will consider
1. What coordinate am I currently at?
2. What dice rolls do I have to work with this round?
3. What actions are available to me?
4. (Pilot only) what pilot action is available to me?

They will use _predict_proba_ to attempt to predict which action (and pilot action, if available) results in the highest chance of a successful landing, then take that action. This process repeats itself in "training cycles", each cycle consisting of 100,000 landing attempts.

## Results
After training each model on **2 million landing attempts**, each model was evaluated by having it attempt to use its knowledge to land the plane **10,000 times**.

The results are as follows:

| Model | Landing Chance |
| ----- | -------------- |
| Regular | 22.4% |
| Pilot | **89.7%** |

The Pilot Licence, when given to an MLP model, **quadruples** the chance to successfully land the plane! The chance of landing goes from "infrequently" to "almost always".

Given these results, it seems that the Pilot Licence, if you know what to do with it, is incredibly powerful. 

So the question is, what exactly is this model doing with the Pilot Licence that is drastically improving its success rates? We can answer this question by iterating through every possible position (coordinate pair) that the plane could be in its descent, as well as every possible dice result that could be rolled. We call each one of these combinations a "state" that the player may find themselves in. Going through every model state, we get responses from the model to build up an overall decision matrix that answers: "At coordinate (x,y) with dice roll (i,j), which Pilot Licence action should I take?". These answers allow us to see the percentage probability (%) of the model taking any given Pilot Licence action (or none) for any given state. The results are as follows:

![base](/media/2.svg)

Note that there are 1470 different possible states, since we exclude the special braking roll when on the ground.

| Pilot Licence Action | Chance to Take Action|
| ----- | -------------- |
| ↑ (+Y) | 58% |
| ← (-X) | 15% |
| None | 27% |

The results indicate that overall the Pilot Licence action to reduce vertical fall (+Y) is used far more frequently. In fact, it is almost twice as likely to take no Pilot Licence action at all (27%) than it is to take the action to reduce horizontal travel (-X at 15%), wheras the action to reduce vertical fall (+Y) is used more often than not (58%).

So really, the strength of the Pilot Licence lies mostly in its ability to reduce vertical fall. The reason that is the case is rather simple. On average, over millions of dice rolls, the plane would, without any other external forces, tend to travel down and to the right at an equal rate, i.e, for every space it falls, it would travel 1 space to the right. Again, this is a trend, on average. However, this path doesn't actually get us on to the landing strip! The ideal path would be to land on the very left-most tip of the landing strip to maximise our braking chance. So, what we need to improve the path trend is to push it slightly more to the right. The Pilot Licence cannot directly achieve that, since it does not have a "glide more" (+X) action allowed, but it can get close to this by doing the opposite, reducing the vertical fall. This action shifts our path trend closer to the ideal path.

![base](/media/3.svg)

