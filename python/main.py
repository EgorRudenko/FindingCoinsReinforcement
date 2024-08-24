import asyncio
import numpy as np
import websockets
import pickle
from scipy import stats

rng = np.random.seed(2)

# some variables important for playing around
alpha = 1e-2  # learning coefficient (how much we change our weights) if too big the solution will usually diverge
gamma = 0.99  # coefficient for discounting reward (bigger coefficient - bigger long-term reward)
decay_rate = 0.9  # while learning we want to use several summed delta w's to get more stable learning. This shows how fast old delta w's decay (loose importance). It will be used while applying rmsprop
batch_size = 30  # we generally want to use dw's from several games for more stable learning. Difference with previous is that previous applies on a stage of changing weights and this one kind of before
episode = 0  # not really to change. It shows how many games we already played
toLearn = True
to_load = True
saveFrequency = 50
modelToLoad = None  # 16 is the first version which isn't fully shit


def xavier_init(next, prev) -> np.array:
    # Short explanation: With this weights initialization "vanishing or exploding gradients problem" is not as big
    # Normal explanation: https://ai.stackexchange.com/questions/21531/what-is-the-intuition-behind-the-xavier-initialization-for-deep-neural-networks
    # Formula: https://paperswithcode.com/method/xavier-initialization
    border = np.sqrt(6) / (np.sqrt(prev + next))
    w = np.random.uniform(-border, border, size=(prev, next))
    return w


I = 6  # size of input layers
h1 = 20  # size of first hidden layer
h2 = 10  # size of second hidden layer
h3 = 4  # size of the third hidden layer
O = 2  # size of output layer

weights = {"W1": xavier_init(I, h1),
           "W2": xavier_init(h1, h2),
           "W3": xavier_init(h2, h3),
           "W4": xavier_init(h3, O)
           }


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)


def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di


if modelToLoad != None:
    num = modelToLoad + 1
else:
    try:
        num = np.load("num.npy")
    except:
        num = 0

if to_load:
    print(f'model{num - 1}')
    weights = load_dict(f'model{num - 1}')


def sigmoid(a):  # Normally I use ReLU (on hidden layers), but it is handy to have something more dual for the final result (because there are 2 options for y and 2 for x movement)
    return 1.0 / (1.0 + np.exp(-a))


def forward_propagation(weights, input):
    h1 = np.dot(weights["W1"], input)
    h1[h1 < 0] *= 0.1  # leaky ReLU function
    h2 = np.dot(weights["W2"], h1)
    h2[h2 < 0] *= 0.1
    h3 = np.dot(weights["W3"], h2)
    h3[h3 < 0] *= 0.1
    output = np.dot(weights["W4"], h3)
    output = sigmoid(output)
    return output, [h1, h2, h3]  # we will need not only results, but also hidden states later on


def backward_propagation(I, h1, h2, h3, grad, weights):
    # Explanation for this: https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html
    dw4 = np.zeros_like(weights["W4"])
    dw3 = np.zeros_like(weights["W3"])
    dw2 = np.zeros_like(weights["W2"])
    dw1 = np.zeros_like(weights["W1"])
    for i in range(len(h1)):
        dw4 += np.dot(np.array([grad[i]]).T, [h3[i]])

        dh3 = np.dot(grad[i], weights["W4"])
        dh3[h3[i] < 0] *= 0.1  # apply derivative of third hidden layer
        dw3 += np.dot(np.array([dh3]).T, [h2[i]])

        dh2 = np.dot(dh3, weights["W3"])
        dh2[h2[i] < 0] *= 0.1
        dw2 += np.dot(np.array([dh2]).T, [h1[i]])

        dh1 = np.dot(dh2, weights["W2"])
        dh1[h1[i] < 0] *= 0.1
        dw1 += np.dot(np.array([dh1]).T, [I[i]])

    return {"W1": dw1, "W2": dw2, "W3": dw3, "W4": dw4}


def discount_reward(rewards, gamma):
    # in this function we just want to prioritize long-term rewards over short-term ones
    # it is important if we want AI to have strategy and not only tactics
    # the idea is to go from the end. If there are many positive consequential rewards running_reward will become bigger
    # the more detailed explanation of the similar concept one can find on page 54 (returns and episodes): http://incompleteideas.net/book/RLbook2020.pdf
    discounted_reward = np.zeros_like(rewards)
    running_reward = 0
    for i in range(1, len(rewards)):
        running_reward = running_reward * gamma + rewards[len(rewards) - i]
        discounted_reward[len(rewards) - i] = running_reward
    return discounted_reward


# some vars used here
ih = []  # inputs history
lph = []  # log probabilities history
hsh1 = []  # first hidden layer hidden states history
hsh2 = []
hsh3 = []
rh = []  # reward history
gradBuffer = {k: np.zeros_like(v) for k, v in weights.items()}  # for summing gradients between weight updates
rmspropCache = {k: np.zeros_like(v) for k, v in weights.items()}  # for updating weights with respect to previous experience
running_reward = None  # is needed just to show progress
reward_sum = 0  # is also needed for showing progress only (reward sum over an episode)


def ai(inp):
    x = np.array(inp[0:6])  # out actual inputs
    gameOver = inp[6]  # we need it to decide wether we should do backpropagation
    isCoinGotten = inp[7]  # tells us wether we catch a coin on a previous step
    isBorderTouched = inp[8]    # tells us wether the border is touched. Difference from gameover is that "time death" isn't accounted for here

    global ih, lph, hsh1, hsh2, hsh3, rh, gradBuffer, rmspropCache, running_reward, reward_sum, alpha, gamma, decay_rate, batch_size, episode, num

    decision, hidden_states = forward_propagation(weights, x)

    # we want some definite action. Not only that,
    # we want to "explore" to not stuck in local minima (so I have random choice with different probabilities for
    # different model certainty levels) the fitst now commented section gives a bit more exploration, the second which is
    # actually used has more exploitation which is defined by truncated normal distribution

    #action1 = 1 if decision[0] > np.random.uniform(0,1) else -1  
    #action2 = 1 if decision[1] > np.random.uniform(0, 1) else -1
    
    action1 = 1 if decision[0] > stats.truncnorm.rvs(-1, 1,loc = 0.5, scale = 0.5, size = 1) else -1 
    action2 = 1 if decision[1] > stats.truncnorm.rvs(-1, 1,loc = 0.5, scale = 0.5, size = 1)  else -1

    y1 = 1 if action1 == 1 else 0
    y2 = 1 if action2 == 1 else 0

    # technically the following aren't log probabilities. They are derivatives of cross-entropy with respect to output taking
    # into account the fact that decision is processed by sigmoid: https://www.pinecone.io/learn/cross-entropy-loss/
    # I'm too lazy to rewrite their names now
    logprob = np.array([y1 - decision[0], y2 - decision[1]])

    lph.append(logprob)

    hsh1.append(hidden_states[0])
    hsh2.append(hidden_states[1])
    hsh3.append(hidden_states[2])

    # now I have to evaluate, how good neural network did. And give reward accordingly

    reward = 0.0

    if isCoinGotten:
        reward += 20.0
    elif isBorderTouched:
        reward += -10.0
    else:
        reward += -0.01          
        # I don't like it, but without it the network diverges on the first weights update
        # it is supposed to motivate network to do anything in labirinth like tasks
        # this program isn't exactly one, but as I said it has to be this way

    reward_sum += reward
    ih.append(x)
    rh.append(reward)

    if not gameOver:
        return " ".join([str(action1), str(action2)])   # send the dicision made by network
    elif toLearn:
        episode += 1

        # here I convert arrays into np arrays. The goal is to make some calculations easier and dimensions more defined
        hsh1 = np.vstack(hsh1)
        hsh2 = np.vstack(hsh2)
        hsh3 = np.vstack(hsh3)
        lph = np.vstack(lph)
        ih = np.vstack(ih)
        rh = np.vstack(rh)

        discounted_rewards = discount_reward(rh, gamma)

        # it is generally good idea to normalize values in deep learning, because big and small ones can cause
        # gradients to explode of to vanish I did it however because I've seen it done earlier and I'm not sure if
        # previous explanation is good enough

        discounted_rewards -= np.mean(discounted_rewards)  # np.mean is just normal average, so we make values be
        # roughly equally spread under and above zero + bad, but not as bad as all others actions are made good now
        discounted_rewards /= np.std(discounted_rewards)  # np.std is standard deviation. Such operation is called
        # standartizing values, but I'm not sure whether it is relevant here

        grad = np.multiply(discounted_rewards, lph)     # evaluation of network results with "direction" difined by derivative of cross-entropy
                                                        # so if network didn't want to do something but did because of exploration it will be accounted for

        g = backward_propagation(ih, hsh1, hsh2, hsh3, grad, weights)  # (approximate???) patial derivatives of individual weights
        for k in weights:
            gradBuffer[k] += g[k]
        if episode % batch_size == 0:
            print("New Iteration")
            # Explanation/formula for rmsprop: https://ml-cheatsheet.readthedocs.io/en/latest/optimizers.html#rmsprop
            for k, v in weights.items():
                # Explanation on given webpage. Difference is plus instead of minus because this is gradient accent
                # and not gradient decent
                rmspropCache[k] = rmspropCache[k] * decay_rate + (1 - decay_rate) * gradBuffer[k] ** 2  # Formula
                weights[k] += alpha * (gradBuffer[k] / (np.sqrt(rmspropCache[k]) + 1e-8))

                gradBuffer[k] = np.zeros_like(v)
        if running_reward != None:
            running_reward = 0.99 * running_reward + 0.01 * reward_sum
        else:
            running_reward = 10.0  # Because I want to see weather it grows or just average becomes closer to real
            # value if first reward is too negative
        print(f"Episode: {episode}      Reward in this episode: {reward_sum}        Approximate average current level reward: {running_reward}")

        if episode % saveFrequency == 0:
            save_dict(weights, f'model{num}')
            num += 1
            np.save('num', num)

        reward_sum = 0
        ih = list()
        lph = list()
        hsh1 = list()
        hsh2 = list()
        hsh3 = list()
        rh = list()
    return "0 0"


# Fremdquellcode
# Quelle: https://www.youtube.com/watch?v=Bx0knQcKoqI

async def server(ws, path):
    async for msg in ws:
        response = ai(list(map(float, msg.split(" "))))
        await ws.send(response)


start_server = websockets.serve(server, "localhost", 5000)
print("server started")
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
