import asyncio
import numpy as np
import websockets
import pickle
from scipy import stats


rng = np.random.seed(2)

# some variables important for playing around
alpha = 1e-2        # learning coefficient (how much we change our weights) if too big the solution will usually diverge
gamma = 0.99         # coefficient for discounting reward (bigger coefficient - bigger long-term reward)
decay_rate = 0.9    # while learning we want to use several summed delta w's to get more stable learning. This shows how fast old delta w's decay (loose importance). It will be used while applying rmsprop
batch_size = 5      # we generally want to use dw's from several games for more stable learning. Difference with previous is that previous applies on a stage of changing weights and this one kind of before
episode = 0         # not really to change. It shows how many games we already played
toLearn = True
to_load = True
saveFrequency = 50


def xavier_init(next, prev) -> np.array:
    # Short explanation: With this weights initialization "vanishing or exploding gradients problem" is not as big
    # Normal explanation: https://ai.stackexchange.com/questions/21531/what-is-the-intuition-behind-the-xavier-initialization-for-deep-neural-networks
    # Formula: https://paperswithcode.com/method/xavier-initialization
    border = np.sqrt(6)/(np.sqrt(prev + next))
    w = np.random.uniform(-border, border, size = (prev, next))
    return w

I = 4       # size of input layers
h1 = 200    # size of first hidden layer
h2 = 100    # size of second hidden layer
h3 = 50
h4 = 14
h5 = 5
O = 2       # size of output layer

weights = {"W1":xavier_init(I, h1),
           "W2":xavier_init(h1, h2),
           "W3":xavier_init(h2, h3),
           "W4":xavier_init(h3, h4),
           "W5":xavier_init(h4, h5),
           "W6":xavier_init(h5, O)
           }

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)


def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

try:
    num = np.load("num.npy")
except:
    num = 0

if to_load:
    print(f"loading: model{num-1}")
    weights = load_dict(f'model{num-1}')



def sigmoid(a):                 # Normally I use ReLU (on hidden layers), but it is handy to have something more dual for the final result (cuz there are 2 options for y and 2 for x movement)
    return 1.0/(1.0+np.exp(-a))


def forward_propagation(weights, input):
    h1 = np.dot(weights["W1"], input)
    h1[h1 < 0] *= 0.1                   # leaky ReLU function
    h2 = np.dot(weights["W2"], h1)
    h2[h2 < 0] *= 0.1
    h3 = np.dot(weights["W3"], h2)
    h3[h3 < 0] *= 0.1
    h4 = np.dot(weights["W4"], h3)
    h4[h4 < 0] *= 0.1
    h5 = np.dot(weights["W5"], h4)
    h5[h5 < 0] *= 0.1
    output = np.dot(weights["W6"], h5)
    output = sigmoid(output)
    return output, [h1, h2, h3, h4, h5]         # we will need not only results, but also hidden states later on



def backward_propagation(I, h1, h2, h3, h4, h5, grad, weights):
    # Explanation for this: https://ml-cheatsheet.readthedocs.io/en/latest/backpropagation.html
    dw6 = np.zeros_like(weights["W6"])
    dw5 = np.zeros_like(weights["W5"])
    dw4 = np.zeros_like(weights["W4"])
    dw3 = np.zeros_like(weights["W3"])
    dw2 = np.zeros_like(weights["W2"])
    dw1 = np.zeros_like(weights["W1"])
    for i in range(len(h1)):
        # As far as I understand grad (J gradients) replace the derivative(cost function)*derivative(activation fucntion). From backpropagation cheetsheet and gradient in stack exchange formula
        dw6 += np.dot(np.array([grad[i]]).T, [h5[i]])
        
        dh5 = np.dot(grad[i], weights["W6"])
        dh5[h5[i] < 0] *= 0.1
        dw5 += np.dot(np.array([dh5]).T, [h4[i]])

        dh4 = np.dot(dh5, weights["W5"])
        dh4[h4[i] < 0] *= 0.1
        dw4 += np.dot(np.array([dh4]).T, [h3[i]])

        dh3 = np.dot(dh4, weights["W4"])
        dh3[h3[i] < 0] *= 0.1                  # apply derivative of third hidden layer
        dw3 += np.dot(np.array([dh3]).T, [h2[i]])

        dh2 = np.dot(dh3, weights["W3"])
        dh2[h2[i] < 0] *= 0.1
        dw2 += np.dot(np.array([dh2]).T, [h1[i]])

        dh1 = np.dot(dh2, weights["W2"])
        dh1[h1[i] < 0] *= 0.1
        dw1 += np.dot(np.array([dh1]).T, [I[i]])


    return {"W1": dw1, "W2": dw2, "W3": dw3, "W4": dw4, "W5": dw5, "W6":dw6}


def discount_reward(rewards, gamma):
    # in this function we just want to prioratize long-term rewards over short-term ones
    # it is important if we want ai to have strategy and not only tactics
    # the idea is to go from the end. If there are many positive consequantial rewards running_reward will become bigger
    # the full explanation one can find page 54 (returns and episodes): http://incompleteideas.net/book/RLbook2020.pdf
    discounted_reward = np.zeros_like(rewards)
    running_reward = 0
    for i in range(1,len(rewards)):
        running_reward = running_reward * gamma + rewards[len(rewards)-i]
        discounted_reward[len(rewards) - i] = running_reward
    return discounted_reward

# some vars used here
ih = []                 # inputs history
lph = []      # log probabilities history
hsh1 = []               # first hidden layer hidden states history
hsh2 = []
hsh3 = []
hsh4 = []
hsh5 = []
rh = []                 # reward history
gradBuffer = {k:np.zeros_like(v) for k,v in weights.items()}      # for summing gradients between weight updates
rmspropCache = {k:np.zeros_like(v) for k,v in weights.items()}    # for updating weights with respect to previous experience
running_reward = None                                               # is needed just to show progress
reward_sum = 0                                                  # is also needed for showing progress only (reward sum over an episode)

def ai(inp):
    x = np.array(inp[0:4])            # out actual inputs
    gameOver = inp[4]       # we need it to evaluate reward
    isCoinGotten = inp[5]   # did we catch a coin on a previous step
    
    
    global ih, lph, hsh1, hsh2, hsh3, hsh4, hsh5, rh, gradBuffer, rmspropCache, running_reward, reward_sum, alpha, gamma, decay_rate, batch_size, episode, num

    descision, hidden_states = forward_propagation(weights, x)
    # a lot of further steps are based on this: https://math.stackexchange.com/questions/2845971/solving-for-policy-gradient-in-reinforcement-learning
    # notation used can be found on a digital page 19 of this (very good material for reinforcement learning by the way) http://incompleteideas.net/book/RLbook2020.pdf (I even think, that it is actually free and not piracy)

    #action1 = 1 if descision[0] > np.random.uniform(0.25, 0.75) else -1       # we want some definite action. Not only that, we want to "explore" to not stuck in local minima (so I have random choice with different probabilities for different modell certainty levels)
    #action2 = 1 if descision[1] > np.random.uniform(0.25, 0.75) else -1 
    
    action1 = 1 if descision[0] > stats.truncnorm.rvs(-1, 1,loc = 0.5, scale = 0.5, size = 1) else -1 
    action2 = 1 if descision[1] > stats.truncnorm.rvs(-1, 1,loc = 0.5, scale = 0.5, size = 1)  else -1

    y1 = 1 if action1 == 1 else 0
    y2 = 1 if action2 == 1 else 0
   
    # technically these aren't log probabilities. They are gradients of cross-entropy with respect to output taking into account the fact that descision is processed by sigmoid: https://www.pinecone.io/learn/cross-entropy-loss/
    # I'm too lazy to rewrite them now
    logprob = np.array([y1 - descision[0], y2 - descision[1]])

    lph.append(logprob)

    hsh1.append(hidden_states[0])
    hsh2.append(hidden_states[1])
    hsh3.append(hidden_states[2])
    hsh4.append(hidden_states[3])
    hsh5.append(hidden_states[4])
    
    # now I have to evaluate, how good neural network did. 
    
    reward = 0.0

    if gameOver:
        reward += -20.0
    elif isCoinGotten:
        reward += 30.0 
    else:
        reward += -0.1
    

    reward_sum += reward
    ih.append(x) 
    rh.append(reward)


    if not gameOver:
        return " ".join([str(action1), str(action2)])
    elif toLearn:
        episode += 1
        
        # here I convert arrays into np arrays. I've seen it being done with vstack, so I do so. I don't think there is much of a difference with np.array, except I will have to account for different sizes
        hsh1 = np.vstack(hsh1)
        hsh2 = np.vstack(hsh2)
        hsh3 = np.vstack(hsh3)
        hsh4 = np.vstack(hsh4)
        hsh5 = np.vstack(hsh5)
        lph = np.vstack(lph)
        rh = np.vstack(rh)
        ih = np.vstack(ih)

        discounted_rewards = discount_reward(rh, gamma)
        
        # it is generally good idea to normalize values in deep learning, because big and small ones can cause gradients to explode of to vanish
        # I did it however because I've seen it done earlier and I'm not sure if previous explanation is good enough

        #discounted_rewards -= np.mean(discounted_rewards)   # mean is just normal average, so we make values be roughtly equally spread under and above zero
        discounted_rewards /= np.std(discounted_rewards)    # np.std is standart deviation. Such operation is called standartizing values, but I'm not sure whether it is relevant here

        grad = np.multiply(discounted_rewards, lph)                 # gradients of the whole model

        g = backward_propagation(ih, hsh1, hsh2, hsh3, hsh4, hsh5, grad, weights)   # gradients of individual weights
        for k, v in g.items():
            gradBuffer[k] += g[k]       

        if episode % batch_size == 0:
            print("New Iteration", descision)
            # Explanation/formula: https://ml-cheatsheet.readthedocs.io/en/latest/optimizers.html#rmsprop
            for k,v in weights.items():
                # Explanation on given webpage. Difference is plus instead of minus because this is gradient accent and not decent
                rmspropCache[k] = rmspropCache[k]*decay_rate + (1-decay_rate)*gradBuffer[k]**2   # Formula
                weights[k] += alpha * (gradBuffer[k]/(np.sqrt(rmspropCache[k]) + 1e-8))
            
                gradBuffer[k] = np.zeros_like(v)
        if running_reward != None:
            running_reward = 0.99*running_reward + 0.01 * reward_sum
        else:
            running_reward = reward_sum
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
        hsh4 = list()
        hsh5 = list()
        rh = list()
    return "0 0"

# Fremdquellcode
# Quelle: https://www.youtube.com/watch?v=Bx0knQcKoqI

async def server(ws, path):
    async for msg in ws:
        response = ai(list(map(float, msg.split(" "))))
        await ws.send(response)


start_server = websockets.serve(server, "localhost", 5000)
#print(ai([1, 1, 1, 1, 1,1]))
print("server started")
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
