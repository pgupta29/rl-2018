import numpy as np


def bandit(q_stars, As, rew_sigma):
    runs = As.shape[0]
    norm_samples = np.random.normal(loc=0, scale=rew_sigma, size=(runs))
    R_center = q_stars[As, np.arange(runs)]
    return norm_samples + R_center
    

def simple_eps_greedy_bandit(k, q_stars, timesteps, eps=0, sigma=1):
    def bandit(q_stars, A, sigma=sigma):
        return np.random.normal(loc=q_stars[A], scale=sigma)

    Q = np.ones(k) # current value estimate
    N = np.zeros(k) # num each action reward seen
    As = np.zeros(timesteps) # all actions
    Rs = np.zeros(timesteps) # all rewards
    
    for t in range(timesteps):
        if np.random.uniform(0,1) < eps:
            curr_A = np.random.randint(0, k)
        else:
            curr_A = np.random.choice(np.flatnonzero(Q == Q.max())) # breaks ties randomly
            
        curr_R = bandit(q_stars, curr_A)
        N[curr_A] += 1
        Q[curr_A] += (1/N[curr_A]) * (curr_R - Q[curr_A])
        As[t] = curr_A
        Rs[t] = curr_R
    return Q, N, As, Rs

#-------------------------------------------------------------------#
#-------------------------------------------------------------------#
#-------------------------------------------------------------------#

# more useful function
# modified to take 3 dim q_stars (timesteps, k, runs)
def eps_greedy_bandit(k, runs, q_stars, timesteps, eps=0, rew_sigma=1, dyn_rew_fn=None, alpha=None, q_init=None):
    
    As = np.zeros(runs, dtype=int)
    if q_init is None:
        curr_Qs_est = np.zeros((k, runs)) # current action value estimate
    else:
        curr_Qs_est = q_init
    avg_Qs = np.zeros((k, timesteps))
    Ns = np.zeros((k, runs))

    avg_Rs = np.zeros((timesteps))
    opt_act_frac = np.zeros((timesteps))
    opt_act_count = np.zeros((timesteps))
    
    if dyn_rew_fn is None:
        def dyn_rew_fn(q_stars, t):
            return q_stars

    curr_q_stars_true = q_stars
    for t in range(timesteps):
        curr_q_stars_true = dyn_rew_fn(curr_q_stars_true, t)
        
        eps_sample = (np.random.uniform(0, 1, runs) < eps).astype(int) # sampling eps 
        eps_indx = np.where(eps_sample == 1)[0]  # index of bandits with random action
        greedy_indx = np.where(eps_sample == 0)[0] # index of bandits with greedy action

        eps_As = np.random.randint(0, k, eps_indx.shape[0]) # picking eps (random) action
        greedy_As = np.argmax(curr_Qs_est[:, greedy_indx], axis=0) # pick greedy actions
        As[eps_indx] = eps_As # insert eps actions
        As[greedy_indx] = greedy_As  # insert greedy actions

        optimal_actions = np.argmax(curr_q_stars_true, axis=0)
        opt_act_count[t] = np.count_nonzero(As == optimal_actions)
        
        Rs = bandit(curr_q_stars_true, As, rew_sigma) # takes a while if a lot of steps/numbers)
        avg_Rs[t] = np.mean(Rs)

        if alpha is None:
            np.add.at(Ns, (As, np.arange(runs)), 1)  # updates Ns for specific As
            update = (Rs - curr_Qs_est[As, np.arange(runs)]) / Ns[As, np.arange(runs)]
        else:
            update = (Rs - curr_Qs_est[As, np.arange(runs)]) * np.full((runs), alpha)
        oldvals = curr_Qs_est[As, np.arange(runs)]
        newvals = oldvals + update 
        curr_Qs_est[As, np.arange(runs)] = newvals
        
    
    return avg_Rs, opt_act_count, curr_Qs_est

#-------------------------------------------------------------------#
#-------------------------------------------------------------------#
#-------------------------------------------------------------------#


def ucb_greedy_bandit(k, runs, q_stars, timesteps, ucb_c, rew_sigma=1, dyn_rew_fn=None, alpha=None, q_init=None):
    As = np.zeros(runs, dtype=int)
    if q_init is None:
        curr_Qs_est = np.zeros((k, runs)) # current action value estimate
    else:
        curr_Qs_est = q_init
    avg_Qs = np.zeros((k, timesteps))
    Ns = np.zeros((k, runs))

    avg_Rs = np.zeros((timesteps))
    opt_act_frac = np.zeros((timesteps))
    opt_act_count = np.zeros((timesteps))
    
    if dyn_rew_fn is None:
        def dyn_rew_fn(q_stars, t):
            return q_stars

    curr_q_stars_true = q_stars
    for t in range(timesteps):
        curr_q_stars_true = dyn_rew_fn(curr_q_stars_true, t)
        
        if t < k:  # if timesteps less than avaialable options, pick options. all pick sequentially
            As = (np.ones(runs) * t).astype(np.int)
        else:
            inv_Ns = 1/Ns
            curr_As_select = curr_Qs_est + (ucb_c * np.sqrt( (np.log(t)) * inv_Ns))
            As = np.argmax(curr_As_select, axis=0).astype(np.int)
        
        optimal_actions = np.argmax(curr_q_stars_true, axis=0)
        opt_act_count[t] = np.count_nonzero(As == optimal_actions)
        
        Rs = bandit(curr_q_stars_true, As, rew_sigma) # takes a while if a lot of steps/numbers)
        avg_Rs[t] = np.mean(Rs)

        np.add.at(Ns, (As, np.arange(runs)), 1)  # updates Ns for specific As
        update = (Rs - curr_Qs_est[As, np.arange(runs)]) / Ns[As, np.arange(runs)]

        oldvals = curr_Qs_est[As, np.arange(runs)]
        newvals = oldvals + update 
        curr_Qs_est[As, np.arange(runs)] = newvals
        
    
    return avg_Rs, opt_act_count, curr_Qs_est


def gradient_bandit(k, runs, q_stars, timesteps, alpha, h_init=None):

    if h_init is None:
        Hs = np.zeros((k, runs))
    else:
        Hs = h_init
    Ns = np.zeros((k,runs))
    R_avg = np.zeros((k, runs))
    avg_Rs = np.zeros((timesteps))
    opt_act_count = np.zeros((timesteps))

    for t in range(timesteps):
        pis = np.exp(Hs) / np.sum(np.exp(Hs), axis=0) # softmax for choosing actions
        rand = np.random.uniform(0, 1, size=(runs))  # inverse sampling
        As = np.argmax(np.cumsum(pis, axis=0) > rand, axis=0)  # argmax returns first true sample

        # get rewards given actions
        Rs = bandit(q_stars, As, rew_sigma=1)

        # updating R_avg
        np.add.at(Ns, (As, np.arange(runs)), 1)  # updates Ns for specific As
        update = (Rs - R_avg[As, np.arange(runs)]) / Ns[As, np.arange(runs)]
        oldvals = R_avg[As, np.arange(runs)]
        newvals = oldvals + update 
        R_avg[As, np.arange(runs)] = newvals   # updating R_avg for each run

        # updating preferences for actions chosen (runs actions)
        Hs[As, np.arange(runs)] += alpha * (Rs - R_avg[As, np.arange(runs)]) * (1 - pis[As, np.arange(runs)])

        # updating preferences for actions not chosen ((k-1) * runs)
        non_A_update = -alpha * (Rs - R_avg) * (pis)
        non_A_update[As, np.arange(runs)] = 0
        Hs += non_A_update

        # calculating things
        optimal_actions = np.argmax(q_stars, axis=0)
        opt_act_count[t] = np.count_nonzero(As == optimal_actions)
        avg_Rs[t] = np.mean(Rs)

    return opt_act_count, avg_Rs