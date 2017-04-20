import numpy as np
import tensorflow as tf
import gym
import logz
import scipy.signal

def clear_dir(path):
    import os, shutil
    if not os.path.isdir(path):
        return
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

def normc_initializer(std=1.0):
    """
    Initialize array with normalized columns
    """
    def _initializer(shape, dtype=None, partition_info=None): #pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def dense(x, size, name, weight_init=None):
    """
    Dense (fully connected) layer
    """
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
    b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer())
    return tf.matmul(x, w) + b

def fancy_slice_2d(X, inds0, inds1):
    """
    Like numpy's X[inds0, inds1]
    """
    inds0 = tf.cast(inds0, tf.int64)
    inds1 = tf.cast(inds1, tf.int64)
    shape = tf.cast(tf.shape(X), tf.int64)
    ncols = shape[1]
    Xflat = tf.reshape(X, [-1])
    return tf.gather(Xflat, inds0 * ncols + inds1)

def discount(x, gamma):
    """
    Compute discounted sum of future values
    out[i] = in[i] + gamma * in[i+1] + gamma^2 * in[i+2] + ...
    """
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]

def explained_variance_1d(ypred,y):
    """
    Var[ypred - y] / var[y]. 
    https://www.quora.com/What-is-the-meaning-proportion-of-variance-explained-in-linear-regression
    """
    assert y.ndim == 1 and ypred.ndim == 1    
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

def categorical_sample_logits(logits):
    """
    Samples (symbolically) from categorical distribution, where logits is a NxK
    matrix specifying N categorical distributions with K categories

    specifically, exp(logits) / sum( exp(logits), axis=1 ) is the 
    probabilities of the different classes

    Cleverly uses gumbell trick, based on
    https://github.com/tensorflow/tensorflow/issues/456
    """
    U = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(U)), dimension=1)

def pathlength(path):
    return len(path["reward"])

class LinearValueFunction(object):
    coef = None
    def fit(self, X, y):
        Xp = self.preproc(X)
        A = Xp.T.dot(Xp)
        nfeats = Xp.shape[1]
        A[np.arange(nfeats), np.arange(nfeats)] += 1e-3 # a little ridge regression
        b = Xp.T.dot(y)
        self.coef = np.linalg.solve(A, b)
        print ('|||||||linear_value_loss {}'.format(np.square(y - self.predict(X)).mean()))
    def predict(self, X):
        if self.coef is None:
            return np.zeros(X.shape[0])
        else:
            return self.preproc(X).dot(self.coef)
    def preproc(self, X):
        return np.concatenate([np.ones([X.shape[0], 1]), X, np.square(X)/2.0], axis=1)

class NnValueFunction(object):
    # YOUR CODE HERE
    def __init__(self, ob_dim=3, hidden_dims=[16], n_epochs=10, stepsize=0.001):
        self.init = False
        self.hidden_dims = hidden_dims
        self.n_epochs = n_epochs
        self.stepsize = stepsize
        
        with tf.name_scope('nn_value'):
            self.sy_X = tf.placeholder(tf.float32, [None, ob_dim * 2], 'X')
            self.sy_y = tf.placeholder(tf.float32, [None], 'y')
            self.sy_stepsize = tf.placeholder(tf.float32, [], 'nn_value_stepsize')
            
            with tf.name_scope('network'):  
                sy_h = self.sy_X
                for layer_i, dim in enumerate(self.hidden_dims):
                    sy_h =lrelu(dense(sy_h, dim, 'nn_value/'+str(layer_i), weight_init=normc_initializer(1.0)))
                self.sy_out = dense(sy_h, 1, 'nn_value/output', weight_init=normc_initializer(0.05))
                
            with tf.name_scope('square_loss'):
                self.sy_loss = tf.squared_difference(self.sy_out, self.sy_y)

            self.update = tf.train.AdamOptimizer(self.sy_stepsize).minimize(self.sy_loss)
    
            with tf.name_scope('init'):
                variables = tf.trainable_variables()
                variables = list(filter(lambda x: x.name.startswith('nn_value'), variables))
                self.sy_reinit = tf.variables_initializer(variables)
                
            
    def bond_sess(self, sess):
        self.sess = sess
        
    def _sample(self, X, y, batch_size):
        n = X.shape[0]
        idx = np.random.randint(0, n-1, size=batch_size)
        return X[idx], y[idx]
    
    def fit(self, X, y, on_batch=True):
        assert self.sess, 'Please invoke bond_sess first'
        self.sess.run(self.sy_reinit)
        Xp = self.preproc(X)
        for e in range(self.n_epochs):
            stepsize = self.stepsize
            if on_batch:
                for i in range(round(X.shape[0] / 32)):
                    X_batch, y_batch = self._sample(Xp, y, 32)
                    _, loss = self.sess.run([self.update, self.sy_loss], feed_dict={self.sy_X: X_batch, self.sy_y: y_batch, self.sy_stepsize: stepsize})
            else:
                _, loss = self.sess.run([self.update, self.sy_loss], feed_dict={self.sy_X: self.preproc(X), self.sy_y: y, self.sy_stepsize: stepsize})
            if e%10 == 9:
                print ('|||||nn value loss {}'.format(np.square(y - self.predict(X)).mean()))
            
    def predict(self, X):
        assert self.sess, 'Please invoke bond_sess first'
        y = self.sess.run(self.sy_out, feed_dict={self.sy_X: self.preproc(X)})
        return y.flatten()
            
    def preproc(self, X):
        return np.concatenate([X, np.square(X)/2.0], axis=1)
    
def lrelu(x, leak=0.2):
    with tf.name_scope('leak_relu'):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)



def main_cartpole(n_iter=100, gamma=1.0, min_timesteps_per_batch=1000, stepsize=1e-2, vf_type='linear', vf_params={}, animate=True, logdir=None):
    env = gym.make("CartPole-v0")
    ob_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    logz.configure_output_dir(logdir)
    if vf_type == 'linear':
        vf = LinearValueFunction(**vf_params)
    elif vf_type == 'nn':
        vf = NnValueFunction(ob_dim=ob_dim, **vf_params)

    # Symbolic variables have the prefix sy_, to distinguish them from the numerical values
    # that are computed later in these function
    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32) # batch of observations
    sy_ac_n = tf.placeholder(shape=[None], name="ac", dtype=tf.int32) # batch of actions taken by the policy, used for policy gradient computation
    sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32) # advantage function estimate
    #### define neural network with 1 hidden layer
    sy_h1 = lrelu(dense(sy_ob_no, 32, "h1", weight_init=normc_initializer(1.0))) # hidden layer
    sy_logits_na = dense(sy_h1, num_actions, "final", weight_init=normc_initializer(0.05)) # "logits", describing probability distribution of final layer
    # we use a small initialization for the last layer, so the initial policy has maximal entropy
    sy_oldlogits_na = tf.placeholder(shape=[None, num_actions], name='oldlogits', dtype=tf.float32) # logits BEFORE update (just used for KL diagnostic)
    sy_logp_na = tf.nn.log_softmax(sy_logits_na) # logprobability of actions
    sy_sampled_ac = categorical_sample_logits(sy_logits_na)[0] # sampled actions, used for defining the policy (NOT computing the policy gradient)
    sy_n = tf.shape(sy_ob_no)[0]
    sy_logprob_n = fancy_slice_2d(sy_logp_na, tf.range(sy_n), sy_ac_n) # log-prob of actions taken -- used for policy gradient calculation

    # The following quantities are just used for computing KL and entropy, JUST FOR DIAGNOSTIC PURPOSES >>>>
    sy_oldlogp_na = tf.nn.log_softmax(sy_oldlogits_na)
    sy_oldp_na = tf.exp(sy_oldlogp_na) 
    sy_kl = tf.reduce_sum(sy_oldp_na * (sy_oldlogp_na - sy_logp_na)) / tf.to_float(sy_n)
    sy_p_na = tf.exp(sy_logp_na)
    sy_ent = tf.reduce_sum( - sy_p_na * sy_logp_na) / tf.to_float(sy_n)
    # <<<<<<<<<<<<<

    sy_surr = - tf.reduce_mean(sy_adv_n * sy_logprob_n) # Loss function that we'll differentiate to get the policy gradient ("surr" is for "surrogate loss")

    sy_stepsize = tf.placeholder(shape=[], dtype=tf.float32) # Symbolic, in case you want to change the stepsize during optimization. (We're not doing that currently)
    update_op = tf.train.AdamOptimizer(sy_stepsize).minimize(sy_surr)

    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1) 
    # use single thread. on such a small problem, multithreading gives you a slowdown
    # this way, we can better use multiple cores for different experiments
    sess = tf.Session(config=tf_config)
    sess.__enter__() # equivalent to `with sess:`
    tf.global_variables_initializer().run() #pylint: disable=E1101

    if vf_type == 'nn':
        vf.bond_sess(sess)

    total_timesteps = 0

    for i in range(n_iter):
        print("********** Iteration %i ************"%i)

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            terminated = False
            obs, acs, rewards = [], [], []
            animate_this_episode=(len(paths)==0 and (i % 10 == 0) and animate)
            while True:
                if animate_this_episode:
                    env.render()
                obs.append(ob)
                ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no : ob[None]})
                acs.append(ac)
                ob, rew, done, _ = env.step(ac)
                rewards.append(rew)
                if done:
                    break                    
            path = {"observation" : np.array(obs), "terminated" : terminated,
                    "reward" : np.array(rewards), "action" : np.array(acs)}
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch
        # Estimate advantage function
        vtargs, vpreds, advs = [], [], []
        for path in paths:
            rew_t = path["reward"]
            return_t = discount(rew_t, gamma)
            vpred_t = vf.predict(path["observation"])
            adv_t = return_t - vpred_t
            advs.append(adv_t)
            vtargs.append(return_t)
            vpreds.append(vpred_t)

        # Build arrays for policy update
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_n = np.concatenate([path["action"] for path in paths])
        adv_n = np.concatenate(advs)
        standardized_adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)
        vtarg_n = np.concatenate(vtargs)
        vpred_n = np.concatenate(vpreds)
        vf.fit(ob_no, vtarg_n)

        # Policy update
        _, oldlogits_na = sess.run([update_op, sy_logits_na], feed_dict={sy_ob_no:ob_no, sy_ac_n:ac_n, sy_adv_n:standardized_adv_n, sy_stepsize:stepsize})
        kl, ent = sess.run([sy_kl, sy_ent], feed_dict={sy_ob_no:ob_no, sy_oldlogits_na:oldlogits_na})

        # Log diagnostics
        logz.log_tabular("EpRewMean", np.mean([path["reward"].sum() for path in paths]))
        logz.log_tabular("EpLenMean", np.mean([pathlength(path) for path in paths]))
        logz.log_tabular("KLOldNew", kl)
        logz.log_tabular("Entropy", ent)
        logz.log_tabular("EVBefore", explained_variance_1d(vpred_n, vtarg_n))
        logz.log_tabular("EVAfter", explained_variance_1d(vf.predict(ob_no), vtarg_n))
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        # If you're overfitting, EVAfter will be way larger than EVBefore.
        # Note that we fit value function AFTER using it to compute the advantage function to avoid introducing bias
        logz.dump_tabular()

def main_pendulum(logdir, seed, n_iter, gamma, min_timesteps_per_batch, initial_stepsize, desired_kl, vf_type, vf_params, animate=False):
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env = gym.make("Pendulum-v0")
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]
    logz.configure_output_dir(logdir)
    if vf_type == 'linear':
        vf = LinearValueFunction(**vf_params)
    elif vf_type == 'nn':
        vf = NnValueFunction(ob_dim=ob_dim, **vf_params)


    ####YOUR_CODE_HERE####
    # Symbolic variables have the prefix sy_, to distinguish them from the numerical values
    # that are computed later in these function
    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32) # batch of observations
    sy_ac_n = tf.placeholder(shape=[None], name="ac", dtype=tf.float32) # batch of actions taken by the policy, used for policy gradient computation
    sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32) # advantage function estimate
    ######## define neural network with 2 hidden layer
    #### parameterize mean action
    num_dim_1 = 32
    with tf.name_scope('mean_action_network'):
        sy_h1 = lrelu(dense(sy_ob_no, num_dim_1, "h1", weight_init=normc_initializer(1.0))) # hidden layer 1
        # sy_h2 = lrelu(dense(sy_h1, num_dim_2, "h2", weight_init=normc_initializer(1.0)))
        mean_na = dense(sy_h1, ac_dim, "final", weight_init=normc_initializer(0.1)) # output layer
        tf.summary.histogram('mean_na', mean_na)
    #### parameterize action of logstd
    with tf.name_scope('std_action'):
        logstd_a = tf.get_variable('logstddev', [ac_dim], initializer=tf.constant_initializer(0.0)) # initialize as 0 for log dev
        tf.summary.histogram('logstd_a', logstd_a)
        tf.summary.scalar('logstd_a', tf.squeeze(logstd_a))
        std_a = tf.exp(logstd_a)
        tf.summary.histogram('std_a', std_a)
        tf.summary.scalar('std_a', tf.squeeze(std_a))
    ########
    #### choose an action
    with tf.name_scope('sample_action'):
        sy_distribution_ac = tf.contrib.distributions.Normal(mu=mean_na, sigma=std_a)
        # sy_sample_ac = sy_distribution_ac.sample()
        sy_sample_ac = tf.stop_gradient(sy_distribution_ac.sample()) # assert shape ?
    #### compute the logprob of action fed in sy_ac_n, do not use sampled above
    with tf.name_scope('action_logprob'):
        sy_distribution = tf.contrib.distributions.Normal(mu=tf.reshape(mean_na, [-1]), sigma=std_a)
        sy_prob = sy_distribution.pdf(sy_ac_n)
        sy_logprob_n = tf.log(sy_prob)
        sy_logprob_n = tf.reshape(sy_logprob_n, [-1])        
    #### kl divergence and entropy
    # define placeholder for old
    sy_mean_na_old = tf.placeholder(shape=[None, ac_dim], name='old_mean', dtype=tf.float32)
    sy_std_a_old = tf.placeholder(shape=[ac_dim], name='old_std', dtype=tf.float32)
    # kl divergence
    with tf.name_scope('kl_divergence'):
        sy_kl = tf.log(std_a / sy_std_a_old) + (tf.square(sy_std_a_old) + tf.square(sy_mean_na_old - mean_na)) / (2 * tf.square(std_a)) - 0.5
        sy_kl = tf.reduce_mean(sy_kl)
    # entropy
    with tf.name_scope('policy_entropy'):
        sy_ent = -0.5 * tf.log(2 * 3.141592653 * tf.square(std_a) + 1)
        sy_ent = tf.squeeze(sy_ent)
    ####YOUR_CODE_END####

    with tf.name_scope('surrogate_loss'):
        sy_surr = - tf.reduce_mean(sy_adv_n * sy_logprob_n) # Loss function that we'll differentiate to get the policy gradient ("surr" is for "surrogate loss")

    #### calcalate gradient, for visulization purprose
    with tf.name_scope('all_summary'):
        grads = tf.gradients(sy_surr, tf.trainable_variables())
        grads = list(zip(grads, tf.trainable_variables()))
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.name + '/gradient', grad)
        grads = list(filter(lambda x: x[0] is not None, grads))
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)

    sy_stepsize = tf.placeholder(shape=[], dtype=tf.float32) # Symbolic, in case you want to change the stepsize during optimization. (We're not doing that currently)
    update_op = tf.train.AdamOptimizer(sy_stepsize).minimize(sy_surr)

    sess = tf.Session()
    sess.__enter__() # equivalent to `with sess:`
    tf.global_variables_initializer().run() #pylint: disable=E1101
    if vf_type == 'nn':
        vf.bond_sess(sess)
    total_timesteps = 0
    stepsize = initial_stepsize
    merged = tf.summary.merge_all()
    clear_dir('log/train')
    file_writer = tf.summary.FileWriter('log/train', sess.graph)

    for i in range(n_iter):
        print("********** Iteration %i ************"%i)

        ####YOUR_CODE_HERE####
        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            terminated = False
            obs, acs, rewards = [], [], []
            animate_this_episode=(len(paths)==0 and (i % 10 == 0) and animate)
            while True:
                if animate_this_episode:
                    env.render()
                obs.append(ob)
                # assert ob.shape == (3,)
                ac = sess.run(sy_sample_ac, feed_dict={sy_ob_no : ob[None]})
                acs.append(ac.flatten())
                ob, rew, done, _ = env.step(ac)
                rew = rew.flatten()
                ob = ob.flatten()
                rewards.append(rew)
                if done:
                    break                    
            path = {"observation" : np.array(obs), "terminated" : terminated,
                    "reward" : np.array(rewards), "action" : np.array(acs)}
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch
        # Estimate advantage function
        vtargs, vpreds, advs = [], [], []
        for path in paths:
            rew_t = path["reward"]
            return_t = discount(rew_t, gamma)
            vpred_t = vf.predict(path["observation"])
            adv_t = return_t.flatten() - vpred_t
            advs.append(adv_t)
            vtargs.append(return_t)
            vpreds.append(vpred_t)

        # Build arrays for policy update
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_n = np.concatenate([path["action"] for path in paths])
        adv_n = np.concatenate(advs)
        standardized_adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)
        vtarg_n = np.concatenate(vtargs).flatten()
        vpred_n = np.concatenate(vpreds)
        vf.fit(ob_no, vtarg_n)

        # Policy update
        summary, _, _, mean_na_old, std_a_old = sess.run([merged, update_op, grads, mean_na, std_a], feed_dict={sy_ob_no:ob_no, sy_ac_n:ac_n.flatten(), sy_adv_n:standardized_adv_n, sy_stepsize:stepsize})
        file_writer.add_summary(summary, i)
        kl, ent = sess.run([sy_kl, sy_ent], feed_dict={sy_ob_no:ob_no, sy_mean_na_old:mean_na_old, sy_std_a_old:std_a_old})
        #YOUR_CODE_END

        if kl > desired_kl * 2: 
            stepsize /= 1.5
            print('stepsize -> %s'%stepsize)
        elif kl < desired_kl / 2: 
            stepsize *= 1.5
            print('stepsize -> %s'%stepsize)
        else:
            print('stepsize OK')




        # Log diagnostics
        logz.log_tabular("EpRewMean", np.mean([path["reward"].sum() for path in paths]))
        logz.log_tabular("EpLenMean", np.mean([pathlength(path) for path in paths]))
        logz.log_tabular("KLOldNew", kl)
        logz.log_tabular("Entropy", ent)
        logz.log_tabular("EVBefore", explained_variance_1d(vpred_n, vtarg_n))
        logz.log_tabular("EVAfter", explained_variance_1d(vf.predict(ob_no), vtarg_n))
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        # If you're overfitting, EVAfter will be way larger than EVBefore.
        # Note that we fit value function AFTER using it to compute the advantage function to avoid introducing bias
        logz.dump_tabular()


def main_pendulum1(d):
    return main_pendulum(**d)

if __name__ == "__main__":
    tf.reset_default_graph()
    clear_dir('/tmp/ref')
    if 0:
        main_cartpole(logdir='/tmp/ref/linear_vf') # when you want to start collecting results, set the logdir
    if 0:
        main_pendulum(logdir=None, seed=0, desired_kl=2e-3, vf_type='linear', vf_params={}, gamma=0.97, animate=False, min_timesteps_per_batch=2500, n_iter=300, initial_stepsize=1e-3) # when you want to start collecting results, set the logdir
    if 0:
        main_pendulum(logdir=None, seed=0, desired_kl=2e-3, vf_type='nn', vf_params=dict(n_epochs=10, stepsize=1e-3), gamma=0.97, animate=False, min_timesteps_per_batch=2500, n_iter=300, initial_stepsize=1e-3)
    if 0:
        main_cartpole(logdir='/tmp/ref/nn_vf', vf_type='nn')
    if 1:
        general_params = dict(gamma=0.97, animate=False, min_timesteps_per_batch=2500, n_iter=300, initial_stepsize=1e-3)
        params = [
            dict(logdir='/tmp/ref/linearvf-kl2e-3-seed0', seed=0, desired_kl=2e-3, vf_type='linear', vf_params={}, **general_params),
            dict(logdir='/tmp/ref/nnvf-kl2e-3-seed0', seed=0, desired_kl=2e-3, vf_type='nn', vf_params=dict(n_epochs=10, stepsize=1e-3), **general_params),
            dict(logdir='/tmp/ref/linearvf-kl2e-3-seed1', seed=1, desired_kl=2e-3, vf_type='linear', vf_params={}, **general_params),
            dict(logdir='/tmp/ref/nnvf-kl2e-3-seed1', seed=1, desired_kl=2e-3, vf_type='nn', vf_params=dict(n_epochs=10, stepsize=1e-3), **general_params),
            dict(logdir='/tmp/ref/linearvf-kl2e-3-seed2', seed=2, desired_kl=2e-3, vf_type='linear', vf_params={}, **general_params),
            dict(logdir='/tmp/ref/nnvf-kl2e-3-seed2', seed=2, desired_kl=2e-3, vf_type='nn', vf_params=dict(n_epochs=10, stepsize=1e-3), **general_params),
        ]
        """
        import multiprocessing
        p = multiprocessing.Pool()
        p.map(main_pendulum1, params)
        """
        for param in params:
            tf.reset_default_graph()
            main_pendulum1(param)
