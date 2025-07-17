import gymnasium as gym
import numpy as np
from PIL import Image
import io
from collections import deque
from flax import nnx
import jax.numpy as jnp
import optax
import jax
from tqdm import tqdm
import matplotlib.pyplot as plt

class NN(nnx.Module):
    def __init__(self,
                in_channels: int,
                n_hidden: int,
                out_channels: int,
                *,
                rngs: nnx.Rngs):
        self.layer1 = nnx.Linear(in_channels, n_hidden, rngs = rngs)
        self.layer2 = nnx.Linear(n_hidden, n_hidden, rngs = rngs)
        self.layer3 = nnx.Linear(n_hidden, out_channels, rngs = rngs)

    def __call__(self,x):
        x = nnx.selu(self.layer1(x))
        x = nnx.selu(self.layer2(x))
        x = self.layer3(x)
        return x

def loss_fn(model: nnx.Module,
            observations: jax.Array,
            actions: jax.Array,
            returns: jax.Array):
    logits = model(observations)
    log_probs = jax.nn.log_softmax(logits)
    action_log_probs = log_probs[jnp.arange(len(actions)), actions]
    loss = -jnp.mean(action_log_probs * returns)
    return loss

@nnx.jit
def train_step( model: nnx.Module,
                optimizer: nnx.Optimizer,
                observations: jax.Array,
                actions: jax.Array,
                returns: jax.Array):
    loss, grads = nnx.value_and_grad(loss_fn)(model, observations, actions, returns)
    optimizer.update(grads)
    pass

class Agent:
    def __init__(self):
        self.env = gym.make("CartPole-v1", render_mode = "human")
        self.discount_factor = 0.9999
        self.model = None
        self.optimizer = None
        self.obs_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.replay_buffer = deque(maxlen=100)
        self.key = jax.random.PRNGKey(42)
        self.episodic_returns = []


    def train(self):
        self.model = NN(in_channels = self.obs_size,
                    n_hidden = 128,
                    out_channels = self.action_size,
                    rngs = nnx.Rngs(42))

        self.optimizer = nnx.Optimizer(self.model, optax.adam(learning_rate = 0.001))


        # Outer loop: number of runs of environment
        for episode in range(1000):
            obs, info = self.env.reset()
            experience = []

            # Inner loop: number of time steps within each environment instantiation
            for i in range(1000):
                self.key, subkey = jax.random.split(self.key)

                # Model returns a probability distribution over actions, we take the action with the highest probability
                logits = self.model(jnp.array(obs).reshape(1,-1))
                action = int(jax.random.categorical(subkey, logits[0]))

                # Take action and observe next state, reward, and termination
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                experience.append({"obs": obs, "action": action, "reward": reward, "next_obs": next_obs, "terminated": terminated})
                obs = next_obs

                if terminated or truncated or i == 999:
                    N = len(experience)
                    discounted_return = 0
                    for t in reversed(range(N)):
                        discounted_return = experience[t]["reward"] + (1-int(experience[t]["terminated"])) * self.discount_factor*discounted_return
                        experience[t]["discounted_return"] = discounted_return

                    self.replay_buffer.append(experience)
                    episodic_return = experience[0]["discounted_return"]
                    self.episodic_returns.append(episodic_return)

                    break

            if len(self.replay_buffer) > 0:
                episode_data = self.replay_buffer[-1]

                observations = jnp.array([exp["obs"] for exp in experience])
                actions = jnp.array([exp["action"] for exp in experience])
                returns = jnp.array([exp["discounted_return"] for exp in experience])

                #Normalise returns
                returns = (returns - jnp.mean(returns))/(jnp.std(returns) + 1e-8)

                loss = train_step(model = self.model,
                                  optimizer = self.optimizer,
                                  observations = observations,
                                  actions = actions, 
                                  returns = returns)
            print(f"Episodic return: {episodic_return}")
                
        plt.plot(self.episodic_returns)
        plt.show()
                
            
                            

def main():
    agent = Agent()
    agent.train()



if __name__ == "__main__":
    main()