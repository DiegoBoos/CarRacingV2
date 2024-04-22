from collections import deque
from car_racing import CarRacing
import numpy as np


class CarRacingV2(CarRacing):
    """
    Enhances the default 'CarRacing' environment tailored for the PPO agent's usage. 
    It fine-tunes the step() function by adjusting the observation space and customizing the reward scheme to align with the PPO algorithm's requirements.
    """

    def __init__(self, image_stack_size=4, step_repeat=8):
        """
        This constructor establishes an enhanced environment by aggregating a sequence of image frames and executing multiple environment steps in a single operation.
        :param image_stack_size: Specifies the count of sequential frames combined to form a comprehensive observation, facilitating the inference of dynamics like speed.
        :param step_repeat: Dictates the repetition frequency of the environment's step function per single external invocation, effectively accelerating the training process due to the high similarity between consecutive frames. 
        Externally, it will seem as though the environment's frame rate is reduced to its native FPS divided by the 'step_repeat' value.
        """

        super().__init__(verbose=0)
        self._image_stack_size = image_stack_size
        self._step_repeat = step_repeat

        # A deque is used to limit the size of the image stack.
        self._image_stack = deque(maxlen=self._image_stack_size)

        # Keep track of past rewards. This is used to end the simulation early if the agent consistently performs poorly
        self._reward_history = deque([0] * 100, maxlen=100)

    def reset(self):
        self._image_stack.clear()
        self._reward_history.extend([0] * 100)
        return super().reset()

    def step(self, action):
        total_reward = 0

        # Repeat steps, accumulating the rewards
        for i in range(self._step_repeat):
            observation, reward, done, info = super().step(action)

            # Punish the agent for going off of the track
            if np.mean(observation[64:80, 42:54, 1]) > 120:
                reward -= 0.25

            # Punish brake usage
            if action is not None:
                punish = np.interp(action[2], [0, 1], [0, 0.05])
                reward -= punish

            # End early if the agent consistently does poorly
            self._reward_history.append(reward)
            if np.mean(self._reward_history) < -0.1:
                done = True

            total_reward += reward
            if done:
                break

        # Add the latest observation to the image stack
        observation = self._preprocess_observation(observation)
        self._image_stack.append(observation)
        while len(self._image_stack) < self._image_stack_size:
            self._image_stack.append(observation)

        # Convert image stack to numpy array
        image_stack_array = np.empty(
            (32, 32, self._image_stack_size), dtype=np.float32)
        for i in range(self._image_stack_size):
            image_stack_array[..., i] = self._image_stack[i]

        return image_stack_array, total_reward, done, info

    @staticmethod
    def _preprocess_observation(observation):

        # Keep only the green channel of the RGB image and reduce the resolution
        observation = observation[::3, ::3, 1]

        # Normalize values between -1 and 1
        observation = (observation / 128) - 1

        # Convert to float
        return observation.astype(np.float32)
