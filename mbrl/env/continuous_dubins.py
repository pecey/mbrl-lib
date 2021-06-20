from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

import numpy as np
import json
import gym
from gym import spaces

DEGREE_TO_RADIAN_MULTIPLIER = np.pi/180

# Actions:
# velocity, steering_angle

class ContinuousDubinsEnv(gym.Env):
    metadata = {
        'render.modes': ['human'],
    }

    def __init__(self, noisy=False, noise_mean=0, noise_var=0, starting_point=(2, 4), goal_position=(5, 1.5), add_obstacle=False, config_file=None):
        # Min and Max positions for goal and starting point
        self.min_y_position = 0
        self.min_x_position = 0
        self.max_y_position = 8
        self.max_x_position = 8
        self.min_theta = -60
        self.max_theta = 60

        # Min velocity is positive. Car cannot go back
        self.min_velocity = 0
        self.max_velocity = 1
        self.min_steering = -30
        self.max_steering = 30
        # Random position for goal
        self.goal_x = np.random.randint(self.min_x_position, self.max_x_position)
        self.goal_y = np.random.randint(self.min_y_position, self.max_y_position)

        # Random starting point for agent
        self.x = np.random.randint(self.min_x_position, self.max_x_position)
        self.y = np.random.randint(self.min_y_position, self.max_y_position)

        self.config_file = config_file

        self.add_obstacle = add_obstacle
        if self.add_obstacle:
            self.obstacles = []
            # self.obstacle_x, self.obstacle_y = 5, 5
            # self.obstacle_height = 1
            # self.obstacle_width = 1
            self.max_obstacle_width = 3
            self.max_obstacle_height = 1

        # self.starting_point = starting_point
        # self.goal_position = goal_position
        #
        # self.goal_x, self.goal_y = self.goal_position
        # self.x, self.y = self.starting_point

        # Define the observation space.
        self.low_state = np.array([self.min_x_position, self.min_y_position, self.min_theta], dtype=np.float32)
        self.high_state = np.array([self.max_x_position, self.max_y_position, self.max_theta], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )

        # self.nS = 3
        # self.nA = 2

        # Define the action space
        self.low_action = np.array([self.min_velocity, self.min_steering], dtype=np.float32)
        self.high_action = np.array([self.max_velocity, self.max_steering], dtype=np.float32)
        self.action_space = spaces.Box(
            low=self.low_action,
            high=self.high_action,
            dtype=np.float32
        )

        self.theta = 0.0
        self.time_interval = 0.2
        # self.time_interval = 1

        self.goal_boundary = 0.5

        self.x_traj = [self.x]
        self.y_traj = [self.y]

        self.done = False

        self.add_noise = noisy

        self.reset_canvas()

    def step(self, actions):
        reward = 0
        noise = 0
        velocity, steering = actions
        steering = steering

        if self.add_noise:
            noise = np.random.normal(0, 1)

        x_old = self.x
        y_old = self.y

        self.x, self.y, self.theta = self.update_state(self.x, self.y, self.theta, steering, velocity, self.time_interval, noise)

        # Give a "bubble" of 1
        if np.isclose(self.x, self.goal_x, atol=self.goal_boundary) and np.isclose(self.y, self.goal_y, atol=self.goal_boundary):
            self.done = True

        if self.add_obstacle:
            if self.check_collision(self.x, self.y):
                reward += -10
                self.x = x_old
                self.y = y_old

        self.x_traj.append(self.x)
        self.y_traj.append(self.y)

        if self.done:
            reward += 1

        observation = self.x, self.y, self.theta

        return np.array(observation), reward, self.done, {"goal":(self.goal_x, self.goal_y)}

    def check_collision(self, x, y):
        for obstacle in self.obstacles:
            collision_x = True if obstacle["x"] <= x <= obstacle["x"] + obstacle["width"] else False
            collision_y = True if obstacle["y"] <= y <= obstacle["y"] + obstacle["height"] else False
            if collision_x and collision_y:
                return True
        return False

    def reset(self):
        if self.config_file is not None:
            with open(self.config_file, 'r') as f:
                config_data = f.read()
            configs = json.loads(config_data)
            selected_config = np.random.choice(configs)
            self.x, self.y = selected_config['x'], selected_config['y']
            self.goal_x, self.goal_y = selected_config['goal_x'], selected_config['goal_y']
            if self.add_obstacle:
                self.obstacles = selected_config['obstacles']
                # self.obstacle_x, self.obstacle_y = selected_config['obstacle_x'], selected_config['obstacle_y']
                # self.obstacle_width, self.obstacle_height = selected_config['obstacle_width'], selected_config['obstacle_height']
        else:
            goal_behind_car = np.random.randint(0, 2)
            if goal_behind_car:
                self.goal_x = np.random.randint(self.min_x_position, self.max_x_position/2)
                self.goal_y = np.random.randint(self.min_y_position, self.max_y_position/2)

                self.x = np.random.randint(self.goal_x, self.max_x_position)
                self.y = np.random.randint(self.goal_y, self.max_y_position)
            else:
                # Random position for goal
                self.goal_x = np.random.randint(self.min_x_position, self.max_x_position)
                self.goal_y = np.random.randint(self.min_y_position, self.max_y_position)
                # Random starting point for agent
                self.x = np.random.randint(self.min_x_position, self.max_x_position)
                self.y = np.random.randint(self.min_y_position, self.max_y_position)


        self.theta = float(np.random.randint(self.min_theta, self.max_theta))

        self.x_traj = [self.x]
        self.y_traj = [self.y]

        plt.close(self.fig)
        self.reset_canvas()

        self.done = False
        return np.array((self.x, self.y, self.theta))

    def reset_canvas(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)

        self.ax.set_xlim(self.min_x_position, self.max_x_position)
        self.ax.set_ylim(self.min_y_position, self.max_y_position)

        self.graph, = self.ax.plot(self.x_traj, self.y_traj, '--', alpha=0.8)
        self.car, = self.ax.plot(self.x, self.y, 'o')

        # Goal plotted by a green "x"
        self.ax.plot(self.goal_x, self.goal_y, "gx")

        # Plot the goal boundary
        boundary = plt.Circle((self.goal_x, self.goal_y), radius=self.goal_boundary, color='orange', alpha=0.8)
        self.ax.add_artist(boundary)

        # Plot the obstacle
        if self.add_obstacle:
            for obstacle in self.obstacles:
                self.ax.add_patch(Rectangle((obstacle["x"], obstacle["y"]), obstacle["width"], obstacle["height"]))

    def render(self, mode='human', close=False):
        """
        Update the trajectory
        """
        self.graph.set_data(self.x_traj, self.y_traj)
        self.car.set_data(self.x, self.y)

        plt.pause(0.01)

    def save_trajectory(self, path):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.set_xlim(self.min_x_position, self.max_x_position)
        ax.set_ylim(self.min_y_position, self.max_y_position)

        ax.plot(self.x_traj, self.y_traj, '--', alpha=0.8)
        ax.plot(self.x, self.y, 'o')

        ax.plot(self.goal_x, self.goal_y, "gx")
        boundary = plt.Circle((self.goal_x, self.goal_y), radius=self.goal_boundary, color='orange', alpha=0.8)
        ax.add_artist(boundary)

        if self.add_obstacle:
            for obstacle in self.obstacles:
                ax.add_patch(Rectangle((obstacle["x"], obstacle["y"]), obstacle["width"], obstacle["height"]))

        plt.savefig(path)
        plt.close()

    # remove all variables and use self.
    # keep direction variable
    def update_state(self, x, y, theta, steering, velocity, time_interval, noise):
        """
        Update the state [x, y, theta] of the robot according to Dubins dynamics.
        """
        new_x = np.clip(x + velocity * np.cos(theta) * time_interval, self.min_x_position, self.max_x_position)
        new_y = np.clip(y + velocity * np.sin(theta) * time_interval, self.min_y_position, self.max_y_position)
        new_theta = theta + (steering + noise) * time_interval * DEGREE_TO_RADIAN_MULTIPLIER
        return new_x, new_y, (new_theta)


if __name__ == '__main__':
    from gym.envs.registration import register

    register(
        id='continuous_dubins-v0',
        entry_point='env.continuous_dubins_env:ContinuousDubinsEnv',
        max_episode_steps=500
    )

    env = gym.make('continuous_dubins-v0')
    env.reset()
    done = False
    step = 0
    while not done:
        move = (np.clip(np.random.normal(1, 1), a_min=0.02, a_max=None), np.random.uniform(-30, 30))
        obs, rew, done, info = env.step(move)
        step += 1
        print(f"Step: {step}, Move: {move}, Observation: {obs}")
    env.render().show()