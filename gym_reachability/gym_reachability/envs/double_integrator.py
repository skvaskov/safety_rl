"""
This module implements an environment considering Double integrator dynamics. This
environemnt shows reach-avoid reinforcement learning's performance on a
well-known reachability analysis benchmark.
"""

import gym.spaces
import numpy as np
import gym
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
import torch
import random
from .double_integrator_dyn import DoubleIntegratorDyn
from .env_utils import calculate_margin_rect
import numpy as np

class DoubleIntegratorEnv(gym.Env):
  """A gym environment considering Double integrator dynamics.
  """

  def __init__(
      self, device, mode="RA", doneType="toEnd", sample_inside_obs=False,
      sample_inside_tar=True
  ):
    """Initializes the environment with given arguments.

    Args:
        device (str): device type (used in PyTorch).
        mode (str, optional): reinforcement learning type. Defaults to "RA".
        doneType (str, optional): the condition to raise `done flag in
            training. Defaults to "toEnd".
        sample_inside_obs (bool, optional): consider sampling the state inside
            the obstacles if True. Defaults to False.
        sample_inside_tar (bool, optional): consider sampling the state inside
            the target if True. Defaults to True.
    """

    # Set random seed.
    self.set_seed(0)

    # State bounds.
    self.bounds = np.array([[-1.1, 1.1], [-1.1, 1.1]])
    self.low = self.bounds[:, 0]
    self.high = self.bounds[:, 1]
    self.sample_inside_obs = sample_inside_obs
    self.sample_inside_tar = sample_inside_tar

    # Gym variables.
    self.action_space = gym.spaces.Discrete(3)
    midpoint = (self.low + self.high) / 2.0
    interval = self.high - self.low
    self.observation_space = gym.spaces.Box(
        np.float32(midpoint - interval/2),
        np.float32(midpoint + interval/2),
    )

    # Constraint set parameters.
    self.constraint_x_y_w_h = np.array([0, 0, 2, 2])

    # Target set parameters.
    self.target_x_y_w_h = np.array([0.8, 0.0, 0.1, 1e-2])

    # Internal state.
    self.mode = mode
    self.state = np.zeros(2)
    self.doneType = doneType

    # Dubins car parameters.
    self.time_step = 0.05
    self.acceleration_limits = [-1,1]
    self.car = DoubleIntegratorDyn(doneType=doneType)

    # Visualization params
    self.visual_initial_states = [
        np.array([0.5, -0.5]),
        np.array([-0.5, -0.5]),
        np.array([0.5, 0.5]),
        np.array([-0.5, 0.5]),
        np.array([-0.95, 0.0]),
        np.array([0.95,0.0]),
    ]

    # Cost Params
    self.targetScaling = 1.0
    self.safetyScaling = 1.0
    self.penalty = 1.0
    self.reward = -1.0
    self.costType = "sparse"
    self.device = device
    self.scaling = 1.0

    print(
        "Env: mode-{:s}; doneType-{:s}; sample_inside_obs-{}".format(
            self.mode, self.doneType, self.sample_inside_obs
        )
    )

  def init_car(self):
    """
    Initializes the dynamics, constraint and the target set of a Dubins car.
    """
    self.car.set_bounds(bounds=self.bounds)
    self.car.set_constraint(self.constraint_x_y_w_h)
    self.car.set_target(self.target_x_y_w_h)
    self.car.set_time_step(time_step=self.time_step)

  # == Reset Functions ==
  def reset(self, start=None):
    """Resets the state of the environment.

    Args:
        start (np.ndarray, optional): state to reset the environment to.
            If None, pick the state uniformly at random. Defaults to None.

    Returns:
        np.ndarray: The state that the environment has been reset to.
    """
    self.state = self.car.reset(
        start=start,
        sample_inside_obs=self.sample_inside_obs,
        sample_inside_tar=self.sample_inside_tar,
    )
    return np.copy(self.state)

  def sample_random_state(
      self, sample_inside_obs=False, sample_inside_tar=True
  ):
    """Picks the state of the environment uniformly at random.

    Args:
        sample_inside_obs (bool, optional): consider sampling the state inside
            the obstacles if True. Defaults to False.
        sample_inside_tar (bool, optional): consider sampling the state inside
            the target if True. Defaults to True.


    Returns:
        np.ndarray: the sampled initial state.
    """
    state = self.car.sample_random_state(
        sample_inside_obs=sample_inside_obs,
        sample_inside_tar=sample_inside_tar,
    )
    return state

  # == Dynamics Functions ==
  def step(self, action):
    """Evolves the environment one step forward given an action.

    Args:
        action (int): the index of the action in the action set.

    Returns:
        np.ndarray: next state.
        float: the standard cost used in reinforcement learning.
        bool: True if the episode is terminated.
        dict: consist of target margin and safety margin at the new state.
    """
    distance = np.linalg.norm(self.state - self.car.state)
    assert distance < 1e-8, (
        "There is a mismatch between the env state"
        + "and car state: {:.2e}".format(distance)
    )

    state_nxt, _ = self.car.step(action)
    self.state = state_nxt
    l_x = self.target_margin(self.state[:2])
    g_x = self.safety_margin(self.state[:2])

    fail = g_x > 0
    success = l_x <= 0

    # cost
    if self.mode == "RA":
      if fail:
        cost = self.penalty
      elif success:
        cost = self.reward
      else:
        cost = 0.0
    else:
      if fail:
        cost = self.penalty
      elif success:
        cost = self.reward
      else:
        if self.costType == "dense_ell":
          cost = l_x
        elif self.costType == "dense_ell_g":
          cost = l_x + g_x
        elif self.costType == "sparse":
          cost = 0.0 * self.scaling
        elif self.costType == "max_ell_g":
          cost = max(l_x, g_x)
        else:
          cost = 0.0

    # = `done` signal
    if self.doneType == "toEnd":
      done = not self.car.check_within_bounds(self.state)
    elif self.doneType == "fail":
      done = fail
    elif self.doneType == "TF":
      done = fail or success
    else:
      raise ValueError("invalid done type!")

    # = `info`
    if done and self.doneType == "fail":
      info = {"g_x": self.penalty * self.scaling, "l_x": l_x}
    else:
      info = {"g_x": g_x, "l_x": l_x}
    return np.copy(self.state), cost, done, info

  # == Setting Hyper-Parameter Functions ==
  def set_costParam(
      self, penalty=1.0, reward=-1.0, costType="sparse", targetScaling=1.0,
      safetyScaling=1.0
  ):
    """
    Sets the hyper-parameters for the `cost` signal used in training, important
    for standard (Lagrange-type) reinforcement learning.

    Args:
        penalty (float, optional): cost when entering the obstacles or
            crossing the environment boundary. Defaults to 1.0.
        reward (float, optional): cost when reaching the targets.
            Defaults to -1.0.
        costType (str, optional): providing extra information when in
            neither the failure set nor the target set.
            Defaults to 'sparse'.
        targetScaling (float, optional): scaling factor of the target
            margin. Defaults to 1.0.
        safetyScaling (float, optional): scaling factor of the safety
            margin. Defaults to 1.0.
    """
    self.penalty = penalty
    self.reward = reward
    self.costType = costType
    self.safetyScaling = safetyScaling
    self.targetScaling = targetScaling

  def set_seed(self, seed):
    """Sets the seed for `numpy`, `random`, `PyTorch` packages.

    Args:
        seed (int): seed value.
    """
    self.seed_val = seed
    np.random.seed(self.seed_val)
    torch.manual_seed(self.seed_val)
    torch.cuda.manual_seed(self.seed_val)
    # if you are using multi-GPU.
    torch.cuda.manual_seed_all(self.seed_val)
    random.seed(self.seed_val)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

  def set_bounds(self, bounds):
    """Sets the boundary and the observation space of the environment.

    Args:
        bounds (np.ndarray): of the shape (n_dim, 2). Each row is [LB, UB].
    """
    self.bounds = bounds

    # Get lower and upper bounds
    self.low = np.array(self.bounds)[:, 0]
    self.high = np.array(self.bounds)[:, 1]

    # Double the range in each state dimension for Gym interface.
    midpoint = (self.low + self.high) / 2.0
    interval = self.high - self.low
    self.observation_space = gym.spaces.Box(
        np.float32(midpoint - interval/2),
        np.float32(midpoint + interval/2),
    )
    self.car.set_bounds(bounds)



  def set_constraint(self,xywh):
    """Sets the constraint set (complement of failure set).

    Args:
        center (np.ndarray, optional): center of the constraint set.
            Defaults to np.array([0.,0.]).
        radius (float, optional): radius of the constraint set.
            Defaults to 1.0.
    """
    self.constraint_x_y_w_h = xywh
    self.car.set_constraint(*xywh)

  def set_target(self,xywh):
    """Sets the target set.

    Args:
        center (np.ndarray, optional): center of the target set.
            Defaults to np.array([0.,0.]).
        radius (float, optional): radius of the target set. Defaults to .4.
    """
    self.target_x_y_w_h = xywh
    self.car.set_target(*xywh)

  # == Margin Functions ==
  def safety_margin(self, s):
    """Computes the margin (e.g. distance) between the state and the failue set.

    Args:
        s (np.ndarray): the state of the agent.

    Returns:
        float: postivive numbers indicate being inside the failure set (safety
            violation).
    """
    return self.car.safety_margin(s[:2])

  def target_margin(self, s):
    """Computes the margin (e.g. distance) between the state and the target set.

    Args:
        s (np.ndarray): the state of the agent.

    Returns:
        float: negative numbers indicate reaching the target. If the target set
            is not specified, return None.
    """
    return self.car.target_margin(s[:2])

  # == Getting Functions ==
  def get_warmup_examples(self, num_warmup_samples=100):
    """Gets warmup samples.

    Args:
        num_warmup_samples (int, optional): # warmup samples. Defaults to 100.

    Returns:
        np.ndarray: sampled states.
        np.ndarray: the heuristic values, here we used max{ell, g}.
    """
    rv = np.random.uniform(
        low=self.low, high=self.high, size=(num_warmup_samples, 2)
    )
    x_rnd, y_rnd = rv[:, 0], rv[:, 1]

    heuristic_v = np.zeros((num_warmup_samples, self.action_space.n))
    states = np.zeros((num_warmup_samples, self.observation_space.shape[0]))

    for i in range(num_warmup_samples):
      x, y = x_rnd[i], y_rnd[i]
      l_x = self.target_margin(np.array([x, y]))
      g_x = self.safety_margin(np.array([x, y]))
      heuristic_v[i, :] = np.maximum(l_x, g_x)
      states[i, :] = x, y

    return states, heuristic_v

  def get_axes(self):
    """Gets the axes bounds and aspect_ratio.

    Returns:
        np.ndarray: axes bounds.
        float: aspect ratio.
    """
    aspect_ratio = ((self.bounds[0, 1] - self.bounds[0, 0]) /
                    (self.bounds[1, 1] - self.bounds[1, 0]))
    axes = np.array([
        self.bounds[0, 0],
        self.bounds[0, 1],
        self.bounds[1, 0],
        self.bounds[1, 1],
    ])
    return [axes, aspect_ratio]

  def get_value(self, q_func, nx=101, ny=101, addBias=False):
    """
    Gets the state values given the Q-network. We fix the heading angle of the

    Args:
        q_func (object): agent's Q-network.

        nx (int, optional): # points in x-axis. Defaults to 101.
        ny (int, optional): # points in y-axis. Defaults to 101.
        addBias (bool, optional): adding bias to the values or not.
            Defaults to False.

    Returns:
        np.ndarray: values
    """
    v = np.zeros((nx, ny))
    it = np.nditer(v, flags=["multi_index"])
    xs = np.linspace(self.bounds[0, 0], self.bounds[0, 1], nx)
    ys = np.linspace(self.bounds[1, 0], self.bounds[1, 1], ny)
    while not it.finished:
      idx = it.multi_index
      x = xs[idx[0]]
      y = ys[idx[1]]
      l_x = self.target_margin(np.array([x, y]))
      g_x = self.safety_margin(np.array([x, y]))

      if self.mode == "normal" or self.mode == "RA":
        state = (torch.FloatTensor([x, y]).to(self.device).unsqueeze(0))
      else:
        z = max([l_x, g_x])
        state = (
            torch.FloatTensor([x, y, z]).to(self.device).unsqueeze(0)
        )
      if addBias:
        v[idx] = q_func(state).min(dim=1)[0].item() + max(l_x, g_x)
      else:
        v[idx] = q_func(state).min(dim=1)[0].item()
      it.iternext()
    return v

  # == Trajectory Functions ==
  def simulate_one_trajectory(
      self, q_func, T=10, state=None, sample_inside_obs=True,
      sample_inside_tar=True, toEnd=False
  ):
    """Simulates the trajectory given the state or randomly initialized.

    Args:
        q_func (object): agent's Q-network.
        T (int, optional): the maximum length of the trajectory. Defaults
            to 250.
        state (np.ndarray, optional): if provided, set the initial state to
            its value. Defaults to None.
   
        sample_inside_obs (bool, optional): sampling initial states inside
            of the obstacles or not. Defaults to True.
        sample_inside_tar (bool, optional): sampling initial states inside
            of the targets or not. Defaults to True.
        toEnd (bool, optional): simulate the trajectory until the robot
            crosses the boundary or not. Defaults to False.

    Returns:
        np.ndarray: states of the trajectory, of the shape (length, 3).
        int: result.
        float: the minimum reach-avoid value of the trajectory.
        dictionary: extra information, (v_x, g_x, ell_x) along the traj.
    """
    # reset
    if state is None:
      state = self.car.sample_random_state(
          sample_inside_obs=sample_inside_obs,
          sample_inside_tar=sample_inside_tar,
      )
    traj = []
    result = 0  # not finished
    valueList = []
    gxList = []
    lxList = []
    for t in range(T):
      traj.append(state)

      g_x = self.safety_margin(state)
      l_x = self.target_margin(state)

      # = Rollout Record
      if t == 0:
        maxG = g_x
        current = max(l_x, maxG)
        minV = current
      else:
        maxG = max(maxG, g_x)
        current = max(l_x, maxG)
        minV = min(current, minV)

      valueList.append(minV)
      gxList.append(g_x)
      lxList.append(l_x)

      if toEnd:
        done = not self.car.check_within_bounds(state)
        if done:
          result = 1
          break
      else:
        if g_x > 0:
          result = -1  # failed
          break
        elif l_x <= 0:
          result = 1  # succeeded
          break

      q_func.eval()
      state_tensor = (torch.FloatTensor(state).to(self.device).unsqueeze(0))
      action_index = q_func(state_tensor).min(dim=1)[1].item()
      u = self.car.discrete_controls[action_index]

      state = self.car.integrate_forward(state, u)
    traj = np.array(traj)
    info = {"valueList": valueList, "gxList": gxList, "lxList": lxList}
    return traj, result, minV, info

  def simulate_trajectories(
      self, q_func, T=10, num_rnd_traj=None, states=None, toEnd=False
  ):
    """
    Simulates the trajectories. If the states are not provided, we pick the
    initial states from the discretized state space.

    Args:
        q_func (object): agent's Q-network.
        T (int, optional): the maximum length of the trajectory.
            Defaults to 250.
        num_rnd_traj (int, optional): #trajectories. Defaults to None.
        states (list of np.ndarray, optional): if provided, set the initial
            states to its value. Defaults to None.
        toEnd (bool, optional): simulate the trajectory until the robot
            crosses the boundary or not. Defaults to False.

    Returns:
        list of np.ndarray: each element is a tuple consisting of x and y
            positions along the trajectory.
        np.ndarray: the binary reach-avoid outcomes.
        np.ndarray: the minimum reach-avoid values of the trajectories.
    """
    assert ((num_rnd_traj is None and states is not None)
            or (num_rnd_traj is not None and states is None)
            or (len(states) == num_rnd_traj))
    trajectories = []

    if states is None:
      nx = 41
      ny = nx
      xs = np.linspace(self.bounds[0, 0], self.bounds[0, 1], nx)
      ys = np.linspace(self.bounds[1, 0], self.bounds[1, 1], ny)
      results = np.empty((nx, ny), dtype=int)
      minVs = np.empty((nx, ny), dtype=float)

      it = np.nditer(results, flags=["multi_index"])
      print()
      while not it.finished:
        idx = it.multi_index
        print(idx, end="\r")
        x = xs[idx[0]]
        y = ys[idx[1]]
        state = np.array([x, y])
        traj, result, minV, _ = self.simulate_one_trajectory(
            q_func, T=T, state=state, toEnd=toEnd
        )
        trajectories.append((traj))
        results[idx] = result
        minVs[idx] = minV
        it.iternext()
      results = results.reshape(-1)
      minVs = minVs.reshape(-1)

    else:
      results = np.empty(shape=(len(states),), dtype=int)
      minVs = np.empty(shape=(len(states),), dtype=float)
      for idx, state in enumerate(states):
        traj, result, minV, _ = self.simulate_one_trajectory(
            q_func, T=T, state=state, toEnd=toEnd
        )
        trajectories.append(traj)
        results[idx] = result
        minVs[idx] = minV

    return trajectories, results, minVs

  # == Plotting Functions ==
  def render(self):
    pass

  def visualize(
      self, q_func, vmin=-1, vmax=1, nx=101, ny=101, cmap="seismic",
      labels=None, boolPlot=False, addBias=False,
      rndTraj=False, num_rnd_traj=10
  ):
    """
    Visulaizes the trained Q-network in terms of state values and trajectories
    rollout.

    Args:
        q_func (object): agent's Q-network.
        vmin (int, optional): vmin in colormap. Defaults to -1.
        vmax (int, optional): vmax in colormap. Defaults to 1.
        nx (int, optional): # points in x-axis. Defaults to 101.
        ny (int, optional): # points in y-axis. Defaults to 101.
        cmap (str, optional): color map. Defaults to 'seismic'.
        labels (list, optional): x- and y- labels. Defaults to None.
        boolPlot (bool, optional): plot the values in binary form.
            Defaults to False.
        addBias (bool, optional): adding bias to the values or not.
            Defaults to False.
 
        rndTraj (bool, optional): randomli choose trajectories if True.
            Defaults to False.
        num_rnd_traj (int, optional): #trajectories. Defaults to None.
    """
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot()

    ax.cla()
    cbarPlot = True

    # == Plot failure / target set ==
    self.plot_target_failure_set(ax)

    # == Plot reach-avoid set ==
    self.plot_reach_avoid_set(ax)

    # == Plot V ==
    self.plot_v_values(
        q_func,
        ax=ax,
        fig=fig,
        vmin=vmin,
        vmax=vmax,
        nx=nx,
        ny=ny,
        cmap=cmap,
        boolPlot=boolPlot,
        cbarPlot=cbarPlot,
        addBias=addBias,
    )
    # == Formatting ==
    self.plot_formatting(ax=ax, labels=labels)

    # == Plot Trajectories ==
    if rndTraj:
      self.plot_trajectories(
          q_func,
          T=200,
          num_rnd_traj=num_rnd_traj,
          toEnd=False,
          ax=ax,
          c="y",
          lw=2,
      )
    else:
      self.plot_trajectories(
          q_func,
          T=200,
          states=self.visual_initial_states,
          toEnd=False,
          ax=ax,
          c="y",
          lw=2,
      )

    plt.tight_layout()

  def plot_v_values(
      self, q_func, ax=None, fig=None, vmin=-1, vmax=1,
      nx=201, ny=201, cmap="seismic", boolPlot=False, cbarPlot=True,
      addBias=False
  ):
    """Plots state values.

    Args:
        q_func (object): agent's Q-network.

        ax (matplotlib.axes.Axes, optional): Defaults to None.
        fig (matplotlib.figure, optional): Defaults to None.
        vmin (int, optional): vmin in colormap. Defaults to -1.
        vmax (int, optional): vmax in colormap. Defaults to 1.
        nx (int, optional): # points in x-axis. Defaults to 201.
        ny (int, optional): # points in y-axis. Defaults to 201.
        cmap (str, optional): color map. Defaults to 'seismic'.
        boolPlot (bool, optional): plot the values in binary form.
            Defaults to False.
        cbarPlot (bool, optional): plot the color bar or not. Defaults to True.
        addBias (bool, optional): adding bias to the values or not.
            Defaults to False.
    """
    axStyle = self.get_axes()
    ax.plot([0.0, 0.0], [axStyle[0][2], axStyle[0][3]], c="k")
    ax.plot([axStyle[0][0], axStyle[0][1]], [0.0, 0.0], c="k")


    v = self.get_value(q_func, nx, ny, addBias=addBias)

    if boolPlot:
      im = ax.imshow(
          v.T > 0.0,
          interpolation="none",
          extent=axStyle[0],
          origin="lower",
          cmap=cmap,
          zorder=-1,
      )
    else:
      im = ax.imshow(
          v.T,
          interpolation="none",
          extent=axStyle[0],
          origin="lower",
          cmap=cmap,
          vmin=vmin,
          vmax=vmax,
          zorder=-1,
      )
      if cbarPlot:
        cbar = fig.colorbar(
            im,
            ax=ax,
            pad=0.01,
            fraction=0.05,
            shrink=0.95,
            ticks=[vmin, 0, vmax],
        )
        cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=16)

  def plot_trajectories(
      self, q_func, T=100, num_rnd_traj=None, states=None, 
      toEnd=False, ax=None, c="y", lw=1.5, orientation=0, zorder=2
  ):
    """Plots trajectories given the agent's Q-network.

    Args:
        q_func (object): agent's Q-network.
        T (int, optional): the maximum length of the trajectory.
            Defaults to 100.
        num_rnd_traj (int, optional): #states. Defaults to None.
        states (list of np.ndarray, optional): if provided, set the initial
            states to its value. Defaults to None.

        toEnd (bool, optional): simulate the trajectory until the robot
            crosses the boundary or not. Defaults to False.
        ax (matplotlib.axes.Axes, optional): Defaults to None.
        c (str, optional): color. Defaults to 'y'.
        lw (float, optional): linewidth. Defaults to 1.5.
        orientation (float, optional): counter-clockwise angle. Defaults
            to 0.
        zorder (int, optional): graph layers order. Defaults to 2.

    Returns:
        np.ndarray: the binary reach-avoid outcomes.
        np.ndarray: the minimum reach-avoid values of the trajectories.
    """
    assert ((num_rnd_traj is None and states is not None)
            or (num_rnd_traj is not None and states is None)
            or (len(states) == num_rnd_traj))


    trajectories, results, minVs = self.simulate_trajectories(
        q_func, T=T, num_rnd_traj=num_rnd_traj, states=states, toEnd=toEnd
    )
    if ax is None:
      ax = plt.gca()
    for traj in trajectories:
      traj_x = traj[:, 0]
      traj_y = traj[:, 1]
      ax.scatter(traj_x[0], traj_y[0], s=48, c=c, zorder=zorder)
      ax.plot(traj_x, traj_y, color=c, linewidth=lw, zorder=zorder)

    return results, minVs

  def plot_target_failure_set(self, ax=None, c_c="m", c_t="y", lw=3, zorder=0):
    """Plots the boundary of the target and the failure set.

    Args:
        ax (matplotlib.axes.Axes, optional): ax to plot.
        c_c (str, optional): color of the constraint set boundary.
            Defaults to 'm'.
        c_t (str, optional): color of the target set boundary.
            Defaults to 'y'.
        lw (float, optional): linewidth of the boundary. Defaults to 3.
        zorder (int, optional): graph layers order. Defaults to 0.
    """
    constraint_x = [self.constraint_x_y_w_h[0] - self.constraint_x_y_w_h[2]/2,
                    self.constraint_x_y_w_h[0] + self.constraint_x_y_w_h[2]/2,
                    self.constraint_x_y_w_h[0] + self.constraint_x_y_w_h[2]/2,
                    self.constraint_x_y_w_h[0] - self.constraint_x_y_w_h[2]/2,
                    self.constraint_x_y_w_h[0] - self.constraint_x_y_w_h[2]/2]
    
    constraint_y = [self.constraint_x_y_w_h[1] - self.constraint_x_y_w_h[3]/2,
                self.constraint_x_y_w_h[1] - self.constraint_x_y_w_h[3]/2,
                self.constraint_x_y_w_h[1] + self.constraint_x_y_w_h[3]/2,
                self.constraint_x_y_w_h[1] + self.constraint_x_y_w_h[3]/2,
                self.constraint_x_y_w_h[1] - self.constraint_x_y_w_h[3]/2]

    ax.plot(constraint_x,constraint_y,color=c_c,linewidth=lw,zorder=zorder)

    target_x = [self.target_x_y_w_h[0] - self.target_x_y_w_h[2]/2,
                    self.target_x_y_w_h[0] + self.target_x_y_w_h[2]/2,
                    self.target_x_y_w_h[0] + self.target_x_y_w_h[2]/2,
                    self.target_x_y_w_h[0] - self.target_x_y_w_h[2]/2,
                    self.target_x_y_w_h[0] - self.target_x_y_w_h[2]/2]
    
    target_y = [self.target_x_y_w_h[1] - self.target_x_y_w_h[3]/2,
                self.target_x_y_w_h[1] - self.target_x_y_w_h[3]/2,
                self.target_x_y_w_h[1] + self.target_x_y_w_h[3]/2,
                self.target_x_y_w_h[1] + self.target_x_y_w_h[3]/2,
                self.target_x_y_w_h[1] - self.target_x_y_w_h[3]/2]

    ax.plot(target_x,target_y,color=c_t,linewidth=lw,zorder=zorder)

  def plot_reach_avoid_set(
      self, ax=None, c="g", lw=3, zorder=1
  ):
    """Plots the analytic reach-avoid set.

    Args:
        ax (matplotlib.axes.Axes, optional): ax to plot.
        c (str, optional): color of the rach-avoid set boundary.
            Defaults to 'g'.
        lw (int, optional): linewidth of the boundary. Defaults to 3.
        orientation (float, optional): counter-clockwise angle. Defaults
            to 0.
        zorder (int, optional): graph layers order. Defaults to 1.
    """
    x_lower = self.constraint_x_y_w_h[0] - self.constraint_x_y_w_h[2]/2
    x_upper = self.constraint_x_y_w_h[0] + self.constraint_x_y_w_h[2]/2

    v_lower = self.constraint_x_y_w_h[1] - self.constraint_x_y_w_h[3]/2
    v_upper = self.constraint_x_y_w_h[1] + self.constraint_x_y_w_h[3]/2

    stopping_points_high = np.linspace(x_upper+0.5*v_upper**2/self.car.acceleration_limits[0],x_upper)
    v_high = np.sqrt((x_upper-stopping_points_high)*2*abs(self.car.acceleration_limits[0]))
    stopping_points_low = np.linspace(x_lower, x_lower+0.5*v_lower**2/self.car.acceleration_limits[1])
    v_low = -np.sqrt((stopping_points_low-x_lower)*2*abs(self.car.acceleration_limits[1]))

    x_s = np.concatenate([stopping_points_low,[x_upper],np.flip(stopping_points_high),[x_lower,x_lower]])
    v_s = np.concatenate([v_low,[v_lower],np.flip(v_high),[v_upper,0]])
    ax.plot(x_s,v_s,color=c,linewidth=lw,zorder=zorder)


  def plot_formatting(self, ax=None, labels=None):
    """Formats the visualization.

    Args:
        ax (matplotlib.axes.Axes, optional): ax to plot.
        labels (list, optional): x- and y- labels. Defaults to None.
    """
    axStyle = self.get_axes()
    # == Formatting ==
    ax.axis(axStyle[0])
    ax.set_aspect(axStyle[1])  # makes equal aspect ratio
    ax.grid(False)
    if labels is not None:
      ax.set_xlabel(labels[0], fontsize=52)
      ax.set_ylabel(labels[1], fontsize=52)

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
    )
    ax.xaxis.set_major_locator(LinearLocator(5))
    ax.xaxis.set_major_formatter("{x:.1f}")
    ax.yaxis.set_major_locator(LinearLocator(5))
    ax.yaxis.set_major_formatter("{x:.1f}")
