"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu        ( kaichieh@princeton.edu )

This module implements the parent class for the Dubins car environments, e.g.,
one car environment and pursuit-evasion game between two Dubins cars.
"""

import numpy as np
from .env_utils import calculate_margin_circle, calculate_margin_rect


class DoubleIntegratorDyn(object):
  """
  This base class implements a Dubins car dynamical system as well as the
  environment with concentric circles. The inner circle is the target set
  boundary, while the outer circle is the boundary of the constraint set.
  """

  def __init__(self, doneType='toEnd'):
    """Initializes the environment with the episode termination criterion.

    Args:
        doneType (str, optional): conditions to raise `done` flag in
            training. Defaults to 'toEnd'.
    """
    # State bounds.
    self.bounds = np.array([[-1.1, 1.1], [-1.1, 1.1]])
    self.low = self.bounds[:, 0]
    self.high = self.bounds[:, 1]

    # Dubins car parameters.
    self.alive = True
    self.time_step = 0.05

    # Control parameters.
    self.acceleration_limits = [-1,1]
    self.discrete_controls = np.array([
        self.acceleration_limits[0], 0., self.acceleration_limits[1]
    ])

    # Constraint set parameters.
    self.constraint_x_y_w_h= None

    # Target set parameters.
    self.target_x_y_w_h = None

    # Internal state.
    self.state = np.zeros(2)
    self.doneType = doneType

    # Set random seed.
    self.seed_val = 0
    np.random.seed(self.seed_val)

    # Cost Params
    self.targetScaling = 1.
    self.safetyScaling = 1.

  def reset(
      self, start=None, sample_inside_obs=False,
      sample_inside_tar=True
  ):
    """Resets the state of the environment.

    Args:
        start (np.ndarray, optional): the state to reset the Dubins car to. If
            None, pick the state uniformly at random. Defaults to None.

        sample_inside_obs (bool, optional): consider sampling the state inside
            the obstacles if True. Defaults to False.
        sample_inside_tar (bool, optional): consider sampling the state inside
            the target if True. Defaults to True.

    Returns:
        np.ndarray: the state that Dubins car has been reset to.
    """
    if start is None:
      x_rnd, y_rnd = self.sample_random_state(
          sample_inside_obs=sample_inside_obs,
          sample_inside_tar=sample_inside_tar
      )
      self.state = np.array([x_rnd, y_rnd])
    else:
      self.state = start
    return np.copy(self.state)

  def sample_random_state(
      self, sample_inside_obs=False, sample_inside_tar=True,
  ):
    """Picks the state uniformly at random.

    Args:
        sample_inside_obs (bool, optional): consider sampling the state inside
            the obstacles if True. Defaults to False.
        sample_inside_tar (bool, optional): consider sampling the state inside
            the target if True. Defaults to True.

    Returns:
        np.ndarray: the sampled initial state.
    """

    # random sample [`x`, `y`]
    flag = True
    while flag:
      rnd_state = np.random.uniform(low=self.low[:2], high=self.high[:2])
      l_x = self.target_margin(rnd_state)
      g_x = self.safety_margin(rnd_state)

      if (not sample_inside_obs) and (g_x > 0):
        flag = True
      elif (not sample_inside_tar) and (l_x <= 0):
        flag = True
      else:
        flag = False
    x_rnd, y_rnd = rnd_state

    return x_rnd, y_rnd

  # == Dynamics ==
  def step(self, action):
    """Evolves the environment one step forward given an action.

    Args:
        action (int): the index of the action in the action set.

    Returns:
        np.ndarray: next state.
        bool: True if the episode is terminated.
    """
    l_x_cur = self.target_margin(self.state[:2])
    g_x_cur = self.safety_margin(self.state[:2])

    u = self.discrete_controls[action]
    state = self.integrate_forward(self.state, u)
    self.state = state

    # done
    if self.doneType == 'toEnd':
      done = not self.check_within_bounds(self.state)
    else:
      assert self.doneType == 'TF', 'invalid doneType'
      fail = g_x_cur > 0
      success = l_x_cur <= 0
      done = fail or success

    if done:
      self.alive = False

    return np.copy(self.state), done

  def integrate_forward(self, state, u):
    """Integrates the dynamics forward by one step.

    Args:
        state (np.ndarray): (x, y, yaw).
        u (float): the contol input, angular speed.

    Returns:
        np.ndarray: next state.
    """
    x, v = state

    x = x + self.time_step * v + 0.5*u*self.time_step**2
    y = v + self.time_step * u

    state = np.array([x, y])
    return state

  # == Setting Hyper-Parameter Functions ==
  def set_bounds(self, bounds):
    """Sets the boundary of the environment.

    Args:
        bounds (np.ndarray): of the shape (n_dim, 2). Each row is [LB, UB].
    """
    self.bounds = bounds

    # Get lower and upper bounds
    self.low = np.array(self.bounds)[:, 0]
    self.high = np.array(self.bounds)[:, 1]


  def set_time_step(self, time_step=.05):
    """Sets the time step for dynamics integration.

    Args:
        time_step (float, optional): time step used in the integrate_forward.
            Defaults to .05.
    """
    self.time_step = time_step

  def set_constraint(self, x,y,w,h):
    """Sets the constraint set (complement of failure set).

    Args:
        center (np.ndarray, optional): center of the constraint set.
        radius (float, optional): radius of the constraint set.
    """
    self.constraint_x_y_w_h = np.array([x,y,w,h])

  def set_target(self, x,y,w,h):
    """Sets the target set.

    Args:
        center (np.ndarray, optional): center of the target set.
        radius (float, optional): radius of the target set.
    """
    self.target_x_y_w_h = np.array([x,y,w,h])

  # == Getting Functions ==
  def check_within_bounds(self, state):
    """Checks if the agent is still in the environment.

    Args:
        state (np.ndarray): the state of the agent.

    Returns:
        bool: False if the agent is not in the environment.
    """
    for dim, bound in enumerate(self.bounds):
      flagLow = state[dim] < bound[0]
      flagHigh = state[dim] > bound[1]
      if flagLow or flagHigh:
        return False
    return True

  # == Compute Margin ==
  def safety_margin(self, s):
    """Computes the margin (e.g. distance) between the state and the failue set.

    Args:
        s (np.ndarray): the state of the agent.

    Returns:
        float: postivive numbers indicate being inside the failure set (safety
            violation).
    """
    x, y = (self.low + self.high)[:2] / 2.0
    w, h = (self.high - self.low)[:2]
    boundary_margin = calculate_margin_rect(
        s, [x, y, w, h], negativeInside=True
    )
    g_xList = [boundary_margin]

    if self.constraint_x_y_w_h is not None:
      g_x = calculate_margin_rect(
          s, self.constraint_x_y_w_h,
          negativeInside=True
      )
      g_xList.append(g_x)

    safety_margin = np.max(np.array(g_xList))
    return self.safetyScaling * safety_margin

  def target_margin(self, s):
    """Computes the margin (e.g. distance) between the state and the target set.

    Args:
        s (np.ndarray): the state of the agent.

    Returns:
        float: negative numbers indicate reaching the target. If the target set
            is not specified, return None.
    """
    if self.target_x_y_w_h is not None:
      target_margin = calculate_margin_rect(
          s, self.target_x_y_w_h, negativeInside=True
      )
      return self.targetScaling * target_margin
    else:
      return None
