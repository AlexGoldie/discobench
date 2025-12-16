DESCRIPTION
In Brax Ant, a quadrupedal ant robot with 8 degrees of freedom must learn to walk forward as quickly as possible. The ant consists of a torso with four legs, each having two joints (hip and ankle). The goal is to coordinate the leg movements to achieve stable and fast forward locomotion while maintaining balance.

MULTI AGENT ENVIRONMENT
This environment is multi-agent: one agent controls each of the ant's four legs.

OBSERVATION SPACE
The combined observation of all agents is a ndarray with shape (87,) containing:

Joint angles and angular velocities of the 8 joints
Position and orientation of the torso (x, y, z position and quaternion orientation)
Linear and angular velocities of the torso
Contact forces at the feet

Each agent's local observation vector is composed of the local state of the joints it controls, as well as the state of joints at distance 1 away in the body graph, and the state of the root body. State here refers to the position and velocity of the joint or body. All observations are continuous numbers in the range [-inf, inf].

ACTION SPACE
The combined action space consists of 8 continuous actions in the range [-1, 1]:

4 hip joint torques (one for each leg)
4 ankle joint torques (one for each leg)

Each agent's action space is the input torques to the joints it controls.

TRANSITION DYNAMICS

Physics simulation advances the ant's state based on applied torques
Contact forces are computed when feet touch the ground
The ant must maintain balance to avoid falling
Forward velocity is encouraged through the reward function

REWARD

Positive reward for forward velocity (x-direction)
Small negative reward for energy expenditure (control cost)
Negative reward for deviation from upright posture
Episode reward typically ranges from 0 to 6000+
All agents receive the same joint reward.

STARTING STATE

Ant starts in an upright position at the origin
Small random noise is added to initial joint positions and velocities
All joints begin near their neutral positions

EPISODE END
The episode ends if either of the following happens:

Termination: Ant falls over (torso height below threshold or extreme orientation)
Truncation: The length of the episode reaches max_steps (default: 1000)
