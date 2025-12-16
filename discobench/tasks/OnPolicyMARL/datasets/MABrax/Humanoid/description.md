DESCRIPTION
A 3D humanoid robot with 21 degrees of freedom must learn to walk forward while maintaining balance and upright posture. The humanoid has a complex structure with torso, arms, and legs. The goal is to achieve stable bipedal locomotion.

MULTI AGENT ENVIRONMENT
This environment is multi-agent: one agent controls the upper body and one agent controls the lower body.

OBSERVATION SPACE
The observation is a ndarray with shape (376,) containing:

Joint angles and angular velocities of all 21 joints
Root position and orientation (3D position and quaternion)
Root linear and angular velocities
Center of mass information

Each agent's local observation vector is composed of the local state of the joints it controls, as well as the state of joints at distance 1 away in the body graph, and the state of the root body. State here refers to the position and velocity of the joint or body. All observations are continuous numbers in the range [-inf, inf].

ACTION SPACE
The combined action space of all agents consists of 17 continuous actions in the range [-0.4, 0.4]:

Torques applied to all joints including spine, arms, hips, knees, and ankles

Each agent's action space is the input torques to the joints it controls.

TRANSITION DYNAMICS
Full 3D physics simulation with complex multi-body dynamics
Multiple contact points with ground (feet, potentially other body parts)
Must maintain balance in 3D space
Coordinated movement of all limbs required

REWARD
Positive reward for forward velocity
Reward for maintaining upright posture
Small negative reward for energy expenditure (control cost)
Penalty for impact forces (falling)
Episode reward typically ranges from 0 to 5000+

STARTING STATE
Humanoid starts in upright standing position
Small random noise added to all joint positions and velocities
Center of mass positioned above support polygon
All agents receive the same joint reward.

EPISODE END
The episode ends if either of the following happens:

Termination: Humanoid falls over (torso height below threshold or extreme orientation)
Truncation: The length of the episode reaches max_steps (default: 1000)
