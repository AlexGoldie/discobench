DESCRIPTION
In Brax Walker2d, a 2D bipedal walker robot must learn to walk forward while maintaining balance. The walker has 6 degrees of freedom controlling its legs, thighs, and torso. The goal is to achieve stable forward locomotion without falling over.

MULTI AGENT ENVIRONMENT
This environment is multi-agent: each agent controls one of the walker's two legs.

OBSERVATION SPACE
The combined observation of all agents is a ndarray with shape (17,) containing:

Joint angles and angular velocities of the 6 joints
Root height and orientation
Root linear and angular velocities
Contact forces with the ground

Each agent's local observation vector is composed of the local state of the joints it controls, as well as the state of joints at distance 1 away in the body graph, and the state of the root body. State here refers to the position and velocity of the joint or body. All observations are continuous numbers in the range [-inf, inf].

ACTION SPACE
The combined action space of all agents consists of 6 continuous actions in the range [-1, 1]:

Torques applied to thigh, leg, and foot joints for both legs

Each agent's action space is the input torques to the joints it controls.

TRANSITION DYNAMICS

2D physics simulation in the sagittal plane
Bipedal contact with ground through both feet
Must maintain balance to avoid falling
Forward locomotion through coordinated leg movements

REWARD

Positive reward for forward velocity
Small negative reward for energy expenditure (control cost)
Reward for staying alive (not falling)
Episode reward typically ranges from 0 to 5000+
All agents receive the same joint reward.

STARTING STATE

Walker starts in upright standing position
Small random noise added to initial joint angles and velocities
Both feet in contact with ground

EPISODE END
The episode ends if either of the following happens:

Termination: Walker falls over (torso height below threshold or extreme angle)
Truncation: The length of the episode reaches max_steps (default: 1000)
