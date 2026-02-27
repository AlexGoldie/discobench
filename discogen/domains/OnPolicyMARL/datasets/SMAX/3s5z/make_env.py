from jaxmarl.environments.smax import HeuristicEnemySMAX, map_name_to_scenario
from jaxmarl.wrappers.baselines import SMAXLogWrapper


def make_env():
    env_kwargs = {"see_enemy_actions": True, "walls_cause_death": True, "attack_mode": "closest"}
    scenario = map_name_to_scenario("3s5z")
    env = HeuristicEnemySMAX(scenario=scenario, **env_kwargs)
    env = SMAXLogWrapper(env)
    return env
