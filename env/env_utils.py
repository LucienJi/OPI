import importlib
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import random
from collections import deque
from typing import Optional, Tuple, Dict, Any, Literal

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from omegaconf import OmegaConf


def _is_missing_checkpoint(x) -> bool:
    return x is None or x == "None"

def _banner(msg: str):
    print("#" * 56)
    print(msg)
    print("#" * 56)

def build_obs_shape_from_obs_config(obs_config):
    obs_dim = 0
    for k in obs_config.keys():
        obs_dim += np.prod(obs_config[k])
    return (obs_dim,)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def _optional_dependency_error(module_name: str, feature_name: str, install_hint: str) -> ImportError:
    return ImportError(
        f"{feature_name} requires the optional dependency '{module_name}'. {install_hint}"
    )


def _import_optional_module(module_name: str, feature_name: str, install_hint: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise _optional_dependency_error(module_name, feature_name, install_hint) from exc


def _import_optional_attr(module_name: str, attr_name: str, feature_name: str, install_hint: str):
    module = _import_optional_module(module_name, feature_name, install_hint)
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise ImportError(
            f"{feature_name} expected '{attr_name}' inside module '{module_name}', but it was not found."
        ) from exc


def _require_dm_control_suite():
    return _import_optional_module(
        "dm_control.suite",
        "DeepMind Control environments",
        "Install the DMC dependencies (see README).",
    )


def _obs_dict_to_space(obs: Dict[str, np.ndarray]) -> spaces.Dict:
    box_map = {}
    for k, v in obs.items():
        arr = np.asarray(v)
        box_map[k] = spaces.Box(low=-np.inf, high=np.inf, shape=arr.shape, dtype=np.float32)
    return spaces.Dict(box_map)

def _cast_obs_float32(obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v, dtype=np.float32) for k, v in obs.items()}


class DMCGym(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        domain: str,
        task: str,
        *,
        n_sub_steps: int = 1,
        seed: Optional[int] = None,
        render_size: Tuple[int, int] = (240, 320),
        camera_id: int = 0,
    ):
        super().__init__()
        self.domain = domain
        self.task = task
        self._n_sub_steps = n_sub_steps
        self._step_count = 0

        self._render_h, self._render_w = render_size
        self._camera_id = camera_id

        self._last_seed = int(seed) if seed is not None else np.random.randint(0, 2**31 - 1)

        self._build_env(self._last_seed)

        a_spec = self.env.action_spec()
        self._a_low = a_spec.minimum.astype(np.float32)
        self._a_high = a_spec.maximum.astype(np.float32)
        self.action_space = spaces.Box(low=self._a_low, high=self._a_high, dtype=np.float32)

        ts0 = self.env.reset()
        self.observation_space = _obs_dict_to_space(ts0.observation)

    def _build_env(self, seed: int):
        suite = _require_dm_control_suite()
        self.env = suite.load(
            domain_name=self.domain,
            task_name=self.task,
            task_kwargs={"random": int(seed)},
        )
        self._max_steps = self.env._step_limit

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self._last_seed = int(seed)
            self._build_env(self._last_seed)

        self._step_count = 0
        ts = self.env.reset()
        obs = _cast_obs_float32(ts.observation)
        info = {"episode_seed": self._last_seed,"ts":ts}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(action, self._a_low, self._a_high).astype(np.float32)

        reward = 0.0
        ts = None
        ts = self.env.step(action)
        reward += float(ts.reward or 0.0)
        self._step_count += 1
        obs = _cast_obs_float32(ts.observation)

        terminated = bool(ts.last())
        truncated = (self._step_count >= self._max_steps)
        terminated = False if truncated else terminated
        return obs, reward, terminated, truncated, {'ts':ts}

    def render(self) -> np.ndarray:
        return self.env.physics.render(
            height=self._render_h, width=self._render_w, camera_id=self._camera_id
        )

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass

class DMCWrapper(gym.Wrapper):
    def __init__(self,env:DMCGym,obs_config):
        super().__init__(env)
        self.step_count = 0
        self.max_steps = self.unwrapped._max_steps
        self.obs_config = obs_config
        self.action_space = env.action_space
        self.env_category = 'dmc'

    def _parse_obs(self,obs):
        final_obs = []
        for k in self.obs_config.keys():
            final_obs.append(obs[k].reshape(-1))
        final_obs = np.concatenate(final_obs, axis=0)
        return final_obs

    def step(self,action):
        obs,reward,done,truncated,info = self.env.step(action)
        obs = self._parse_obs(obs)
        info['is_success'] = False
        return obs,reward,done,truncated,info
    def reset(self,seed=None,options=None):
        o,info =self.env.reset(seed=seed,options=options)
        o = self._parse_obs(o)
        return o,info
    def close(self):
        self.unwrapped.close()
    def render(self):
        return self.unwrapped.render()


def make_dmc_env(domain,task,obs_config, seed=0,render_size = (128,128),n_sub_steps=1):
    env = DMCGym(domain,task,n_sub_steps=n_sub_steps,seed=seed,render_size=render_size)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = DMCWrapper(env,obs_config)
    return env


def make_metaworld_env(env_name,obs_config,seed=0,camera_to_render='corner4',reward_scale = 1.0, step_penality = 0.01,max_steps = 400,sparse_reward = False):
    _import_optional_module(
        "metaworld",
        "Metaworld environments",
        "Install the optional Metaworld dependency (see README).",
    )
    env = gym.make('Meta-World/MT1', env_name=env_name, seed=seed,render_mode="rgb_array",camera_name=camera_to_render)
    env = MetaworldWrapper(env,obs_config,reward_scale,step_penality,max_steps,sparse_reward)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

class MetaworldWrapper(gym.Wrapper):
    def __init__(self,env,obs_config,reward_scale = 0.1, step_penality = 1.0,max_steps = 500,sparse_reward = False):
        super().__init__(env)
        self.step_count = 0
        self.obs_config = obs_config
        self.action_space = env.action_space
        self.reward_scale = reward_scale
        self.step_penality = step_penality
        self.env_category = 'metaworld'
        self.max_steps = max_steps
        self.sparse_reward = sparse_reward

    def _parse_obs(self,obs):
        final_obs = obs
        return final_obs

    def step(self,action):
        self.step_count += 1
        obs,raw_reward,done,truncated,info = self.env.step(action)
        if 'episode' in info:
            info.pop('episode')
        info['is_success'] = info['success'] > 0.0

        if not info['success']:
            if self.prev_raw_reward is not None:
                reward = (raw_reward - self.prev_raw_reward) * self.reward_scale  - self.step_penality
                self.prev_raw_reward = raw_reward
            else:
                reward = 0.0
                self.prev_raw_reward = raw_reward
            if self.sparse_reward:
                reward = 0.0
        else:
            reward = min(raw_reward , 5)

        obs = self._parse_obs(obs)
        done = done or info['success'] > 0.0
        truncated = truncated or self.step_count >= self.max_steps

        return obs,reward,done,truncated,info
    def reset(self,seed=None,options=None):
        self.prev_raw_reward = None
        self.step_count = 0
        obs,info = self.env.reset()
        obs = self._parse_obs(obs)
        return obs,info
    def close(self):
        self.env.close()
    def render(self):
        image = self.env.render()
        return image[::-1, :, :]


def make_myosuite_env(env_name,obs_config,seed=0,is_sparse_reward = False,camera_id_render = 4, camera_size = (256,256)):
    from myosuite.utils import gym as myosuite_gym
    env = myosuite_gym.make(env_name,seed=seed)
    env = MyoSuiteWrapper(env,obs_config,is_sparse_reward,camera_id_render, camera_size)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

class MyoSuiteWrapper(gym.Wrapper):
    def __init__(self,env,obs_config,is_sparse_reward = False,camera_id_render = 4, camera_size = (256,256)):
        super().__init__(env)
        self.obs_config = obs_config
        self.action_space = env.action_space
        self.env_category = 'myosuite'
        self.is_sparse_reward = is_sparse_reward
        self.camera_id_render = camera_id_render
        self.camera_size = camera_size
        mj_renderer_cls = _import_optional_attr(
            "myosuite.renderer.mj_renderer",
            "MJRenderer",
            "MyoSuite rendering",
            "Install the optional MyoSuite dependency before running MyoSuite experiments.",
        )
        self.renderer = mj_renderer_cls(env.unwrapped.sim)

    def _parse_obs(self,obs):
        final_obs = []
        for k in self.obs_config.keys():
            final_obs.append(obs[k])
        final_obs = np.concatenate(final_obs, axis=0)
        return final_obs
    def step(self,action):
        obs,reward,done,truncated,info = self.env.step(action)
        if self.is_sparse_reward:
            reward = info['rwd_sparse']
        else:
            reward = info['rwd_dense']
        obs = self._parse_obs(obs)
        done = done or info['solved']
        done = bool(done)
        truncated = bool(truncated)
        info['is_success'] = info['solved']

        return obs,reward,done,truncated,info
    def reset(self,seed=None,options=None):
        obs,info = self.env.reset()
        obs = self._parse_obs(obs)
        return obs,info
    def close(self):
        self.env.close()

    def render(self):
        image = self.renderer.render_offscreen(width=self.camera_size[0], height=self.camera_size[1],camera_id=self.camera_id_render)
        return image


class HighwayEnvWrapper(gym.Wrapper):
    def __init__(self,env,obs_config,flatten_obs = False):
        super().__init__(env)
        self.flatten_obs = flatten_obs
        self.obs_config = obs_config
        self.action_space = env.action_space
        self.env_category = 'highway'
        self.max_steps = int(self.unwrapped.config['duration'] *  self.unwrapped.config["policy_frequency"])
        print("MAX STEPS", self.max_steps)
    def step(self,action):
        obs,reward,done,truncated,info = self.env.step(action)
        info['is_success'] = info['rewards']['goal'] > 0.0
        obs_dict = {}
        obs_dict['robot_states'] = obs if not self.flatten_obs else obs.reshape(-1)
        return obs_dict,reward,done,truncated,info
    def reset(self,seed=None,options=None):
        obs,info = self.env.reset(seed=seed,options=options)
        obs_dict = {}
        obs_dict['robot_states'] = obs if not self.flatten_obs else obs.reshape(-1)
        return obs_dict,info
    def close(self):
        self.env.close()
    def render(self):
        return self.env.render()


class HighwayWeatherWrapper(gym.Wrapper):
    CLEAR = 0
    RAINY = 1
    FOGGY = 2
    SNOWY = 3
    WEATHER_TYPES = [CLEAR, RAINY, FOGGY, SNOWY]
    DURATION_RANGE = {
        CLEAR: (100, 200),
        RAINY: (50, 150),
        FOGGY: (50, 150),
        SNOWY: (30, 100),
    }
    WEATHER_COLORS = {
        CLEAR: (0, 255, 0),
        RAINY: (255, 0, 0),
        FOGGY: (192, 192, 192),
        SNOWY: (255, 255, 255)
    }
    WEATHER_NAMES = {
        CLEAR: 'CLEAR',
        RAINY: 'RAINY',
        FOGGY: 'FOGGY',
        SNOWY: 'SNOWY'
    }
    def __init__(self,env,weather_probs = None):
        super().__init__(env)
        self.action_space = env.action_space
        self.env_category = 'highway'
        self.max_steps = int(self.unwrapped.config['duration'] *  self.unwrapped.config["policy_frequency"])
        self._cv2 = _import_optional_module(
            "cv2",
            "Highway weather rendering",
            "Install opencv-python before rendering Highway weather environments.",
        )
        print("MAX STEPS", self.max_steps)
        self.current_weather = self.CLEAR
        if weather_probs is None:
            self.weather_probs = [1.0, 0.0, 0.0, 0.0]
        else:
            self.weather_probs = np.array(weather_probs)
            self.weather_probs = self.weather_probs / self.weather_probs.sum()

    def _parse_obs(self,obs):
        weather_type_one_hot = np.zeros(len(self.WEATHER_TYPES))
        weather_type_one_hot[self.current_weather] = 1
        obs = np.concatenate([weather_type_one_hot, obs.reshape(-1)], axis=0)
        return obs


    def step(self,action):
        if self.steps_remaining <= 0:
            self._sample_new_weather()
            self._set_perception_distance(self.current_weather)
        obs,reward,done,truncated,info = self.env.step(action)
        self._set_friction(self.current_weather)
        self.steps_remaining -= 1
        if 'goal' in info['rewards']:
            info['is_success'] = info['rewards']['goal'] > 0.0
        else:
            info['is_success'] = False

        info['current_weather'] = self.current_weather
        obs = self._parse_obs(obs)
        return obs,reward,done,truncated,info
    def reset(self,seed=None,options=None):
        self._sample_new_weather()
        self._set_perception_distance(self.current_weather)
        obs,info = self.env.reset(seed=seed,options=options)
        obs = self._parse_obs(obs)
        self._set_friction(self.current_weather)

        obs = obs.reshape(-1)
        return obs,info
    def close(self):
        self.env.close()
    def render(self):
        frame = self.env.render()
        frame = np.ascontiguousarray(frame)
        weather_name = self.WEATHER_NAMES[self.current_weather]
        text_main = f"Weather: {weather_name}"
        text_sub = f"Left: {self.steps_remaining}"
        color = self.WEATHER_COLORS[self.current_weather]
        self._cv2.putText(frame, text_main, (10, 15),
                    self._cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, self._cv2.LINE_AA)

        self._cv2.putText(frame, text_sub, (10, 28),
                        self._cv2.FONT_HERSHEY_SIMPLEX, 0.25, (220, 220, 220), 1, self._cv2.LINE_AA)
        return frame

    def _set_perception_distance(self,weather_type:int):
        self.current_weather = weather_type
        if weather_type == self.RAINY:
            self.env.unwrapped.PERCEPTION_DISTANCE = 200
        elif weather_type == self.FOGGY:
            self.env.unwrapped.PERCEPTION_DISTANCE = 100
        elif weather_type == self.SNOWY:
            self.env.unwrapped.PERCEPTION_DISTANCE = 100
        elif weather_type == self.CLEAR:
            self.env.unwrapped.PERCEPTION_DISTANCE = 200
        else:
            raise ValueError(f"Invalid weather type: {weather_type}")

    def _set_friction(self,weather_type:int):
        if weather_type == self.RAINY:
            self.env.unwrapped.controlled_vehicles[0].FRICTION_REAR = 5
            self.env.unwrapped.controlled_vehicles[0].FRICTION_FRONT = 4
        elif weather_type == self.FOGGY:
            self.env.unwrapped.controlled_vehicles[0].FRICTION_REAR = 15
            self.env.unwrapped.controlled_vehicles[0].FRICTION_FRONT = 15
        elif weather_type == self.SNOWY:
            self.env.unwrapped.controlled_vehicles[0].FRICTION_REAR = 3
            self.env.unwrapped.controlled_vehicles[0].FRICTION_FRONT = 4
        elif weather_type == self.CLEAR:
            self.env.unwrapped.controlled_vehicles[0].FRICTION_REAR = 15
            self.env.unwrapped.controlled_vehicles[0].FRICTION_FRONT = 15
        else:
            raise ValueError(f"Invalid weather type: {weather_type}")

    def _sample_new_weather(self):
        self.current_weather = np.random.choice(self.WEATHER_TYPES, p=self.weather_probs)
        t_min, t_max = self.DURATION_RANGE[self.current_weather]
        self.steps_remaining = np.random.randint(t_min, t_max + 1)


def make_highway_env(env_name,obs_config,seed=0, env_config = None,render_mode = 'rgb_array', flatten_obs = False):
    _import_optional_module(
        "highway_env",
        "Highway environments",
        "Install the optional HighwayEnv dependency before running highway experiments.",
    )
    env = gym.make(env_name, config=env_config,render_mode=render_mode)
    env = HighwayEnvWrapper(env,obs_config,flatten_obs)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

def make_highway_weather(
    env_name,
    weather_probs,
    env_config,
    seed=0
):
    _import_optional_module(
        "highway_env",
        "Highway weather environments",
        "Install the optional HighwayEnv dependency before running highway experiments.",
    )
    env = gym.make(env_name, render_mode="rgb_array", config=env_config)
    env = HighwayWeatherWrapper(env,weather_probs)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

class HybridWalkerWrapper(gym.Wrapper):
    def __init__(self, env, stack_frames=4,max_steps=1000):
        super().__init__(env)
        self.step_count = 0
        self.stack_frames = stack_frames
        self.max_steps = max_steps

        self.frames = deque(maxlen=stack_frames)

        self.action_space = env.action_space

        src_low = self.env.observation_space.low
        src_high = self.env.observation_space.high

        stacked_low = np.tile(src_low, stack_frames)
        stacked_high = np.tile(src_high, stack_frames)

        self.observation_space = gym.spaces.Box(
            low=stacked_low,
            high=stacked_high,
            dtype=self.env.observation_space.dtype
        )

        self.env_category = 'box2d_hybrid'

    def _get_flattened_obs(self):
        return np.concatenate(list(self.frames), axis=0)

    def reset(self, seed=None, options=None):
        self.step_count = 0

        obs, info = self.env.reset(seed=seed, options=options)

        for _ in range(self.stack_frames):
            self.frames.append(obs)

        final_obs = self._get_flattened_obs()
        return final_obs, info

    def step(self, action):
        self.step_count += 1

        obs, reward, terminated, truncated, info = self.env.step(action)

        self.frames.append(obs)

        final_obs = self._get_flattened_obs()
        if self.step_count >= self.max_steps:
            terminated = True
            truncated = True

        return final_obs, reward, terminated, truncated, info

    def close(self):
        self.unwrapped.close()

    def render(self):
        return self.unwrapped.render()


def make_hybrid_walker(mode="learner",stack_frames=4,max_steps=1000):
    hybrid_walker_cls = _import_optional_attr(
        "third_party.customized_box2d.HybridBipedalWalker",
        "HybridBipedalWalker",
        "Box2D hybrid walker environments",
        "Install the Box2D dependencies and keep third_party/customized_box2d available before running Box2D experiments.",
    )
    env = hybrid_walker_cls(mode=mode,render_mode="rgb_array")
    env = HybridWalkerWrapper(env,stack_frames=stack_frames,max_steps=max_steps)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

class HybridLanderWrapper(gym.Wrapper):
    def __init__(self, env, stack_frames=4,max_steps=1000):
        super().__init__(env)
        self.stack_frames = stack_frames
        self.max_steps = max_steps
        self.frames = deque(maxlen=stack_frames)

        self.action_space = env.action_space

        src_low = self.env.observation_space.low
        src_high = self.env.observation_space.high

        stacked_low = np.tile(src_low, stack_frames)
        stacked_high = np.tile(src_high, stack_frames)

        self.observation_space = gym.spaces.Box(
            low=stacked_low,
            high=stacked_high,
            dtype=self.env.observation_space.dtype
        )

        self.env_category = 'box2d_hybrid_lander'

    def _get_flattened_obs(self):
        return np.concatenate(list(self.frames), axis=0)

    def reset(self, seed=None, options=None):
        self.step_count = 0
        obs, info = self.env.reset(seed=seed, options=options)

        for _ in range(self.stack_frames):
            self.frames.append(obs)

        return self._get_flattened_obs(), info

    def step(self, action):
        self.step_count += 1
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.frames.append(obs)

        final_obs = self._get_flattened_obs()

        info['is_success'] = False
        if terminated and reward == 100:
             info['is_success'] = True
        if self.step_count >= self.max_steps:
            terminated = True
            truncated = True
        return final_obs, reward, terminated, truncated, info

    def render(self):
        return self.unwrapped.render()

    def close(self):
        self.unwrapped.close()
def make_hybrid_lander(mode="learner",stack_frames=4,max_steps=1000):
    hybrid_lander_cls = _import_optional_attr(
        "third_party.customized_box2d.HybridLunarLand",
        "HybridLunarLander",
        "Box2D hybrid lander environments",
        "Install the Box2D dependencies and keep third_party/customized_box2d available before running Box2D experiments.",
    )
    env = hybrid_lander_cls(mode=mode,render_mode="rgb_array")
    env = HybridLanderWrapper(env,stack_frames=stack_frames)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env


class HybridCarVectorWrapper(gym.Wrapper):
    def __init__(self, env, stack_frames=4,max_steps=1000):
        super().__init__(env)
        self.stack_frames = stack_frames
        self.frames = deque(maxlen=stack_frames)
        self.max_steps = max_steps
        base_dim = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(base_dim * stack_frames,),
            dtype=np.float32
        )
        self.env_category = 'box2d_hybrid_car_vector'

    def _get_flattened_obs(self):
        return np.concatenate(list(self.frames), axis=0)

    def reset(self, seed=None, options=None):
        self.step_count = 0
        obs, info = self.env.reset(seed=seed, options=options)
        for _ in range(self.stack_frames):
            self.frames.append(obs)
        return self._get_flattened_obs(), info

    def step(self, action):
        self.step_count += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_flattened_obs(), reward, terminated, truncated, info

def make_hybrid_car_racing(mode="learner",stack_frames=4,max_steps=1000):
    hybrid_car_cls = _import_optional_attr(
        "third_party.customized_box2d.HybridCar",
        "HybridCarRacing",
        "Box2D hybrid car racing environments",
        "Install the Box2D dependencies and keep third_party/customized_box2d available before running Box2D experiments.",
    )
    env = hybrid_car_cls(mode=mode,render_mode="rgb_array")
    env = HybridCarVectorWrapper(env,stack_frames=stack_frames,max_steps=max_steps)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

class HybridWeatherVectorWrapper(gym.Wrapper):
    def __init__(self, env, stack_frames=4,max_steps=2000):
        super().__init__(env)
        self.stack_frames = stack_frames
        self.frames = deque(maxlen=stack_frames)
        self.max_steps = max_steps
        base_dim = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(base_dim * stack_frames,),
            dtype=np.float32,
        )
        self.env_category = 'box2d_hybrid_weather_vector'
    def _get_flattened_obs(self):
        return np.concatenate(list(self.frames), axis=0)

    def reset(self, seed=None, options=None):
        self.step_count = 0
        obs, info = self.env.reset(seed=seed, options=options)
        for _ in range(self.stack_frames):
            self.frames.append(obs)
        return self._get_flattened_obs(), info

    def step(self, action):
        self.step_count += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_flattened_obs(), reward, terminated, truncated, info

    def render(self):
        return self.unwrapped.render()

    def close(self):
        self.unwrapped.close()
def make_hybrid_weather(mode="learner",stack_frames=4,max_steps=2000):
    hybrid_weather_cls = _import_optional_attr(
        "third_party.customized_box2d.HybridWeather",
        "HybridWeather",
        "Box2D hybrid weather environments",
        "Install the Box2D dependencies and keep third_party/customized_box2d available before running Box2D experiments.",
    )
    assert mode in ["learner", "Sunny", "Rainy","Foggy"]
    env = hybrid_weather_cls(mode=mode,render_mode="rgb_array")
    env = HybridWeatherVectorWrapper(env,stack_frames=stack_frames,max_steps=max_steps)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

def _get_hydra_config_name() -> Optional[str]:
    try:
        from hydra.core.hydra_config import HydraConfig

        cfg = HydraConfig.get()
        name = getattr(getattr(cfg, "job", None), "config_name", None)
        return str(name) if name else None
    except Exception:
        return None


def _normalize_obs_config(obs_config: Any) -> Dict[str, Tuple[int, ...]]:
    if obs_config is None:
        return {}

    try:
        obs_config = OmegaConf.to_container(obs_config, resolve=True)
    except Exception:
        pass

    if obs_config is None:
        return {}
    if not isinstance(obs_config, dict):
        raise TypeError(f"obs_config must be a dict-like object, got {type(obs_config)}")

    out: Dict[str, Tuple[int, ...]] = {}
    for k, v in obs_config.items():
        if v is None:
            out[str(k)] = tuple()
        elif isinstance(v, tuple):
            out[str(k)] = tuple(int(x) for x in v)
        elif isinstance(v, list):
            out[str(k)] = tuple(int(x) for x in v)
        else:
            out[str(k)] = (int(v),)
    return out


def _infer_env_category_from_config_name_or_cfg(
    config_name: Optional[str],
    cfg: Any,
) -> Literal["dmc", "metaworld", "myosuite", "highway", "weather", "box2d"]:
    if config_name:
        prefix = config_name.split("/", 1)[0].strip().lower()
        if prefix in ("dmc", "metaworld", "myosuite", "box2d"):
            return prefix
        lowered = config_name.lower()
        for p in ("dmc", "metaworld", "myosuite", "box2d"):
            if f"/{p}/" in lowered or lowered.startswith(f"{p}_") or lowered.startswith(f"{p}/"):
                return p

    env_cfg = getattr(cfg, "env", None)
    if env_cfg is not None:
        if getattr(env_cfg, "domain_name", None) is not None and getattr(env_cfg, "task_name", None) is not None:
            return "dmc"
        if getattr(env_cfg, "box2d_env_name", None) is not None or getattr(env_cfg, "is_box2d", None):
            return "box2d"
        if getattr(env_cfg, "env_name", None) in ("bipedal", "lander", "racingcar", "car_racing", "weather", "hybrid_weather"):
            return "box2d"
        if getattr(env_cfg, "mode", None) is not None and getattr(env_cfg, "stack_frames", None) is not None:
            return "box2d"
        if getattr(env_cfg, "env_name", None) is not None:
            return "metaworld"
        if getattr(env_cfg, "myosuite_env_name", None) is not None or getattr(env_cfg, "is_myosuite", None):
            return "myosuite"

    raise ValueError(
        f"Could not infer env category from config_name={config_name!r} or cfg.env keys. "
        f"Please name configs under dmc/ or metaworld/ (etc), or add recognizable keys to cfg.env."
    )


def infer_env_category(cfg: Any, *, config_name: Optional[str] = None) -> str:
    if config_name is None:
        config_name = _get_hydra_config_name()
    return _infer_env_category_from_config_name_or_cfg(config_name, cfg)


def make_env_and_eval_env_from_cfg(
    cfg: Any,
    *,
    obs_config: Optional[Any] = None,
    config_name: Optional[str] = None,
):
    if config_name is None:
        config_name = _get_hydra_config_name()

    env_category = _infer_env_category_from_config_name_or_cfg(config_name, cfg)
    obs_config_norm = _normalize_obs_config(obs_config if obs_config is not None else cfg.get("obs_config", {}))

    env_cfg = cfg.env
    seed = getattr(env_cfg, "seed", 0)

    if env_category == "dmc":
        render_size = getattr(env_cfg, "render_size", (128, 128))
        try:
            render_size = tuple(render_size)
        except Exception:
            pass
        n_sub_steps = getattr(env_cfg, "n_sub_steps", 1)

        env = make_dmc_env(
            domain=env_cfg.domain_name,
            task=env_cfg.task_name,
            obs_config=obs_config_norm,
            seed=seed,
            render_size=render_size,
            n_sub_steps=n_sub_steps,
        )
        eval_env = make_dmc_env(
            domain=env_cfg.domain_name,
            task=env_cfg.task_name,
            obs_config=obs_config_norm,
            seed=seed,
            render_size=render_size,
            n_sub_steps=n_sub_steps,
        )
        return env, eval_env, obs_config_norm

    if env_category == "metaworld":
        env = make_metaworld_env(
            env_name=env_cfg.env_name,
            obs_config=obs_config_norm,
            seed=seed,
            camera_to_render=getattr(env_cfg, "camera_to_render", "corner4"),
            reward_scale= env_cfg.reward_scale ,
            step_penality= env_cfg.step_penality,
            sparse_reward = env_cfg.sparse_reward
        )
        eval_env = make_metaworld_env(
            env_name=env_cfg.env_name,
            obs_config=obs_config_norm,
            seed=seed,
            camera_to_render=getattr(env_cfg, "camera_to_render", "corner4"),
            reward_scale=env_cfg.reward_scale,
            step_penality=env_cfg.step_penality,
            sparse_reward = env_cfg.sparse_reward
        )
        return env, eval_env, obs_config_norm

    if env_category == "myosuite":
        env_name = getattr(env_cfg, "env_name", None) or getattr(env_cfg, "myosuite_env_name", None)
        if env_name is None:
            raise ValueError("myosuite env requires cfg.env.env_name (or cfg.env.myosuite_env_name)")
        env = make_myosuite_env(
            env_name=env_name,
            obs_config=obs_config_norm,
            seed=seed,
            is_sparse_reward=getattr(env_cfg, "is_sparse_reward", False),
            camera_id_render=getattr(env_cfg, "camera_id_render", 4),
            camera_size=getattr(env_cfg, "camera_size", (256, 256)),
        )
        eval_env = make_myosuite_env(
            env_name=env_name,
            obs_config=obs_config_norm,
            seed=seed,
            is_sparse_reward=getattr(env_cfg, "is_sparse_reward", False),
            camera_id_render=getattr(env_cfg, "camera_id_render", 4),
            camera_size=getattr(env_cfg, "camera_size", (256, 256)),
        )
        return env, eval_env, obs_config_norm

    if env_category == "box2d":
        env_name = getattr(env_cfg, "env_name", None) or getattr(env_cfg, "box2d_env_name", None)
        if env_name is None:
            raise ValueError("box2d env requires cfg.env.env_name (or cfg.env.box2d_env_name)")
        mode = getattr(env_cfg, "mode", "learner")
        stack_frames = getattr(env_cfg, "stack_frames", 4)
        max_steps = getattr(env_cfg, "max_steps", 1000)
        env_key = str(env_name).lower()

        if env_key in ("bipedal", "walker", "hybrid_walker"):
            env = make_hybrid_walker(mode=mode, stack_frames=stack_frames, max_steps=max_steps)
            eval_env = make_hybrid_walker(mode=mode, stack_frames=stack_frames, max_steps=max_steps)
            return env, eval_env, obs_config_norm
        if env_key in ("lander", "hybrid_lander"):
            env = make_hybrid_lander(mode=mode, stack_frames=stack_frames, max_steps=max_steps)
            eval_env = make_hybrid_lander(mode=mode, stack_frames=stack_frames, max_steps=max_steps)
            return env, eval_env, obs_config_norm
        if env_key in ("racingcar", "car_racing", "hybrid_car_racing"):
            env = make_hybrid_car_racing(mode=mode, stack_frames=stack_frames, max_steps=max_steps)
            eval_env = make_hybrid_car_racing(mode=mode, stack_frames=stack_frames, max_steps=max_steps)
            return env, eval_env, obs_config_norm
        if env_key in ("weather", "hybrid_weather", "hybrid_weather_car"):
            env = make_hybrid_weather(mode=mode, stack_frames=stack_frames, max_steps=max_steps)
            eval_env = make_hybrid_weather(mode=mode, stack_frames=stack_frames, max_steps=max_steps)
            return env, eval_env, obs_config_norm

        raise ValueError(f"Unsupported box2d env_name: {env_name}")

    raise ValueError(f"Unsupported env category: {env_category}")
