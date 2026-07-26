import cv2
import numpy as np
import gym
import gymnasium as gym_ms
import mani_skill.envs  
from mani_skill.utils.structs.pose import Pose

class PushCubeWrapper(gym.Env):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self.ms_env = gym_ms.make(
            "PushCube-v1", 
            obs_mode="rgb", 
            control_mode="pd_joint_pos", 
            render_mode="rgb_array"
        )
        
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        # FIX: Torniamo al formato immagine standard (0-255, H, W, C)
        self.observation_space = gym.spaces.Dict({
            "visual": gym.spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
            "proprio": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        })
        
        self.last_success = False

    def _process_obs(self, ms_obs):
        sensor_data = ms_obs.get('sensor_data', {})
        cam_name = list(sensor_data.keys())[0] if len(sensor_data) > 0 else None
        
        if cam_name and 'rgb' in sensor_data[cam_name]:
            rgb = sensor_data[cam_name]['rgb']
            if hasattr(rgb, 'cpu'):
                rgb = rgb.cpu().numpy()
            if rgb.ndim == 4:
                rgb = rgb[0]
                
            # FIX: Solo resize. Niente trasposizione e niente divisione per 255. Ci pensa DINO-WM.
            rgb_out = cv2.resize(rgb, (224, 224)).astype(np.uint8)
        else:
            rgb_out = np.zeros((224, 224, 3), dtype=np.uint8)

        qpos = np.zeros(9)
        if 'agent' in ms_obs and 'qpos' in ms_obs['agent']:
            qpos = ms_obs['agent']['qpos']
        elif 'state' in ms_obs:
            qpos = ms_obs['state']

        if hasattr(qpos, 'cpu'):
            qpos = qpos.cpu().numpy()
        if qpos.ndim == 2:
            qpos = qpos[0]
            
        state = np.zeros(9, dtype=np.float32)
        length = min(len(qpos), 9)
        state[:length] = qpos[:length]

        obs_dict = {
            "visual": rgb_out,
            "proprio": state.astype(np.float32)
        }

        return obs_dict, state.astype(np.float32)

    def reset(self, seed=None):
        if isinstance(seed, list):
            seed = seed[0] 
        ms_obs, info = self.ms_env.reset(seed=seed)
        self.last_success = False
        return self._process_obs(ms_obs)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        n_sub = action.shape[-1] // self.action_space.shape[0]   # es. 40 // 8 = 5
        sub_actions = action.reshape(n_sub, self.action_space.shape[0])  # VERIFICA L'ORDINE, vedi sotto

        reward_acc = 0.0
        for sub_act in sub_actions:
            ms_obs, reward, terminated, truncated, info = self.ms_env.step(sub_act)
            reward_acc += reward
            if info.get('success', False):
                self.last_success = True
            if terminated or truncated:
                break

        obs, state = self._process_obs(ms_obs)
        done = terminated or truncated
        info['state_aggiornato'] = state
        return obs, reward_acc, done, info

    def rollout(self, seed, init_state, actions):
        obs, state = self.prepare(seed, init_state)
        
        obses = {
            "visual": [obs["visual"]],
            "proprio": [obs["proprio"]]
        }
        states = [state]
        
        for act in actions:
            obs, reward, done, info = self.step(act)
            state = info['state_aggiornato']
            
            obses["visual"].append(obs["visual"])
            obses["proprio"].append(obs["proprio"])
            states.append(state)
            
        obses["visual"] = np.stack(obses["visual"])
        obses["proprio"] = np.stack(obses["proprio"])
        states = np.stack(states)
        return obses, states
        
    def sample_random_init_goal_states(self, seed):
        # markers interpretati da prepare(): non sono qpos reali, sono solo un flag
        init_marker = np.zeros(9, dtype=np.float32)
        goal_marker = np.ones(9, dtype=np.float32)
        return init_marker, goal_marker

    def prepare(self, seed, init_state=None):
        if isinstance(seed, list):
            seed = seed[0]
        ms_obs, info = self.ms_env.reset(seed=seed)
        self.last_success = False

        is_goal_request = init_state is not None and np.asarray(init_state).flat[0] == 1

        if is_goal_request:
            u = self.ms_env.unwrapped
            orig_pose = u.obj.pose
            goal_p = orig_pose.p.clone()
            goal_p[..., 0] = u.goal_region.pose.p[..., 0]
            goal_p[..., 1] = u.goal_region.pose.p[..., 1]  # z invariata
            u.obj.set_pose(Pose.create_from_pq(p=goal_p, q=orig_pose.q))
            ms_obs = u.get_obs()

        obs, state = self._process_obs(ms_obs)
        return obs, state
        
    def update_env(self, env_info):
        pass

    def eval_state(self, goal_state, cur_state):
        # Calcoliamo l'errore reale sui giunti tra dove voleva andare e dove è arrivato
        dist = float(np.linalg.norm(cur_state - goal_state))
        
        # Consideriamo un successo se l'errore è minimo (es. vicino al goal immaginato)
        is_success = dist < 0.1 
        
        return {
            "state_dist": dist,
            "success": is_success or bool(self.last_success)
        }
