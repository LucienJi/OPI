import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import EzPickle
import Box2D
from Box2D.b2 import (
    circleShape,
    edgeShape,
    fixtureDef,
    polygonShape,
    revoluteJointDef,
    contactListener,
)

# --- Standard Constants ---
FPS = 50
SCALE = 30.0
INITIAL_RANDOM = 1000.0 

# Base Shapes
MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6
LEG_SPRING_TORQUE = 40
SIDE_ENGINE_HEIGHT = 14
SIDE_ENGINE_AWAY = 12
VIEWPORT_W = 600
VIEWPORT_H = 400

# Base Lander Polygon
LANDER_POLY = [(-14, +17), (-17, 0), (-17, -10), (+17, -10), (+17, 0), (+14, +17)]

class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        if self.env.lander == contact.fixtureA.body or self.env.lander == contact.fixtureB.body:
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True
    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False

class HybridLunarLander(gym.Env, EzPickle):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(
        self,
        render_mode: str | None = None,
        continuous: bool = True,
        mode: str = "learner",
        max_episode_steps: int | None = None,
        landing_speed_threshold: float = 0.6,
        landing_angle_threshold: float = 0.2,
        landing_x_tolerance: float = 2.0,
        require_two_legs: bool = True,
        wind_when_grounded: bool = True,
        sleep_terminates: bool = True,
        sleep_penalty: float = -50.0,
        sleep_step_penalty: float = -1.0,
    ):
        """
        mode:
            "stilt":   Oracle 1 - Long legs, High COG, No wind, Normal G.
            "wide":    Oracle 2 - Short legs, Wide body, Wind, Normal G.
            "heavy":   Oracle 3 - Normal shape, High Density, No wind, Normal G.
            "learner": Normal shape, Normal Density, Wind, High G.
        """
        EzPickle.__init__(
            self,
            render_mode,
            continuous,
            mode,
            max_episode_steps,
            landing_speed_threshold,
            landing_angle_threshold,
            landing_x_tolerance,
            require_two_legs,
            wind_when_grounded,
            sleep_terminates,
            sleep_penalty,
            sleep_step_penalty,
        )
        self.render_mode = render_mode
        self.continuous = continuous
        self.mode = mode
        self.max_episode_steps = max_episode_steps
        self.landing_speed_threshold = landing_speed_threshold
        self.landing_angle_threshold = landing_angle_threshold
        self.landing_x_tolerance = landing_x_tolerance
        self.require_two_legs = require_two_legs
        self.wind_when_grounded = wind_when_grounded
        self.landing_y_tolerance = 0.5
        self.sleep_terminates = sleep_terminates
        self.sleep_penalty = sleep_penalty
        self.sleep_step_penalty = sleep_step_penalty
        
        self.isopen = True
        self.world = None
        self.moon = None
        self.lander = None
        self.particles = []
        self.screen = None
        self.clock = None
        self.step_count = 0
        self.landed_success_prev = False
        self.is_gusting = False
        self.gust_steps_remaining = 0
        self.wind_idx = 0
        self.torque_idx = 0
        self.wind_disabled_by_contact = False

        # Observation Space (8-dim)
        low = np.array([-2.5, -2.5, -10.0, -10.0, -2 * math.pi, -10.0, -0.0, -0.0]).astype(np.float32)
        high = np.array([2.5, 2.5, 10.0, 10.0, 2 * math.pi, 10.0, 1.0, 1.0]).astype(np.float32)
        self.observation_space = spaces.Box(low, high)

        if self.continuous:
            self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(4)

    def _init_config(self):
        """
        Initialize Physics & Environment based on Mode
        """
        # --- Defaults (Normal Params) ---
        self.leg_w = 2
        self.leg_h = 8
        self.leg_down = 18    # Standard leg length
        self.leg_away = 20    # Standard leg width
        self.density = 5.0    # Standard density
        self.main_engine_y = 4.0 # Standard engine height
        
        self.gravity = -10.0  # Standard gravity
        self.enable_wind = False
        self.wind_power = 0.0
        self.turbulence_power = 0.0
        
        self.color = (128, 102, 230) # Purple (Default)

        # --- Configurations ---
        
        if self.mode == "stilt":
            # Oracle 1: long legs, high COG, no wind, normal gravity.
            self.leg_down = 30      # Longer legs -> higher COG and less stable landing
            self.leg_h = 16         # Taller leg box (half-extents in Box2D)
            self.main_engine_y = 6.0 # Raise engine location
            self.color = (200, 50, 50) # Red
            
        elif self.mode == "wide":
            # Oracle 2: short legs, wide body, wind, normal gravity.
            self.leg_down = 16      # Shorter legs -> lower COG
            self.leg_away = 35      # Wider stance -> more stable
            self.enable_wind = True
            self.wind_power = 18.0  # Strong wind
            self.turbulence_power = 1.8
            self.color = (50, 50, 200) # Blue
            
        elif self.mode == "heavy":
            # Oracle 3: normal shape, higher density, no wind, normal gravity.
            self.density = 8.5    # Heavier but still learnable
            self.main_engine_y = 4.0 # Lower engine location
            self.color = (50, 200, 50) # Green

        elif self.mode == "learner":
            # Learner: normal shape/density, wind, higher gravity.
            self.enable_wind = True 
            self.leg_down = 20 # longer legs ? better for landing ? 
            self.wind_power = 18.0
            self.turbulence_power = 2.0
            self.gravity = -13.0   # Lower Gravity (Jupiter-like)
            self.color = (128, 102, 230) # Purple
            self.landing_speed_threshold = 0.5 
            self.landing_angle_threshold = 0.15

    def _destroy(self):
        if not self.moon: return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.moon)
        self.moon = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])

    def _neutral_action(self):
        if self.continuous:
            return np.array([0.0, 0.0], dtype=np.float32)
        return 0

    def _init_gust_process(self):
        # Random segment model: alternate gust ON/OFF with random durations.
        self.gust_on_range = (20, 60)
        self.gust_off_range = (40, 120)
        self.is_gusting = bool(self.np_random.integers(0, 2))
        if self.is_gusting:
            self.gust_steps_remaining = int(self.np_random.integers(self.gust_on_range[0], self.gust_on_range[1] + 1))
        else:
            self.gust_steps_remaining = int(self.np_random.integers(self.gust_off_range[0], self.gust_off_range[1] + 1))

    def _advance_gust(self):
        if not self.enable_wind:
            self.is_gusting = False
            self.gust_steps_remaining = 0
            return
        self.gust_steps_remaining -= 1
        if self.gust_steps_remaining <= 0:
            self.is_gusting = not self.is_gusting
            if self.is_gusting:
                self.gust_steps_remaining = int(
                    self.np_random.integers(self.gust_on_range[0], self.gust_on_range[1] + 1)
                )
            else:
                self.gust_steps_remaining = int(
                    self.np_random.integers(self.gust_off_range[0], self.gust_off_range[1] + 1)
                )

    def _get_state(self):
        pos = self.lander.position
        vel = self.lander.linearVelocity
        
        # Half of the viewport width, used for normalization
        viewport_half_w = VIEWPORT_W / SCALE / 2
        
        state = [
            # --- Change: compute X relative to the helipad center ---
            (pos.x - self.helipad_x) / viewport_half_w,
            # -------------------------------------------
            
            (pos.y - (self.helipad_y + self.leg_down / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ]
        return np.array(state, dtype=np.float32)
    def _get_landing_metrics(self):
        pos = self.lander.position
        vel = self.lander.linearVelocity
        helipad_center = self.helipad_x
        x_in_pad = abs(pos.x - helipad_center) <= self.landing_x_tolerance
        y_target = self.helipad_y + self.leg_down / SCALE
        y_low = abs(pos.y - y_target) <= self.landing_y_tolerance
        speed = math.hypot(vel.x, vel.y)
        # Speed threshold is in world units (m/s); defaults aim for learnability.
        speed_ok = speed <= self.landing_speed_threshold
        angle_ok = abs(self.lander.angle) <= self.landing_angle_threshold
        if self.require_two_legs:
            leg_contact = self.legs[0].ground_contact and self.legs[1].ground_contact
        else:
            leg_contact = self.legs[0].ground_contact or self.legs[1].ground_contact
        return {
            "x_in_pad": bool(x_in_pad),
            "y_low": bool(y_low),
            "speed": float(speed),
            "speed_ok": bool(speed_ok),
            "angle": float(self.lander.angle),
            "angle_ok": bool(angle_ok),
            "leg_contact": bool(leg_contact),
        }

    def _compute_landing_success(self):
        metrics = self._get_landing_metrics()
        landed_success = (
            metrics["x_in_pad"]
            and metrics["y_low"]
            and metrics["speed_ok"]
            and metrics["angle_ok"]
            and metrics["leg_contact"]
        )
        info = {
            "landed_success": bool(landed_success),
            **metrics,
        }
        return landed_success, info

    def _precheck_stable_contact_for_wind(self):
        metrics = self._get_landing_metrics()
        return (
            metrics["x_in_pad"]
            and metrics["y_low"]
            and metrics["speed_ok"]
            and metrics["angle_ok"]
            and metrics["leg_contact"]
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._destroy()
        
        # 1. Load Config
        self._init_config()

        # 2. Create World with Configured Gravity
        self.world = Box2D.b2World(gravity=(0, self.gravity))
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None
        self.step_count = 0
        self.landed_success_prev = False
        self._init_gust_process()
        if not self.enable_wind:
            self.is_gusting = False
            self.gust_steps_remaining = 0
        self.wind_disabled_by_contact = False

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        # --- Begin change: randomize the terrain ---
        CHUNKS = 11
        height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        
        # Randomly choose the helipad center index.
        # The original code needs two flat units on each side of the center, so we need margin.
        # The valid range is 2 to CHUNKS - 3 (for 11 chunks, indices 2 through 8).
        helipad_idx = self.np_random.integers(2, CHUNKS - 2)

        self.helipad_x1 = chunk_x[helipad_idx - 1]
        self.helipad_x2 = chunk_x[helipad_idx + 1]
        
        # Record the absolute helipad center position for observations
        self.helipad_x = (self.helipad_x1 + self.helipad_x2) / 2

        self.helipad_y = H / 4
        
        # Flatten the terrain around the helipad
        height[helipad_idx - 2] = self.helipad_y
        height[helipad_idx - 1] = self.helipad_y
        height[helipad_idx + 0] = self.helipad_y
        height[helipad_idx + 1] = self.helipad_y
        height[helipad_idx + 2] = self.helipad_y
        
        smooth_y = [0.33 * (height[i - 1] + height[i + 0] + height[i + 1]) for i in range(CHUNKS)]
        # --- End change ---

        self.moon = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (W, 0)]))
        self.sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])
        self.moon.color1 = (0.0, 0.0, 0.0); self.moon.color2 = (0.0, 0.0, 0.0)

        # Create Lander
        initial_y = VIEWPORT_H / SCALE
        initial_x = VIEWPORT_W / SCALE / 2
        
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y), angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]),
                density=self.density, # Customized Density
                friction=0.1, categoryBits=0x0010, maskBits=0x001, restitution=0.0,
            ),
        )
        self.lander.color1 = self.color
        self.lander.color2 = (int(self.color[0]*0.6), int(self.color[1]*0.6), int(self.color[2]*0.6))
        
        self.lander.ApplyForceToCenter(
            (self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM), self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)), True,
        )

        if self.enable_wind:
            self.wind_idx = int(self.np_random.integers(-9999, 9999))
            self.torque_idx = int(self.np_random.integers(-9999, 9999))

        # Create Legs (Customized Config)
        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i * self.leg_away / SCALE, initial_y), # Customized Leg Away
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(self.leg_w / SCALE, self.leg_h / SCALE)), # Box2D uses half-extents
                    density=1.0, restitution=0.0, categoryBits=0x0020, maskBits=0x001,
                ),
            )
            leg.ground_contact = False
            leg.color1 = self.lander.color1; leg.color2 = self.lander.color2
            
            rjd = revoluteJointDef(
                bodyA=self.lander, bodyB=leg, localAnchorA=(0, 0), 
                localAnchorB=(i * self.leg_away / SCALE, self.leg_down / SCALE), # Customized Anchor
                enableMotor=True, enableLimit=True, maxMotorTorque=LEG_SPRING_TORQUE, motorSpeed=+0.3 * i,
            )
            if i == -1:
                rjd.lowerAngle = +0.9 - 0.5
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs
        if self.render_mode == "human": self.render()
        state = self._get_state()
        landed_success, landing_info = self._compute_landing_success()
        info = {
            **landing_info,
            "wind_mag": 0.0,
            "torque_mag": 0.0,
            "is_gusting": bool(self.is_gusting),
            "termination_reason": None,
            "step_count": self.step_count,
        }
        return state, info

    def _create_particle(self, mass, x, y, ttl):
        p = self.world.CreateDynamicBody(
            position=(x, y), angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=2 / SCALE, pos=(0, 0)),
                density=mass, friction=0.1, categoryBits=0x0100, maskBits=0x001, restitution=0.3,
            ),
        )
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all_particle):
        while self.particles and (all_particle or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    def step(self, action):
        assert self.lander is not None
        self.step_count += 1

        # Wind logic: random gust segments, no "leg anchor" shutoff unless truly landed.
        wind_mag = 0.0
        torque_mag = 0.0
        # Once any leg touches, stop wind for the rest of the episode.
        if (self.legs[0].ground_contact or self.legs[1].ground_contact):
            self.wind_disabled_by_contact = True
        if self.enable_wind and not self.wind_disabled_by_contact and not self.landed_success_prev:
            self._advance_gust()
            gust_scale = 1.0 if self.is_gusting else 0.2
            wind_scale = 1.0
            if not self.wind_when_grounded:
                stable_contact = self._precheck_stable_contact_for_wind()
                if stable_contact:
                    wind_scale = 0.0
            wind_mag = (
                math.tanh(math.sin(0.02 * self.wind_idx) + (math.sin(math.pi * 0.01 * self.wind_idx)))
                * self.wind_power
                * gust_scale
                * wind_scale
            )
            torque_mag = (
                math.tanh(math.sin(0.02 * self.torque_idx) + (math.sin(math.pi * 0.01 * self.torque_idx)))
                * self.turbulence_power
                * gust_scale
                * wind_scale
            )
            self.wind_idx += 1
            self.torque_idx += 1
            if wind_scale > 0.0:
                self.lander.ApplyForceToCenter((wind_mag, 0.0), True)
                self.lander.ApplyTorque(torque_mag, True)

        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float64)
        else:
            assert self.action_space.contains(action), f"{action!r} invalid"

        # Engines
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (not self.continuous and action == 2):
            if self.continuous: m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5
            else: m_power = 1.0
            
            # Use self.main_engine_y
            ox = tip[0] * (self.main_engine_y / SCALE + 2 * dispersion[0]) + side[0] * dispersion[1]
            oy = -tip[1] * (self.main_engine_y / SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
            
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            if self.render_mode is not None:
                p = self._create_particle(3.5, impulse_pos[0], impulse_pos[1], m_power)
                p.ApplyLinearImpulse((ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power), impulse_pos, True)
            self.lander.ApplyLinearImpulse((-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power), impulse_pos, True)

        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (not self.continuous and action in [1, 3]):
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
            else:
                direction = action - 2
                s_power = 1.0

            ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)
            oy = -tip[1] * dispersion[0] - side[1] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)
            impulse_pos = (self.lander.position[0] + ox - tip[0] * 17 / SCALE, self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE)
            if self.render_mode is not None:
                p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
                p.ApplyLinearImpulse((ox * SIDE_ENGINE_POWER * s_power, oy * SIDE_ENGINE_POWER * s_power), impulse_pos, True)
            self.lander.ApplyLinearImpulse((-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power), impulse_pos, True)

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        state = self._get_state()
        landed_success, landing_info = self._compute_landing_success()
        self.landed_success_prev = landed_success

        reward = 0
        shaping = (
            -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
            - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
            - 100 * abs(state[4]) + 10 * state[6] + 10 * state[7]
        )
        if self.prev_shaping is not None: reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= m_power * 0.30
        reward -= s_power * 0.03

        terminated = False
        truncated = False
        termination_reason = None
        if self.game_over:
            terminated = True
            reward = -100
            termination_reason = "crash"
        elif abs(state[0]) >= 1.0:
            terminated = True
            reward = -100
            termination_reason = "out_of_bounds"
        elif landed_success:
            terminated = True
            reward = +100
            termination_reason = "success"
        elif not self.lander.awake:
            # Sleeping away from the pad is not success.
            if self.sleep_terminates:
                terminated = True
                reward = self.sleep_penalty
                termination_reason = "sleep"
            else:
                reward += self.sleep_step_penalty
                termination_reason = None

        if self.max_episode_steps is not None and self.step_count >= self.max_episode_steps:
            if not terminated:
                truncated = True
            else:
                # Prefer terminated over truncated for the same step.
                truncated = False

        info = {
            **landing_info,
            "wind_mag": float(wind_mag),
            "torque_mag": float(torque_mag),
            "is_gusting": bool(self.is_gusting),
            "termination_reason": termination_reason,
            "step_count": self.step_count,
        }
        if truncated:
            info["truncation_reason"] = "time_limit"

        if self.render_mode == "human": self.render()
        return np.array(state, dtype=np.float32), reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None: return
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise ImportError("pygame is required for rendering")

        if self.screen is None and self.render_mode == "human":
            pygame.init(); pygame.display.init(); self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None: self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))
        self.surf = pygame.transform.scale(self.surf, (int(VIEWPORT_W), int(VIEWPORT_H)))
        pygame.draw.rect(self.surf, (255, 255, 255), self.surf.get_rect())

        # Draw Wind Indicator for Learner
        if self.mode == "learner" and hasattr(self, 'is_gusting') and self.is_gusting:
             pygame.draw.rect(self.surf, (220, 240, 255), (0, 0, VIEWPORT_W, VIEWPORT_H))

        # Particles
        for obj in self.particles:
            obj.ttl -= 0.15
            color = (int(max(0.2, 0.15 + obj.ttl) * 255), int(max(0.2, 0.5 * obj.ttl) * 255), int(max(0.2, 0.5 * obj.ttl) * 255))
            obj.color1 = color; obj.color2 = color
        self._clean_particles(False)

        # Terrain
        for p in self.sky_polys:
            scaled_poly = [(c[0] * SCALE, c[1] * SCALE) for c in p]
            pygame.draw.polygon(self.surf, (0, 0, 0), scaled_poly)

        # Lander & Particles
        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    pygame.draw.circle(self.surf, color=obj.color1, center=trans * f.shape.pos * SCALE, radius=f.shape.radius * SCALE)
                else:
                    path = [trans * v * SCALE for v in f.shape.vertices]
                    pygame.draw.polygon(self.surf, color=obj.color1, points=path)
                    pygame.draw.aalines(self.surf, color=obj.color2, points=path, closed=True)

        # Helipad
        for x in [self.helipad_x1, self.helipad_x2]:
            x = x * SCALE
            flagy1 = self.helipad_y * SCALE
            pygame.draw.line(self.surf, (255, 255, 255), (x, flagy1), (x, flagy1 + 50), 1)
            pygame.draw.polygon(self.surf, (204, 204, 0), [(x, flagy1 + 50), (x, flagy1 + 40), (x + 25, flagy1 + 45)])

        self.surf = pygame.transform.flip(self.surf, False, True)

        if self.render_mode == "human":
            self.screen.blit(self.surf, (0, 0)); pygame.event.pump(); self.clock.tick(self.metadata["render_fps"]); pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None: import pygame; pygame.display.quit(); pygame.quit(); self.isopen = False


def _run_sanity_tests():
    print("Sanity: sleep termination")
    env = HybridLunarLander(
        mode="heavy",
        wind_when_grounded=True,
        sleep_terminates=True,
        sleep_penalty=-50.0,
    )
    env.reset(seed=0)
    env.lander.awake = False
    env.lander.linearVelocity = (0.0, 0.0)
    env.lander.angularVelocity = 0.0
    _, reward, terminated, _, info = env.step(env._neutral_action())
    print("sleep terminated:", terminated, "reward:", reward, "termination_reason:", info.get("termination_reason"))
    print("reward_nonpositive:", reward <= 0.0)
    env.close()

    print("Sanity: wind stops only when stable on pad")
    env = HybridLunarLander(mode="learner", wind_when_grounded=False)
    env.reset(seed=1)
    env.legs[0].ground_contact = True
    env.legs[1].ground_contact = True
    env.lander.linearVelocity = (0.0, 0.0)
    env.lander.angle = 0.0

    env.lander.position = (env.helipad_x1 - 5.0, env.helipad_y + env.leg_down / SCALE)
    _, _, _, _, info = env.step(env._neutral_action())
    print(
        "wind out of pad:",
        info["wind_mag"],
        info["torque_mag"],
        "x_in_pad:",
        info["x_in_pad"],
        "y_low:",
        info["y_low"],
    )

    helipad_center = 0.5 * (env.helipad_x1 + env.helipad_x2)
    env.lander.position = (helipad_center, env.helipad_y + env.leg_down / SCALE)
    _, _, _, _, info = env.step(env._neutral_action())
    print(
        "wind in pad:",
        info["wind_mag"],
        info["torque_mag"],
        "x_in_pad:",
        info["x_in_pad"],
        "y_low:",
        info["y_low"],
    )
    env.close()


if __name__ == "__main__":
    _run_sanity_tests()
