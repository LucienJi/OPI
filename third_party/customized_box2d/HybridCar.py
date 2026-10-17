import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import EzPickle
import Box2D
from Box2D.b2 import (
    contactListener,
    fixtureDef,
    polygonShape,
    revoluteJointDef,
    edgeShape,
    rayCastCallback
)
import pygame
from pygame import gfxdraw

# --- Constants ---
FPS = 50
SCALE = 6.0
TRACK_RAD = 900 / SCALE
PLAYFIELD = 2000 / SCALE
TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4
MAX_EPISODE_STEPS = 2000
NO_PROGRESS_STEPS = 250
OUT_OF_BOUNDS_PENALTY = -100.0
MAX_WHEEL_OMEGA = 80.0
MAX_HULL_SPEED = 60.0
MAX_HULL_ANGULAR_VEL = 5.0
PROGRESS_REWARD_SCALE = 1.0
OFF_ROAD_PENALTY = -0.2
CRASH_PENALTY = -500.0
CRASH_SPEED_TH = 0.5

WINDOW_W = 1000
WINDOW_H = 800
ZOOM = 2.7
ZOOM_FOLLOW = True

# Lidar Settings
LIDAR_RAYS = 32
LIDAR_RANGE = 50
LIDAR_FOV = 3.14159

# --- Vehicle Configs ---
# --- 1. Vehicle Configs (Learner Fixed) ---
VEHICLE_CONFIGS = {
    # Oracle 1: Grip monster (AWD, High Grip)
    "Racer": {
        "engine_power_mul": 100_000_000,   # <--- modified here
        "friction_limit_mul": 1_000_000,   # <--- modified here
        "wheel_moment": 4_000,
        "hull_poly": [(-60, +130), (+60, +130), (+60, -90), (-60, -90)],
        "wheel_pos": [(-55, +80), (+55, +80), (-55, -82), (+55, -82)],
        "drive": "AWD",
        "size": 0.02,
        "color": (200, 0, 0)
    },
    # Oracle 2: Drift master (RWD, Slippery)
    "Drifter": {
        "engine_power_mul": 80_000_000,    # <--- modified here
        "friction_limit_mul": 600_000,     # <--- modified here
        "wheel_moment": 2_000,
        "hull_poly": [(-40, +150), (+40, +150), (+40, -100), (-40, -100)],
        "wheel_pos": [(-40, +100), (+40, +100), (-40, -80), (+40, -80)],
        "drive": "RWD",
        "size": 0.02,
        "color": (0, 0, 200)
    },
    # Oracle 3: Heavy bus (FWD, Heavy, Stable)
    "Bus": {
        "engine_power_mul": 60_000_000,    # <--- modified here
        "friction_limit_mul": 1_500_000,   # <--- modified here
        "wheel_moment": 8_000,
        "hull_poly": [(-80, +200), (+80, +200), (+80, -200), (-80, -200)],
        "wheel_pos": [(-80, +150), (+80, +150), (-80, -150), (+80, -150)],
        "drive": "FWD",
        "size": 0.02,
        "color": (200, 200, 0)
    },
    # Learner: Prototype car (fixed config)
    "Prototype": {
        "engine_power_mul": 60_000_000,    # <--- modified here
        "friction_limit_mul": 600_000,     # <--- modified here
        "wheel_moment": 4_000,
        "hull_poly": [(-50, +140), (+50, +140), (+50, -100), (-50, -100)],
        "wheel_pos": [(-50, +90), (+50, +90), (-50, -90), (+50, -90)],
        "drive": "RWD",
        "size": 0.02,
        "color": (0, 200, 0)
    }
}

class FrictionDetector(contactListener):
    def __init__(self, env, lap_complete_percent: float):
        contactListener.__init__(self)
        self.env = env
        self.lap_complete_percent = lap_complete_percent
    def BeginContact(self, contact):
        self._contact(contact, True)
    def EndContact(self, contact):
        self._contact(contact, False)
    def _contact(self, contact, begin):
        if begin and not self.env.crashed:
            b1 = contact.fixtureA.body
            b2 = contact.fixtureB.body
            if getattr(b1.userData, "is_border", False) or getattr(b2.userData, "is_border", False):
                if self.env.car and (b1 == self.env.car.hull or b1 in self.env.car.wheels or b2 == self.env.car.hull or b2 in self.env.car.wheels):
                    v = self.env.car.hull.linearVelocity
                    speed = math.sqrt(v[0] * v[0] + v[1] * v[1])
                    if speed >= CRASH_SPEED_TH:
                        self.env.crashed = True
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__: tile = u1; obj = u2
        if u2 and "road_friction" in u2.__dict__: tile = u2; obj = u1
        if not tile: return
        
        tile.color[:] = self.env.road_color
        if not obj or "tiles" not in obj.__dict__: return
        if begin:
            obj.tiles.add(tile)
            if not tile.road_visited:
                tile.road_visited = True
                self.env.reward += 1000.0 / len(self.env.track)
                self.env.tile_visited_count += 1
                if (
                    tile.idx == 0
                    and len(self.env.track) > 0
                    and (self.env.tile_visited_count / len(self.env.track)) >= self.lap_complete_percent
                ):
                    self.env.new_lap = True
        else:
            obj.tiles.remove(tile)

class DynamicCar:
    def __init__(self, world, init_angle, init_x, init_y, config_name):
        self.world = world
        self.cfg = VEHICLE_CONFIGS[config_name]
        
        SIZE = self.cfg["size"]
        self.engine_power = self.cfg["engine_power_mul"] * SIZE * SIZE
        self.friction_limit = self.cfg["friction_limit_mul"] * SIZE * SIZE
        self.wheel_moment = self.cfg["wheel_moment"] * SIZE * SIZE
        
        # Hull
        self.hull = self.world.CreateDynamicBody(
            position=(init_x, init_y), angle=init_angle,
            fixtures=[fixtureDef(shape=polygonShape(vertices=[(x*SIZE, y*SIZE) for x, y in self.cfg["hull_poly"]]), density=1.0)]
        )
        self.hull.color = self.cfg["color"]
        self.hull.userData = self.hull
        
        # Wheels
        self.wheels = []
        WHEEL_R = 27
        WHEEL_W = 14
        WHEEL_POLY = [(-WHEEL_W, +WHEEL_R), (+WHEEL_W, +WHEEL_R), (+WHEEL_W, -WHEEL_R), (-WHEEL_W, -WHEEL_R)]
        
        for wx, wy in self.cfg["wheel_pos"]:
            w = self.world.CreateDynamicBody(
                position=(init_x + wx * SIZE, init_y + wy * SIZE), angle=init_angle,
                fixtures=fixtureDef(
                    shape=polygonShape(vertices=[(x*SIZE, y*SIZE) for x, y in WHEEL_POLY]),
                    density=0.1, categoryBits=0x0020, maskBits=(0x0001 | 0x0004), restitution=0.0
                )
            )
            w.wheel_rad = WHEEL_R * SIZE
            w.gas = 0.0; w.brake = 0.0; w.steer = 0.0; w.phase = 0.0; w.omega = 0.0
            w.userData = w
            
            rjd = revoluteJointDef(
                bodyA=self.hull, bodyB=w, localAnchorA=(wx*SIZE, wy*SIZE), localAnchorB=(0, 0),
                enableMotor=True, enableLimit=True, maxMotorTorque=180*900*SIZE*SIZE, motorSpeed=0,
                lowerAngle=-0.4, upperAngle=+0.4
            )
            w.joint = self.world.CreateJoint(rjd)
            w.tiles = set()
            self.wheels.append(w)

    def step(self, dt, action):
        steer = -action[0]
        gas = action[1]
        brake = action[2]
        
        self.wheels[0].steer = steer
        self.wheels[1].steer = steer
        
        drive = self.cfg["drive"]
        for i, w in enumerate(self.wheels):
            w.gas = 0.0
            if drive == "AWD" or (drive == "FWD" and i < 2) or (drive == "RWD" and i >= 2):
                w.gas = gas
            w.brake = brake

        SIZE = self.cfg["size"]
        for w in self.wheels:
            # Steer Joint
            # === Fix 1: ensure the computed result is cast to float ===
            dir_sign = np.sign(w.steer - w.joint.angle)
            val = abs(w.steer - w.joint.angle)
            w.joint.motorSpeed = float(dir_sign * min(50.0 * val, 3.0))
            # ======================================

            # Friction Logic
            friction_limit = self.friction_limit * 0.6
            if len(w.tiles) > 0:
                friction_limit = max([self.friction_limit * t.road_friction for t in w.tiles])

            # Physics
            forw = w.GetWorldVector((0, 1))
            side = w.GetWorldVector((1, 0))
            v = w.linearVelocity
            vf = forw[0] * v[0] + forw[1] * v[1]
            vs = side[0] * v[0] + side[1] * v[1]

            # add small coef not to divide by zero
            # Ensure calculations don't explode with numpy types when accumulating
            w.omega += dt * self.engine_power * w.gas / self.wheel_moment / (abs(w.omega) + 5.0)
            
            if w.brake >= 0.9: w.omega = 0
            elif w.brake > 0:
                dir_sign = -np.sign(w.omega)
                val = 15 * w.brake
                if abs(val) > abs(w.omega): val = abs(w.omega)
                w.omega += dir_sign * val
            w.phase += w.omega * dt

            vr = w.omega * w.wheel_rad
            f_force = -vf + vr
            p_force = -vs

            # Stability Constants
            f_force *= 205000 * SIZE * SIZE
            p_force *= 205000 * SIZE * SIZE
            force = np.sqrt(np.square(f_force) + np.square(p_force))

            if abs(force) > friction_limit:
                f_force /= force; p_force /= force
                force = friction_limit
                f_force *= force; p_force *= force

            w.omega -= dt * f_force * w.wheel_rad / self.wheel_moment
            if w.omega > MAX_WHEEL_OMEGA:
                w.omega = MAX_WHEEL_OMEGA
            elif w.omega < -MAX_WHEEL_OMEGA:
                w.omega = -MAX_WHEEL_OMEGA
            
            # === Fix 2: ApplyForceToCenter also needs a float tuple ===
            w.ApplyForceToCenter(
                (
                    float(p_force * side[0] + f_force * forw[0]), 
                    float(p_force * side[1] + f_force * forw[1])
                ), 
                True
            )
            # ====================================================
    def destroy(self):
        self.world.DestroyBody(self.hull)
        for w in self.wheels: self.world.DestroyBody(w)

class LidarCallback(rayCastCallback):
    def __init__(self, ignore_bodies):
        super().__init__()
        self.ignore_bodies = ignore_bodies
        self.fraction = 1.0
        self.point = None

    def ReportFixture(self, fixture, point, normal, fraction):
        if fixture.body in self.ignore_bodies: return -1.0
        if fixture.sensor: return -1.0
        self.fraction = fraction
        self.point = point
        return fraction

class HybridCarRacing(gym.Env, EzPickle):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode: str | None = None, verbose: bool = False, mode: str = "learner", reward_mode: str = "tiles"):
        EzPickle.__init__(self, render_mode, verbose, mode, reward_mode)
        self.mode = mode
        self.reward_mode = reward_mode
        self.lap_complete_percent = 0.95
        
        self.road_color = np.array([102, 102, 102])
        self.bg_color = np.array([102, 204, 102])
        self.grass_color = np.array([102, 230, 102])

        self.contactListener_keepref = FrictionDetector(self, self.lap_complete_percent)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.screen = None
        self.clock = None
        self.road = None
        self.borders = None
        self.car = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.fd_tile = fixtureDef(shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)]))

        obs_dim = 14 + LIDAR_RAYS
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(np.array([-1, 0, 0]).astype(np.float32), np.array([+1, +1, +1]).astype(np.float32))
        self.render_mode = render_mode

    def _destroy(self):
        if not self.road: return
        for t in self.road: self.world.DestroyBody(t)
        self.road = []
        if self.borders:
            for b in self.borders: self.world.DestroyBody(b)
            self.borders = []
        if self.car: self.car.destroy()

    def _create_track(self):
        CHECKPOINTS = 12
        checkpoints = []
        for c in range(CHECKPOINTS):
            alpha = 2 * math.pi * c / CHECKPOINTS + self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
            rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)
            if c == 0: alpha = 0; rad = 1.5 * TRACK_RAD
            if c == CHECKPOINTS - 1: alpha = 2 * math.pi * c / CHECKPOINTS; self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS; rad = 1.5 * TRACK_RAD
            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))
        self.road = []
        x, y, beta = 1.5 * TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0: laps += 1; visited_other_side = False
            if alpha < 0: visited_other_side = True; alpha += 2 * math.pi
            while True:
                failed = True
                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha: failed = False; break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0: break
                if not failed: break
                alpha -= 2 * math.pi
                continue
            r1x = math.cos(beta); r1y = math.sin(beta)
            p1x = -r1y; p1y = r1x
            dest_dx = dest_x - x; dest_dy = dest_y - y
            proj = r1x * dest_dx + r1y * dest_dy
            while beta - alpha > 1.5 * math.pi: beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi: beta += 2 * math.pi
            prev_beta = beta
            proj *= SCALE
            if proj > 0.3: beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
            if proj < -0.3: beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
            x += p1x * TRACK_DETAIL_STEP; y += p1y * TRACK_DETAIL_STEP
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
            if laps > 4: break
            no_freeze -= 1
            if no_freeze == 0: break
        
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0: return False
            pass_through_start = track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha
            if pass_through_start and i2 == -1: i2 = i
            elif pass_through_start and i1 == -1: i1 = i; break
        track = track[i1 : i2 - 1]
        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta); first_perp_y = math.sin(first_beta)
        if np.sqrt(np.square(first_perp_x * (track[0][2] - track[-1][2])) + np.square(first_perp_y * (track[0][3] - track[-1][3]))) > TRACK_DETAIL_STEP: return False
        
        # Draw Road Tiles
        self.road_poly = []
        self.borders = []
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            road1_l = (x1 - TRACK_WIDTH * math.cos(beta1), y1 - TRACK_WIDTH * math.sin(beta1))
            road1_r = (x1 + TRACK_WIDTH * math.cos(beta1), y1 + TRACK_WIDTH * math.sin(beta1))
            road2_l = (track[i-1][2] - TRACK_WIDTH * math.cos(track[i-1][1]), track[i-1][3] - TRACK_WIDTH * math.sin(track[i-1][1]))
            road2_r = (track[i-1][2] + TRACK_WIDTH * math.cos(track[i-1][1]), track[i-1][3] + TRACK_WIDTH * math.sin(track[i-1][1]))
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            t.color = self.road_color + 0.01 * (i % 3) * 255
            t.road_visited = False; t.road_friction = 1.0; t.idx = i; t.fixtures[0].sensor = True
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)
            
            # Borders (Simplified for brevity)
            if True: # Always draw borders for visibility
                b_color = (255, 255, 255) if i % 2 == 0 else (255, 0, 0)
                # Left Border
                b1_l = (x1 - (TRACK_WIDTH+BORDER) * math.cos(beta1), y1 - (TRACK_WIDTH+BORDER) * math.sin(beta1))
                b1_r = (x1 - TRACK_WIDTH * math.cos(beta1), y1 - TRACK_WIDTH * math.sin(beta1))
                b2_l = (track[i-1][2] - (TRACK_WIDTH+BORDER) * math.cos(track[i-1][1]), track[i-1][3] - (TRACK_WIDTH+BORDER) * math.sin(track[i-1][1]))
                b2_r = road2_l
                self.road_poly.append(([b1_l, b1_r, b2_r, b2_l], b_color))
                border_left = self.world.CreateStaticBody(
                    fixtures=fixtureDef(
                        shape=edgeShape(vertices=[b1_l, b2_l]),
                        friction=0.0,
                        categoryBits=0x0004,
                        maskBits=0xFFFF
                    )
                )
                border_left.userData = border_left
                border_left.is_border = True
                self.borders.append(border_left)
                # Right Border
                b1_l = (x1 + TRACK_WIDTH * math.cos(beta1), y1 + TRACK_WIDTH * math.sin(beta1))
                b1_r = (x1 + (TRACK_WIDTH+BORDER) * math.cos(beta1), y1 + (TRACK_WIDTH+BORDER) * math.sin(beta1))
                b2_l = road2_r
                b2_r = (track[i-1][2] + (TRACK_WIDTH+BORDER) * math.cos(track[i-1][1]), track[i-1][3] + (TRACK_WIDTH+BORDER) * math.sin(track[i-1][1]))
                self.road_poly.append(([b1_l, b1_r, b2_r, b2_l], b_color))
                border_right = self.world.CreateStaticBody(
                    fixtures=fixtureDef(
                        shape=edgeShape(vertices=[b1_r, b2_r]),
                        friction=0.0,
                        categoryBits=0x0004,
                        maskBits=0xFFFF
                    )
                )
                border_right.userData = border_right
                border_right.is_border = True
                self.borders.append(border_right)

        self.track = track
        self.track_points = np.array([(x, y) for (_, _, x, y) in track], dtype=np.float32)
        return True

    def _get_vector_obs(self):
        # 1. Lidar Raycast
        lidar_readings = []
        car_pos = self.car.hull.position
        car_angle = self.car.hull.angle
        
        # Bodies to ignore (Self)
        ignore = {self.car.hull}
        for w in self.car.wheels: ignore.add(w)
        
        self.lidar_points = [] # For rendering
        
        for i in range(LIDAR_RAYS):
            # === Fix: add + math.pi / 2 ===
            # Old logic: car_angle - FOV/2 ... (centered on +X)
            # New logic: car_angle + 90 degrees - FOV/2 ... (centered on +Y, i.e. the car heading)
            angle = (car_angle + math.pi/2) - LIDAR_FOV/2 + (LIDAR_FOV * i / (LIDAR_RAYS - 1))
            # ============================
            
            p1 = car_pos
            p2 = (car_pos[0] + math.cos(angle)*LIDAR_RANGE, car_pos[1] + math.sin(angle)*LIDAR_RANGE)
            
            callback = LidarCallback(ignore)
            self.world.RayCast(callback, p1, p2)
            lidar_readings.append(callback.fraction)
            
            # Visualization improvement: draw the actual hit point, or max range if nothing is hit
            if callback.point:
                self.lidar_points.append(callback.point)
            else:
                self.lidar_points.append(p2)

        # 2. Vehicle State
        vel = self.car.hull.linearVelocity
        local_v = (
            vel[0] * math.cos(-car_angle) - vel[1] * math.sin(-car_angle),
            vel[0] * math.sin(-car_angle) + vel[1] * math.cos(-car_angle)
        )
        
        state = [
            local_v[0], local_v[1],
            self.car.hull.angularVelocity,
            self.car.wheels[0].omega, self.car.wheels[1].omega, 
            self.car.wheels[2].omega, self.car.wheels[3].omega,
            self.car.wheels[0].joint.angle,
            0, 0, 0, 0, 0, 0 # Padding to match standard 14 dims if needed
        ]
        state_arr = np.asarray(state, dtype=np.float32) / 10.0
        return np.concatenate([state_arr, np.array(lidar_readings, dtype=np.float32)])

    def _calc_progress_idx(self, pos):
        diffs = self.track_points - np.array([pos[0], pos[1]], dtype=np.float32)
        d2 = np.sum(diffs * diffs, axis=1)
        return int(np.argmin(d2))

    def _is_on_road(self):
        for w in self.car.wheels:
            if len(w.tiles) > 0:
                return True
        return False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._destroy()
        self.world.contactListener = self.contactListener_keepref
        self.reward = 0.0; self.prev_reward = 0.0; self.tile_visited_count = 0; self.t = 0.0; self.new_lap = False
        self.step_count = 0
        self.no_progress_steps = 0
        self.last_tile_visited_count = 0
        self.crashed = False
        self.crash_penalized = False
        
        while True:
            if self._create_track(): break
        
        car_type = "Prototype"
        if self.mode == "racer": car_type = "Racer"
        elif self.mode == "drifter": car_type = "Drifter"
        elif self.mode == "bus": car_type = "Bus"
        
        self.car = DynamicCar(self.world, *self.track[0][1:4], config_name=car_type)
        self.lidar_points = []
        self.last_progress_idx = 0

        if self.render_mode == "human": self.render()
        return self._get_vector_obs(), {}

    def step(self, action):
        if action is not None:
            action = np.clip(action, -1, 1)
            gas = np.clip(action[1], 0, 1)
            brake = np.clip(action[2], 0, 1)
            self.car.step(1.0/FPS, [action[0], gas, brake])

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS
        v = self.car.hull.linearVelocity
        speed = math.sqrt(v[0] * v[0] + v[1] * v[1])
        if speed > MAX_HULL_SPEED:
            scale = MAX_HULL_SPEED / max(speed, 1e-6)
            self.car.hull.linearVelocity = (v[0] * scale, v[1] * scale)
        if self.car.hull.angularVelocity > MAX_HULL_ANGULAR_VEL:
            self.car.hull.angularVelocity = MAX_HULL_ANGULAR_VEL
        elif self.car.hull.angularVelocity < -MAX_HULL_ANGULAR_VEL:
            self.car.hull.angularVelocity = -MAX_HULL_ANGULAR_VEL

        step_reward = 0
        terminated = False
        truncated = False
        info = {}
        info["is_success"] = False
        if action is not None:
            self.reward -= 0.1

            if self.reward_mode == "progress":
                progress_idx = self._calc_progress_idx(self.car.hull.position)
                delta = progress_idx - self.last_progress_idx
                track_len = len(self.track)
                if delta < -track_len / 2:
                    delta += track_len
                elif delta > track_len / 2:
                    delta -= track_len
                self.reward += PROGRESS_REWARD_SCALE * float(delta)
                self.last_progress_idx = progress_idx
            if not self._is_on_road():
                self.reward += OFF_ROAD_PENALTY

            if self.tile_visited_count == len(self.track) or self.new_lap:
                terminated = True
                info["termination"] = "track_complete"
                info["is_success"] = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                terminated = True
                self.reward += OUT_OF_BOUNDS_PENALTY
                info["termination"] = "out_of_bounds"

            self.step_count += 1
            if self.tile_visited_count > self.last_tile_visited_count:
                self.no_progress_steps = 0
                self.last_tile_visited_count = self.tile_visited_count
            else:
                self.no_progress_steps += 1
            if self.step_count >= MAX_EPISODE_STEPS:
                truncated = True
                info["truncation"] = "max_steps"
            if self.no_progress_steps >= NO_PROGRESS_STEPS:
                truncated = True
                info["truncation"] = "no_progress"

            if self.crashed and not self.crash_penalized:
                self.reward += CRASH_PENALTY
                self.crash_penalized = True
            if self.crashed:
                terminated = True
                truncated = False
                info["termination"] = "crash"

            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward

        obs = self._get_vector_obs()
        if self.render_mode == "human": self.render()
        return obs, step_reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None: return
        if self.screen is None:
            pygame.init(); pygame.display.init(); self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        if self.clock is None: self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))
        
        # Camera
        angle = -self.car.hull.angle
        zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
        scroll_x = -(self.car.hull.position[0]) * zoom
        scroll_y = -(self.car.hull.position[1]) * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])

        # Background
        self.surf.fill((102, 204, 102))

        # Road
        for poly, color in self.road_poly:
            poly_px = []
            for p in poly:
                vec = pygame.math.Vector2(p).rotate_rad(angle)
                poly_px.append((vec[0]*zoom + trans[0], vec[1]*zoom + trans[1]))
            pygame.draw.polygon(self.surf, color, poly_px)

        # Car Hull
        for f in self.car.hull.fixtures:
            poly_px = []
            for v in f.shape.vertices:
                world_v = self.car.hull.transform * v
                vec = pygame.math.Vector2(world_v).rotate_rad(angle)
                poly_px.append((vec[0]*zoom + trans[0], vec[1]*zoom + trans[1]))
            pygame.draw.polygon(self.surf, self.car.hull.color, poly_px)

        # Wheels
        for w in self.car.wheels:
            for f in w.fixtures:
                poly_px = []
                for v in f.shape.vertices:
                    world_v = w.transform * v
                    vec = pygame.math.Vector2(world_v).rotate_rad(angle)
                    poly_px.append((vec[0]*zoom + trans[0], vec[1]*zoom + trans[1]))
                pygame.draw.polygon(self.surf, (0,0,0), poly_px)

        # Lidar (Draw Green Lines)
        if hasattr(self, 'lidar_points'):
            car_pos = self.car.hull.position
            c_vec = pygame.math.Vector2(car_pos).rotate_rad(angle)
            c_px = (c_vec[0]*zoom + trans[0], c_vec[1]*zoom + trans[1])
            
            for p in self.lidar_points:
                p_vec = pygame.math.Vector2(p).rotate_rad(angle)
                p_px = (p_vec[0]*zoom + trans[0], p_vec[1]*zoom + trans[1])
                pygame.draw.line(self.surf, (0, 255, 0), c_px, p_px, 1)

        self.surf = pygame.transform.flip(self.surf, False, True) # Not strictly needed if logic matches
        
        # Pygame Flip hack for coordinate system match
        # Actually standard CarRacing doesn't flip at end, it handles coords inside.
        # But our Lidar math assumes Box2D coords. 
        # Simpler: Just blit.
        
        if self.render_mode == "human":
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2))

    def close(self):
        if self.screen: pygame.display.quit(); pygame.quit()
