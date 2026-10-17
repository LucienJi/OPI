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

# --- Constants ---
FPS = 50
SCALE = 6.0
TRACK_RAD = 900 / SCALE
PLAYFIELD = 2000 / SCALE
TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
MAX_EPISODE_STEPS = 2000
NO_PROGRESS_STEPS = 250
OUT_OF_BOUNDS_PENALTY = -100.0
MAX_WHEEL_OMEGA = 80.0
MAX_HULL_SPEED = 60.0
MAX_HULL_ANGULAR_VEL = 5.0
PROGRESS_REWARD_SCALE = 1.0
OFF_ROAD_PENALTY = -0.2
CRASH_PENALTY = -500.0
CRASH_SPEED_THRESHOLD = 0.5

WINDOW_W = 1000
WINDOW_H = 800
ZOOM = 2.7

# Lidar Settings
LIDAR_RAYS = 32
LIDAR_RANGE = 50
LIDAR_FOV = 3.14159

# --- Weather Configs ---
WEATHER_TYPES = ["Sunny", "Rainy", "Foggy", "Snowy"]
WEATHER_CONFIGS = {
    "Sunny": {"friction": 1.0, "lidar_noise": 0.00, "color_mod": (0, 0, 0)},
    "Rainy": {"friction": 0.5, "lidar_noise": 0.1, "color_mod": (-30, -30, 20)},
    "Foggy": {"friction": 1.0, "lidar_noise": 0.20, "color_mod": (40, 40, 40)},
    "Snowy": {"friction": 0.3, "lidar_noise": 0.1, "color_mod": (120, 120, 120)}
}

# --- Vehicle Config (Prototype Only) ---
VEHICLE_CONFIG = {
    "engine_power_mul": 80_000_000,
    "friction_limit_mul": 1_000_000,
    "wheel_moment": 4_000,
    "hull_poly": [(-50, +140), (+50, +140), (+50, -100), (-50, -100)],
    "wheel_pos": [(-50, +90), (+50, +90), (-50, -90), (+50, -90)],
    "drive": "AWD",
    "size": 0.02,
    "color": (0, 200, 0)
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
            b1, b2 = contact.fixtureA.body, contact.fixtureB.body
            if getattr(b1.userData, "is_border", False) or getattr(b2.userData, "is_border", False):
                if self.env.car and (b1 == self.env.car.hull or b1 in self.env.car.wheels or b2 == self.env.car.hull or b2 in self.env.car.wheels):
                    v = self.env.car.hull.linearVelocity
                    if math.sqrt(v[0]**2 + v[1]**2) >= CRASH_SPEED_THRESHOLD:
                        self.env.crashed = True
        
        tile, obj = None, None
        u1, u2 = contact.fixtureA.body.userData, contact.fixtureB.body.userData
        if u1 and hasattr(u1, "road_friction"): tile, obj = u1, u2
        if u2 and hasattr(u2, "road_friction"): tile, obj = u2, u1
        
        if not tile or not obj or not hasattr(obj, "tiles"): return
        
        if begin:
            obj.tiles.add(tile)
            if not tile.road_visited:
                tile.road_visited = True
                self.env.reward += 1000.0 / len(self.env.track)
                self.env.tile_visited_count += 1
                if tile.idx == 0 and (self.env.tile_visited_count / len(self.env.track)) >= self.lap_complete_percent:
                    self.env.new_lap = True
        else:
            obj.tiles.remove(tile)

class DynamicCar:
    def __init__(self, world, init_angle, init_x, init_y):
        self.world = world
        self.cfg = VEHICLE_CONFIG
        SIZE = self.cfg["size"]
        self.engine_power = self.cfg["engine_power_mul"] * SIZE**2
        self.friction_limit = self.cfg["friction_limit_mul"] * SIZE**2
        self.wheel_moment = self.cfg["wheel_moment"] * SIZE**2
        
        # Hull
        self.hull = self.world.CreateDynamicBody(
            position=(init_x, init_y), angle=init_angle,
            fixtures=[fixtureDef(shape=polygonShape(vertices=[(x*SIZE, y*SIZE) for x, y in self.cfg["hull_poly"]]), density=1.0)]
        )
        self.hull.color = self.cfg["color"]
        self.hull.userData = self.hull
        
        # Wheels
        self.wheels = []
        W_R, W_W = 27, 14
        W_POLY = [(-W_W*SIZE, +W_R*SIZE), (+W_W*SIZE, +W_R*SIZE), (+W_W*SIZE, -W_R*SIZE), (-W_W*SIZE, -W_R*SIZE)]
        
        for wx, wy in self.cfg["wheel_pos"]:
            w = self.world.CreateDynamicBody(
                position=(init_x + wx*SIZE, init_y + wy*SIZE), angle=init_angle,
                fixtures=fixtureDef(shape=polygonShape(vertices=W_POLY), density=0.1, categoryBits=0x0020, maskBits=0x0005, restitution=0.0)
            )
            w.wheel_rad = W_R * SIZE
            w.gas = 0.0; w.brake = 0.0; w.steer = 0.0; w.omega = 0.0; w.tiles = set()
            w.userData = w
            
            w.joint = self.world.CreateJoint(revoluteJointDef(
                bodyA=self.hull, bodyB=w, localAnchorA=(wx*SIZE, wy*SIZE), localAnchorB=(0, 0),
                enableMotor=True, enableLimit=True, maxMotorTorque=180*900*SIZE**2, lowerAngle=-0.4, upperAngle=+0.4
            ))
            self.wheels.append(w)

    def step(self, dt, action):
        steer, gas, brake = -action[0], action[1], action[2]
        self.wheels[0].steer = steer
        self.wheels[1].steer = steer
        drive = self.cfg["drive"]
        
        for i, w in enumerate(self.wheels):
            w.gas = gas if drive=="AWD" or (drive=="FWD" and i<2) or (drive=="RWD" and i>=2) else 0.0
            w.brake = brake
            w.joint.motorSpeed = float(np.sign(w.steer - w.joint.angle) * min(50.0 * abs(w.steer - w.joint.angle), 3.0))
            
            # Physics response
            f_limit = self.friction_limit * 0.6
            if len(w.tiles) > 0:
                f_limit = max([self.friction_limit * t.road_friction for t in w.tiles])
            
            forw, side = w.GetWorldVector((0, 1)), w.GetWorldVector((1, 0))
            v = w.linearVelocity
            vf, vs = forw[0]*v[0] + forw[1]*v[1], side[0]*v[0] + side[1]*v[1]
            
            w.omega += dt * self.engine_power * w.gas / self.wheel_moment / (abs(w.omega) + 5.0)
            if w.brake >= 0.9: w.omega = 0
            elif w.brake > 0: w.omega += -np.sign(w.omega) * min(15 * w.brake, abs(w.omega))
            
            f_force = (-vf + w.omega * w.wheel_rad) * 205000 * self.cfg["size"]**2
            p_force = -vs * 205000 * self.cfg["size"]**2
            
            force = math.sqrt(f_force**2 + p_force**2)
            if abs(force) > f_limit:
                f_force, p_force = f_force/force * f_limit, p_force/force * f_limit
            
            w.omega -= dt * f_force * w.wheel_rad / self.wheel_moment
            w.omega = np.clip(w.omega, -MAX_WHEEL_OMEGA, MAX_WHEEL_OMEGA)
            w.ApplyForceToCenter((float(p_force*side[0] + f_force*forw[0]), float(p_force*side[1] + f_force*forw[1])), True)

    def destroy(self):
        self.world.DestroyBody(self.hull)
        for w in self.wheels: self.world.DestroyBody(w)

class LidarCallback(rayCastCallback):
    def __init__(self, ignore):
        super().__init__()
        self.ignore = ignore
        self.fraction = 1.0
        self.point = None

    def ReportFixture(self, fixture, point, normal, fraction):
        if fixture.body in self.ignore or fixture.sensor: return -1.0
        self.fraction = fraction
        self.point = point
        return fraction

class HybridWeather(gym.Env, EzPickle):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}
    
    def __init__(self, render_mode: str | None = None, mode: str = "learner", reward_mode: str = "tiles"):
        EzPickle.__init__(self, render_mode, mode, reward_mode)
        self.mode = mode
        self.reward_mode = reward_mode
        self.lap_complete_percent = 0.95
        self.road_color = np.array([102, 102, 102])
        self.contactListener_keepref = FrictionDetector(self, self.lap_complete_percent)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.screen, self.clock, self.car, self.road, self.borders = None, None, None, [], []
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14 + LIDAR_RAYS,), dtype=np.float32)
        self.action_space = spaces.Box(np.array([-1, 0, 0]).astype(np.float32), np.array([+1, +1, +1]).astype(np.float32))
        self.render_mode = render_mode

    def _destroy(self):
        if not self.road: return
        for t in self.road: self.world.DestroyBody(t)
        self.road = []
        for b in self.borders: self.world.DestroyBody(b)
        self.borders = []
        if self.car: self.car.destroy()

    def _create_track(self):
        CHECKPOINTS = 12
        checkpoints = []
        for c in range(CHECKPOINTS):
            alpha = 2 * math.pi * c / CHECKPOINTS + self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
            rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)
            if c == 0: alpha, rad = 0, 1.5 * TRACK_RAD
            if c == CHECKPOINTS - 1: 
                alpha = 2 * math.pi * c / CHECKPOINTS
                self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                rad = 1.5 * TRACK_RAD
            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))
        
        x, y, beta, dest_i, laps, track, no_freeze, visited_other_side = 1.5 * TRACK_RAD, 0, 0, 0, 0, [], 2500, False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0: laps += 1; visited_other_side = False
            if alpha < 0: visited_other_side, alpha = True, alpha + 2 * math.pi
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
            r1x, r1y = math.cos(beta), math.sin(beta)
            p1x, p1y = -r1y, r1x
            dest_dx, dest_dy = dest_x - x, dest_y - y
            proj = r1x * dest_dx + r1y * dest_dy
            while beta - alpha > 1.5 * math.pi: beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi: beta += 2 * math.pi
            prev_beta = beta
            if proj > 0.3: beta -= min(TRACK_TURN_RATE, abs(0.001 * proj * SCALE))
            if proj < -0.3: beta += min(TRACK_TURN_RATE, abs(0.001 * proj * SCALE))
            x += p1x * TRACK_DETAIL_STEP; y += p1y * TRACK_DETAIL_STEP
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
            if laps > 4 or no_freeze == 0: break
            no_freeze -= 1
        
        i1, i2, i = -1, -1, len(track)
        while True:
            i -= 1
            if i == 0: return False
            pass_through_start = track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha
            if pass_through_start and i2 == -1: i2 = i
            elif pass_through_start and i1 == -1: i1 = i; break
        track = track[i1 : i2 - 1]
        
        self.road_poly, self.borders, self.road = [], [], []
        fd_tile = fixtureDef(shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)]))
        
        ZONE_SIZE = 15
        curr_weather = "Sunny"
        if self.mode != "learner":
            curr_weather = self.mode.capitalize()
            if curr_weather not in WEATHER_TYPES: curr_weather = "Sunny"

        for i in range(len(track)):
            if self.mode == "learner" and i % ZONE_SIZE == 0:
                curr_weather = self.np_random.choice(WEATHER_TYPES)
            
            cfg = WEATHER_CONFIGS[curr_weather]
            
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            l1 = (x1 - TRACK_WIDTH * math.cos(beta1), y1 - TRACK_WIDTH * math.sin(beta1))
            r1 = (x1 + TRACK_WIDTH * math.cos(beta1), y1 + TRACK_WIDTH * math.sin(beta1))
            l2 = (x2 - TRACK_WIDTH * math.cos(beta2), y2 - TRACK_WIDTH * math.sin(beta2))
            r2 = (x2 + TRACK_WIDTH * math.cos(beta2), y2 + TRACK_WIDTH * math.sin(beta2))
            
            vertices = [l1, r1, r2, l2]
            fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=fd_tile)
            t.userData = t
            t.road_friction, t.lidar_noise, t.weather_type = cfg["friction"], cfg["lidar_noise"], curr_weather
            t.idx, t.road_visited = i, False
            t.fixtures[0].sensor = True
            
            base_col = self.road_color + 0.01 * (i % 3) * 255
            t.color = np.clip(base_col + cfg["color_mod"], 0, 255)
            self.road_poly.append((vertices, t.color))
            self.road.append(t)
            
            b_col = (255, 255, 255) if i % 2 == 0 else (255, 0, 0)
            for side in [-1, 1]:
                b1 = (x1 + side * TRACK_WIDTH * math.cos(beta1), y1 + side * TRACK_WIDTH * math.sin(beta1))
                b2 = (x1 + side * (TRACK_WIDTH+BORDER) * math.cos(beta1), y1 + side * (TRACK_WIDTH+BORDER) * math.sin(beta1))
                b3 = (x2 + side * (TRACK_WIDTH+BORDER) * math.cos(beta2), y2 + side * (TRACK_WIDTH+BORDER) * math.sin(beta2))
                b4 = (x2 + side * TRACK_WIDTH * math.cos(beta2), y2 + side * TRACK_WIDTH * math.sin(beta2))
                self.road_poly.append(([b1, b2, b3, b4], b_col))
                border = self.world.CreateStaticBody(fixtures=fixtureDef(shape=edgeShape(vertices=[b1, b4]), friction=0.0, categoryBits=0x0004))
                border.userData = border
                border.is_border = True
                self.borders.append(border)

        self.track = track
        self.track_points = np.array([(x, y) for (_, _, x, y) in track], dtype=np.float32)
        return True

    def _get_vector_obs(self):
        noises = [t.lidar_noise for w in self.car.wheels for t in w.tiles]
        active_noise = np.mean(noises) if noises else 0.0
        
        lidar_readings, self.lidar_points = [], []
        ignore = {self.car.hull, *self.car.wheels}
        
        car_pos = self.car.hull.position
        car_angle = self.car.hull.angle
        
        for i in range(LIDAR_RAYS):
            angle = (car_angle + math.pi/2) - LIDAR_FOV/2 + (LIDAR_FOV * i / (LIDAR_RAYS - 1))
            
            # Safe offset so the ray does not hit the car itself
            p1 = (car_pos[0] + 0.5 * math.cos(angle), car_pos[1] + 0.5 * math.sin(angle))
            p2 = (car_pos[0] + math.cos(angle)*LIDAR_RANGE, car_pos[1] + math.sin(angle)*LIDAR_RANGE)
            
            cb = LidarCallback(ignore)
            self.world.RayCast(cb, p1, p2)
            
            val = cb.fraction
            if active_noise > 0:
                val = np.clip(val + self.np_random.normal(0, active_noise), 0, 1.0)
                if active_noise > 0.2 and self.np_random.random() < 0.1: val = 1.0
            
            lidar_readings.append(val)
            # Critical fix: convert Box2D vectors to tuples explicitly to avoid render errors
            if cb.point:
                if hasattr(cb.point, "x"):
                    self.lidar_points.append((cb.point.x, cb.point.y))
                else:
                    self.lidar_points.append((cb.point[0], cb.point[1]))
            else:
                self.lidar_points.append(p2)

        vel = self.car.hull.linearVelocity
        local_v = (vel[0]*math.cos(-car_angle) - vel[1]*math.sin(-car_angle), vel[0]*math.sin(-car_angle) + vel[1]*math.cos(-car_angle))
        
        state = [
            local_v[0], local_v[1], 
            self.car.hull.angularVelocity, 
            *[w.omega for w in self.car.wheels], 
            self.car.wheels[0].joint.angle, 
            0, 0, 0, 0, 0, 0 
        ]
        
        return np.concatenate([np.asarray(state, dtype=np.float32)/10.0, np.array(lidar_readings, dtype=np.float32)])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._destroy()
        self.reward, self.prev_reward, self.tile_visited_count, self.t, self.new_lap = 0.0, 0.0, 0, 0.0, False
        self.step_count, self.no_progress_steps, self.last_tile_visited_count, self.crashed, self.crash_penalized = 0, 0, 0, False, False
        
        while not self._create_track(): pass
        
        self.car = DynamicCar(self.world, *self.track[0][1:4])
        self.last_progress_idx = 0
        return self._get_vector_obs(), {}

    def step(self, action):
        if action is not None:
            action = np.clip(action, -1, 1)
            self.car.step(1.0/FPS, action)
        
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS
        
        v = self.car.hull.linearVelocity
        speed = math.sqrt(v[0]**2 + v[1]**2)
        if speed > MAX_HULL_SPEED:
            self.car.hull.linearVelocity = (v[0] * MAX_HULL_SPEED/speed, v[1] * MAX_HULL_SPEED/speed)
        
        self.reward -= 0.1
        
        info = {"is_success": False}

        if self.reward_mode == "progress":
            car_pos = np.array([self.car.hull.position[0], self.car.hull.position[1]])
            search_radius = 10
            track_len = len(self.track)
            indices = np.arange(self.last_progress_idx - search_radius, self.last_progress_idx + search_radius)
            indices = indices % track_len
            
            diffs = self.track_points[indices] - car_pos
            min_idx = np.argmin(np.sum(diffs**2, axis=1))
            progress_idx = indices[min_idx]
            
            delta = progress_idx - self.last_progress_idx
            if delta < -track_len // 2: delta += track_len
            elif delta > track_len // 2: delta -= track_len
            
            self.reward += PROGRESS_REWARD_SCALE * float(delta)
            self.last_progress_idx = progress_idx

        on_road = any(len(w.tiles) > 0 for w in self.car.wheels)
        if not on_road: self.reward += OFF_ROAD_PENALTY
        
        terminated = False
        out_of_bounds = abs(self.car.hull.position[0]) > PLAYFIELD or abs(self.car.hull.position[1]) > PLAYFIELD
        if self.tile_visited_count == len(self.track) or self.new_lap:
            terminated = True
            info["termination"] = "track_complete"
            info["is_success"] = True
        if out_of_bounds:
            terminated = True
            info["termination"] = "out_of_bounds"
        if self.crashed:
            terminated = True
            info["termination"] = "crash"
            if not self.crash_penalized:
                self.reward += CRASH_PENALTY
                self.crash_penalized = True
        
        self.step_count += 1
        if self.tile_visited_count > self.last_tile_visited_count:
            self.no_progress_steps, self.last_tile_visited_count = 0, self.tile_visited_count
        else: self.no_progress_steps += 1
        
        truncated = False
        if self.step_count >= MAX_EPISODE_STEPS:
            truncated = True
            info["truncation"] = "max_steps"
        if self.no_progress_steps >= NO_PROGRESS_STEPS:
            truncated = True
            info["truncation"] = "no_progress"
        step_reward = self.reward - self.prev_reward
        self.prev_reward = self.reward
        
        return self._get_vector_obs(), step_reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None: return
        if self.screen is None:
            pygame.init(); self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        if self.clock is None: self.clock = pygame.time.Clock()
        
        surf = pygame.Surface((WINDOW_W, WINDOW_H))
        angle, zoom = -self.car.hull.angle, ZOOM * SCALE
        scroll_x, scroll_y = -self.car.hull.position[0] * zoom, -self.car.hull.position[1] * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])

        surf.fill((102, 204, 102)) 
        for poly, color in self.road_poly:
            px = [(pygame.math.Vector2(p).rotate_rad(angle)*zoom + trans) for p in poly]
            pygame.draw.polygon(surf, color, px)
        
        for body in [self.car.hull] + self.car.wheels:
            for f in body.fixtures:
                # Extract vertices explicitly and build Vector2 objects for Pygame compatibility
                world_vertices = [(body.transform * v) for v in f.shape.vertices]
                px = [(pygame.math.Vector2(v[0], v[1]).rotate_rad(angle)*zoom + trans) for v in world_vertices]
                pygame.draw.polygon(surf, getattr(body, "color", (0,0,0)), px)

        car_px = pygame.math.Vector2(self.car.hull.position[0], self.car.hull.position[1]).rotate_rad(angle)*zoom + trans
        
        for p in self.lidar_points:
            # Here p is already a safe (x, y) tuple
            p_vec = pygame.math.Vector2(p[0], p[1])
            p_px = p_vec.rotate_rad(angle)*zoom + trans
            pygame.draw.line(surf, (0, 255, 0), car_px, p_px, 1)

        surf = pygame.transform.flip(surf, False, True)
        
        if self.render_mode == "human":
            self.screen.blit(surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            # Critical fix: return image data
            return np.transpose(np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2))

    def close(self):
        if self.screen: pygame.display.quit(); pygame.quit()

if __name__ == "__main__":
    env = HybridWeather(render_mode="human", mode="learner")
    obs, _ = env.reset()
    print(f"Env initialized in mode: {env.mode}")
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated: obs, _ = env.reset()
    env.close()
