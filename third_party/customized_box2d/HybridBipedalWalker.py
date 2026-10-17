import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import EzPickle
import Box2D
from Box2D.b2 import (
    circleShape,      # <--- added this
    edgeShape,
    fixtureDef,
    polygonShape,
    revoluteJointDef,
    contactListener,
)
# --- Constants copied from original ---
FPS = 50
SCALE = 30.0
MOTORS_TORQUE = 80
SPEED_HIP = 4
SPEED_KNEE = 6
LIDAR_RANGE = 160 / SCALE
INITIAL_RANDOM = 5
HULL_POLY = [(-30, +9), (+6, +9), (+34, +1), (+34, -8), (-30, -8)]
LEG_DOWN = -8 / SCALE
LEG_W, LEG_H = 8 / SCALE, 34 / SCALE
VIEWPORT_W = 600
VIEWPORT_H = 400
TERRAIN_STEP = 14 / SCALE
TERRAIN_LENGTH = 200
TERRAIN_PRE_FINISH_FLAT = 10
TERRAIN_POST_FINISH_FLAT = 50
TOTAL_TERRAIN_LENGTH = TERRAIN_LENGTH + TERRAIN_POST_FINISH_FLAT
TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4
TERRAIN_GRASS = 10
TERRAIN_STARTPAD = 20
FRICTION = 2.5  # Default friction
FINISH_BONUS = 100.0
# Finish is at the start of the post-finish flat segment.
FINISH_X = TERRAIN_LENGTH * TERRAIN_STEP
# Total terrain extends beyond finish for visuals/stability.
TOTAL_TERRAIN_X_MAX = TOTAL_TERRAIN_LENGTH * TERRAIN_STEP
TIME_PENALTY = 0.01
HEIGHT_CLIP = 5.0
N_LIDAR_RAYS = 20
LIDAR_ANGLE_START = -0.5
LIDAR_ANGLE_END = 1.5
# No progress termination constants
NO_PROGRESS_THRESHOLD_STEPS = 100  # Steps (10 seconds at 50 FPS) without progress
NO_PROGRESS_DISTANCE = 1.0 / SCALE  # Minimum distance to count as progress
NO_PROGRESS_PENALTY = -10.0  # Small penalty for no progress termination
# --- Fixture Definitions ---
HULL_FD = fixtureDef(
    shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in HULL_POLY]),
    density=5.0,
    friction=0.1,
    categoryBits=0x0020,
    maskBits=0x001,
    restitution=0.0,
)

LEG_FD = fixtureDef(
    shape=polygonShape(box=(LEG_W / 2, LEG_H / 2)),
    density=1.0,
    restitution=0.0,
    categoryBits=0x0020,
    maskBits=0x001,
)

LOWER_FD = fixtureDef(
    shape=polygonShape(box=(0.8 * LEG_W / 2, LEG_H / 2)),
    density=1.0,
    restitution=0.0,
    categoryBits=0x0020,
    maskBits=0x001,
)

class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        if self.env.hull == contact.fixtureA.body or self.env.hull == contact.fixtureB.body:
            self.env.game_over = True
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = True
    def EndContact(self, contact):
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = False

class HybridBipedalWalker(gym.Env, EzPickle):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(
        self,
        render_mode: str | None = None,
        mode: str = "learner",
        use_height_map: bool = False,
        height_map_dim: int = 20,
        min_ground_steps: int = 5,
    ):
        """
        mode:
            "learner": Mixed terrain, friction bound to terrain type.
            "climber": Expert 1 - All Stairs, High Friction.
            "skater":  Expert 2 - All Grass/Stump, Low Friction.
            "jumper":  Expert 3 - All Pits, Normal Friction.
        """
        assert not use_height_map, "height_map is wrong"
        EzPickle.__init__(self, render_mode, mode, use_height_map, height_map_dim, min_ground_steps)
        self.isopen = True
        self.world = Box2D.b2World()
        self.terrain = []
        self.hull = None
        self.prev_shaping = None
        
        # --- Context Configuration ---
        self.mode = mode
        self.use_height_map = use_height_map
        self.height_map_dim = int(height_map_dim)
        self.min_ground_steps = max(1, int(min_ground_steps))
        self.FRICTION_ICE = 1.2    # Extremely slippery
        self.FRICTION_NORM = 2.5    # Standard
        self.FRICTION_GRIP = 4.5    # Sticky like glue

        self.fd_polygon = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)]),
            friction=FRICTION,
            # Ensure terrain polygons collide with the robot and are seen by lidar.
            categoryBits=0x0001,
            maskBits=0xFFFF,
        )
        self.fd_edge = fixtureDef(
            shape=edgeShape(vertices=[(0, 0), (1, 1)]),
            friction=FRICTION,
            categoryBits=0x0001,
            maskBits=0xFFFF,
        )

        # Standard Action/Obs spaces
        # Re-using strict bounds from original for safety
        base_low = [
            -math.pi, -5.0, -5.0, -5.0,
            -math.pi, -5.0, -math.pi, -5.0,
            -0.0, -math.pi, -5.0, -math.pi, -5.0, -0.0,
        ]
        base_high = [
            math.pi, 5.0, 5.0, 5.0,
            math.pi, 5.0, math.pi, 5.0,
            1.0, math.pi, 5.0, math.pi, 5.0, 1.0,
        ]
        if self.use_height_map:
            sensor_low = [-HEIGHT_CLIP] * self.height_map_dim
            sensor_high = [HEIGHT_CLIP] * self.height_map_dim
        else:
            sensor_low = [-1.0] *N_LIDAR_RAYS
            sensor_high = [1.0] * N_LIDAR_RAYS
        low = np.array(base_low + sensor_low, dtype=np.float32)
        high = np.array(base_high + sensor_high, dtype=np.float32)
        
        self.action_space = spaces.Box(np.array([-1]*4).astype(np.float32), np.array([1]*4).astype(np.float32))
        self.observation_space = spaces.Box(low, high)

        self.render_mode = render_mode
        self.screen = None
        self.clock = None

    def _destroy(self):
        if not self.terrain: return
        self.world.contactListener = None
        for t in self.terrain: self.world.DestroyBody(t)
        self.terrain = []
        self.world.DestroyBody(self.hull)
        self.hull = None
        for leg in self.legs: self.world.DestroyBody(leg)
        self.legs = []
        self.joints = []
    
    def _generate_terrain(self):
        GRASS, STUMP, STAIRS, PIT, _STATES_ = range(5)
        state = GRASS
        velocity = 0.0
        y = TERRAIN_HEIGHT
        counter = TERRAIN_STARTPAD
        oneshot = False
        self.terrain = []
        self.terrain_x = []
        self.terrain_y = []
        
        self.terrain_properties = []
        self.terrain_type = []

        stair_steps, stair_width, stair_height = 0, 0, 0
        stairs_total = 0
        original_y = 0

        ground_is_ice = False
        ground_steps_left = self.min_ground_steps
        
        finish_idx = TERRAIN_LENGTH
        grass_start_idx = finish_idx - TERRAIN_PRE_FINISH_FLAT
        for i in range(TOTAL_TERRAIN_LENGTH):
            x = i * TERRAIN_STEP
            self.terrain_x.append(x)

            # --- 1. Determine Local Friction ---
            current_friction = self.FRICTION_NORM
            
            force_grass_zone = grass_start_idx <= i < (finish_idx + TERRAIN_POST_FINISH_FLAT)
            if force_grass_zone and not oneshot:
                # Fixed runway (pre-finish) + post-finish flat ground.
                state = GRASS
            if self.mode == "learner":
                if force_grass_zone:
                    ground_is_ice = False
                    ground_steps_left = max(ground_steps_left, self.min_ground_steps)
                elif ground_steps_left <= 0 and state == GRASS:
                    if ground_is_ice:
                        # Force at least one grass segment between ice segments.
                        ground_is_ice = False
                    else:
                        # Make ice less frequent than grass.
                        ground_is_ice = self.np_random.random() < 0.5
                    ground_steps_left = self.min_ground_steps

                if state == STAIRS:
                    current_friction = self.FRICTION_GRIP
                    ground_is_ice = False
                elif state in [PIT, STUMP]:
                    current_friction = self.FRICTION_NORM
                    ground_is_ice = False
                else:
                    current_friction = self.FRICTION_ICE if ground_is_ice else self.FRICTION_NORM
            elif self.mode == "climber":
                current_friction = self.FRICTION_GRIP
            elif self.mode == "skater":
                current_friction = self.FRICTION_ICE
            elif self.mode == "jumper":
                current_friction = self.FRICTION_NORM
            else:
                current_friction = self.FRICTION_NORM

            self.terrain_properties.append(current_friction)
            self.fd_polygon.friction = current_friction

            # --- 2. Terrain Generation Loop ---
            if force_grass_zone:
                # Smoothly clamp to flat height to avoid sudden jumps.
                y = 0.99 * y + 0.01 * TERRAIN_HEIGHT
            elif state == GRASS and not oneshot:
                velocity = 0.8 * velocity + 0.01 * np.sign(TERRAIN_HEIGHT - y)
                if i > TERRAIN_STARTPAD: velocity += self.np_random.uniform(-1, 1) / SCALE
                y += velocity

            elif state == PIT and oneshot:
                counter = self.np_random.integers(3, 6)
                poly = [(x, y), (x + TERRAIN_STEP, y), (x + TERRAIN_STEP, y - 4 * TERRAIN_STEP), (x, y - 4 * TERRAIN_STEP)]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon); t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                self.terrain.append(t)
                self.fd_polygon.shape.vertices = [(p[0] + TERRAIN_STEP * counter, p[1]) for p in poly]
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon); t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                self.terrain.append(t)
                counter += 2
                # counter += 1
                original_y = y

            elif state == PIT and not oneshot:
                y = original_y
                if counter > 1: y -= 4 * TERRAIN_STEP

            elif state == STUMP and oneshot:
                counter = self.np_random.integers(1, 2)
                poly = [(x, y), (x + counter * TERRAIN_STEP, y), (x + counter * TERRAIN_STEP, y + counter * TERRAIN_STEP), (x, y + counter * TERRAIN_STEP)]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon); t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                self.terrain.append(t)

            elif state == STAIRS and oneshot:
                # === Fix 2: choose stair direction intelligently to avoid climbing forever ===
                # If the current height is far above the reference height (for example 20 units),
                # force the stairs downward. If it is too low, force them upward. Otherwise randomize.
                if y > TERRAIN_HEIGHT + 15 * TERRAIN_STEP:
                    stair_height = -1
                elif y < TERRAIN_HEIGHT - 3 * TERRAIN_STEP:
                    stair_height = +1
                else:
                    stair_height = +1 if self.np_random.random() > 0.5 else -1
                
                stair_width = self.np_random.integers(4, 5)
                stair_steps = self.np_random.integers(3, 5)
                stairs_total = stair_steps * stair_width
                original_y = y
                for s in range(stair_steps):
                    poly = [
                        (x + (s * stair_width) * TERRAIN_STEP, y + (s * stair_height) * TERRAIN_STEP),
                        (x + ((1 + s) * stair_width) * TERRAIN_STEP, y + (s * stair_height) * TERRAIN_STEP),
                        (x + ((1 + s) * stair_width) * TERRAIN_STEP, y + (-1 + s * stair_height) * TERRAIN_STEP),
                        (x + (s * stair_width) * TERRAIN_STEP, y + (-1 + s * stair_height) * TERRAIN_STEP),
                    ]
                    self.fd_polygon.shape.vertices = poly
                    t = self.world.CreateStaticBody(fixtures=self.fd_polygon); t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                    self.terrain.append(t)
                counter = stairs_total

            elif state == STAIRS and not oneshot:
                # Keep y piecewise-constant within each stair tread.
                progressed = stairs_total - counter
                step_idx = progressed // stair_width
                y = original_y + step_idx * stair_height * TERRAIN_STEP

            # --- 3. State Transition Logic (Updated for Safety) ---
            oneshot = False
            self.terrain_y.append(y)
            self.terrain_type.append(state)
            counter -= 1
            if self.mode == "learner":
                ground_steps_left -= 1
            
            if not force_grass_zone and counter == 0:
                # Randomly generate the length of the buffer zone
                counter = self.np_random.integers(TERRAIN_GRASS / 2, TERRAIN_GRASS)
                
                # === Fix 1: enforce the buffer logic ===
                # If the current state is an obstacle (not grass), force the next state back to grass.
                # This guarantees spacing between obstacles.
                if state != GRASS:
                    state = GRASS
                    oneshot = True
                
                else:
                    # If the current state is grass, the buffer period is over and a new obstacle can be generated
                    if not force_grass_zone:
                        if self.mode == "learner":
                            if ground_is_ice:
                                state = GRASS
                                oneshot = False
                            else:
                                state = self.np_random.integers(1, _STATES_)
                                if state == STAIRS:
                                    max_stairs_total = 4 * 4  # max_steps * max_width
                                    if i >= grass_start_idx - max_stairs_total:
                                        state = GRASS
                                        oneshot = False
                                    else:
                                        oneshot = True
                                else:
                                    oneshot = True
                        elif self.mode == "climber":
                            max_stairs_total = 4 * 4  # max_steps * max_width
                            if i >= grass_start_idx - max_stairs_total:
                                state = GRASS
                                oneshot = False
                            else:
                                state = STAIRS
                                oneshot = True
                        
                        elif self.mode == "skater":
                            state = STUMP if self.np_random.random() > 0.8 else GRASS
                            oneshot = True
                        
                        elif self.mode == "jumper":
                            max_pit_total = 8
                            if i >= grass_start_idx - max_pit_total:
                                state = GRASS
                                oneshot = False
                            else:
                                state = PIT
                                oneshot = True
                            
                        else:
                            state = GRASS
                            oneshot = False

        # --- 4. Edge Generation ---
        self.terrain_poly = []
        for i in range(TOTAL_TERRAIN_LENGTH - 1):
            if self.terrain_type[i] == STAIRS:
                continue
            poly = [(self.terrain_x[i], self.terrain_y[i]), (self.terrain_x[i + 1], self.terrain_y[i + 1])]
            self.fd_edge.friction = self.terrain_properties[i]
            self.fd_edge.shape.vertices = poly
            t = self.world.CreateStaticBody(fixtures=self.fd_edge)
            
            fric = self.terrain_properties[i]
            if fric <= self.FRICTION_ICE + 0.01:
                color = (150, 255, 255) 
            elif fric >= self.FRICTION_GRIP - 0.01:
                color = (139, 69, 19)   
            else:
                color = (76, 255 if i % 2 == 0 else 204, 76) 
                
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            poly += [(poly[1][0], 0), (poly[0][0], 0)]
            self.terrain_poly.append((poly, fric))
        self.terrain.reverse()

    def _generate_clouds(self):
        # Keep original simple cloud gen
        self.cloud_poly = []
        for i in range(TERRAIN_LENGTH // 20):
            x = self.np_random.uniform(0, TERRAIN_LENGTH) * TERRAIN_STEP
            y = VIEWPORT_H / SCALE * 3 / 4
            poly = [
                (x + 15 * TERRAIN_STEP * math.sin(3.14 * 2 * a / 5) + self.np_random.uniform(0, 5 * TERRAIN_STEP),
                 y + 5 * TERRAIN_STEP * math.cos(3.14 * 2 * a / 5) + self.np_random.uniform(0, 5 * TERRAIN_STEP))
                for a in range(5)
            ]
            x1 = min(p[0] for p in poly)
            x2 = max(p[0] for p in poly)
            self.cloud_poly.append((poly, x1, x2))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed) # Handles EzPickle and seeding
        self._destroy()
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.game_over = False
        self.prev_shaping = None
        self.scroll = 0.0
        self.lidar_render = 0

        self._generate_terrain() # No args needed now
        self._generate_clouds()

        init_x = TERRAIN_STEP * TERRAIN_STARTPAD / 2
        # No progress tracking
        self.last_progress_x = init_x
        self.no_progress_steps = 0
        init_y = TERRAIN_HEIGHT + 2 * LEG_H
        self.hull = self.world.CreateDynamicBody(position=(init_x, init_y), fixtures=HULL_FD)
        self.hull.color1 = (127, 51, 229)
        self.hull.color2 = (76, 76, 127)
        self.hull.ApplyForceToCenter((self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM), 0), True)

        self.legs = []
        self.joints = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(position=(init_x, init_y - LEG_H / 2 - LEG_DOWN), angle=(i * 0.05), fixtures=LEG_FD)
            leg.color1 = (153 - i * 25, 76 - i * 25, 127 - i * 25)
            leg.color2 = (102 - i * 25, 51 - i * 25, 76 - i * 25)
            rjd = revoluteJointDef(
                bodyA=self.hull, bodyB=leg, localAnchorA=(0, LEG_DOWN), localAnchorB=(0, LEG_H / 2),
                enableMotor=True, enableLimit=True, maxMotorTorque=MOTORS_TORQUE, motorSpeed=i,
                lowerAngle=-0.8, upperAngle=1.1,
            )
            self.legs.append(leg)
            self.joints.append(self.world.CreateJoint(rjd))

            lower = self.world.CreateDynamicBody(position=(init_x, init_y - LEG_H * 3 / 2 - LEG_DOWN), angle=(i * 0.05), fixtures=LOWER_FD)
            lower.color1 = (153 - i * 25, 76 - i * 25, 127 - i * 25)
            lower.color2 = (102 - i * 25, 51 - i * 25, 76 - i * 25)
            rjd = revoluteJointDef(
                bodyA=leg, bodyB=lower, localAnchorA=(0, -LEG_H / 2), localAnchorB=(0, LEG_H / 2),
                enableMotor=True, enableLimit=True, maxMotorTorque=MOTORS_TORQUE, motorSpeed=1,
                lowerAngle=-1.6, upperAngle=-0.1,
            )
            lower.ground_contact = False
            self.legs.append(lower)
            self.joints.append(self.world.CreateJoint(rjd))

        self.drawlist = self.terrain + self.legs + [self.hull]
        
        class LidarCallback(Box2D.b2.rayCastCallback):
            def ReportFixture(self, fixture, point, normal, fraction):
                if (fixture.filterData.categoryBits & 1) == 0: return -1
                self.p2 = point
                self.fraction = fraction
                return fraction

        self.lidar = [LidarCallback() for _ in range(N_LIDAR_RAYS)]
        if self.render_mode == "human": self.render()
        return self.step(np.array([0, 0, 0, 0]))[0], {}

    def step(self, action: np.ndarray):
        # ... (Standard Step Logic, mostly identical to original) ...
        assert self.hull is not None
        a = np.clip(action, -1.0, 1.0)
        # self.joints[0].motorSpeed = float(SPEED_HIP * a[0])
        # self.joints[0].maxMotorTorque = float(MOTORS_TORQUE * np.abs(a[0]))
        # self.joints[1].motorSpeed = float(SPEED_KNEE * a[1])
        # self.joints[1].maxMotorTorque = float(MOTORS_TORQUE * np.abs(a[1]))
        # self.joints[2].motorSpeed = float(SPEED_HIP * a[2])
        # self.joints[2].maxMotorTorque = float(MOTORS_TORQUE * np.abs(a[2]))
        # self.joints[3].motorSpeed = float(SPEED_KNEE * a[3])
        # self.joints[3].maxMotorTorque = float(MOTORS_TORQUE * np.abs(a[3]))
        
        self.joints[0].motorSpeed = float(SPEED_HIP * np.sign(action[0]))
        self.joints[0].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[0]), 0, 1)
            )
        self.joints[1].motorSpeed = float(SPEED_KNEE * np.sign(action[1]))
        self.joints[1].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[1]), 0, 1)
            )
        self.joints[2].motorSpeed = float(SPEED_HIP * np.sign(action[2]))
        self.joints[2].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[2]), 0, 1)
            )
        self.joints[3].motorSpeed = float(SPEED_KNEE * np.sign(action[3]))
        self.joints[3].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[3]), 0, 1)
            )


        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.hull.position
        vel = self.hull.linearVelocity

        state = [
            self.hull.angle, 2.0 * self.hull.angularVelocity / FPS,
            0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS, 0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,
            self.joints[0].angle, self.joints[0].speed / SPEED_HIP,
            self.joints[1].angle + 1.0, self.joints[1].speed / SPEED_KNEE,
            1.0 if self.legs[1].ground_contact else 0.0,
            self.joints[2].angle, self.joints[2].speed / SPEED_HIP,
            self.joints[3].angle + 1.0, self.joints[3].speed / SPEED_KNEE,
            1.0 if self.legs[3].ground_contact else 0.0,
        ]
        if self.use_height_map:
            height_samples = []
            hull_y = pos.y
            max_index = len(self.terrain_y) - 1
            half_window = self.height_map_dim // 2
            for i in range(self.height_map_dim):
                offset = i - half_window
                target_x = pos.x + offset * TERRAIN_STEP
                idx = int(target_x / TERRAIN_STEP)
                if idx < 0:
                    idx = 0
                if idx > max_index:
                    if max_index >= 0:
                        terrain_y = self.terrain_y[max_index]
                    else:
                        terrain_y = 0.0
                else:
                    terrain_y = self.terrain_y[idx]
                h_rel = terrain_y - hull_y
                height_samples.append(float(np.clip(h_rel, -HEIGHT_CLIP, HEIGHT_CLIP)))
            state += height_samples
        else:
            for i in range(N_LIDAR_RAYS):
                t = i / (N_LIDAR_RAYS - 1)
                angle = LIDAR_ANGLE_START + t * (LIDAR_ANGLE_END - LIDAR_ANGLE_START)
                self.lidar[i].fraction = 1.0
                self.lidar[i].p1 = pos
                self.lidar[i].p2 = (
                    # pos[0] + math.sin(1.5 * i / 10.0) * LIDAR_RANGE,
                    # pos[1] - math.cos(1.5 * i / 10.0) * LIDAR_RANGE,
                    pos[0] + math.sin(angle) * LIDAR_RANGE,
                    pos[1] - math.cos(angle) * LIDAR_RANGE,
                )
                self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)
            state += [l.fraction for l in self.lidar]
        assert len(state) == (14 + (self.height_map_dim if self.use_height_map else N_LIDAR_RAYS))

        self.scroll = pos.x - VIEWPORT_W / SCALE / 5
        shaping = (130 * pos[0] / SCALE)
        shaping -= 5.0 * abs(state[0])
        reward = 0
        if self.prev_shaping is not None: reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        for a in action: reward -= 0.00035 * MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)
        # Small per-step penalty encourages finishing earlier without dominating shaping.
        reward -= TIME_PENALTY

        terminated = False
        success = False
        if self.game_over or pos[0] < 0:
            reward = -100
            terminated = True
        if pos[0] > FINISH_X:
            reward += FINISH_BONUS
            terminated = True
            success = True
        
        # Check for no progress termination
        if not terminated:
            progress_distance = pos[0] - self.last_progress_x
            if progress_distance >= NO_PROGRESS_DISTANCE:
                # Made progress, reset counter and update last progress position
                self.last_progress_x = pos[0]
                self.no_progress_steps = 0
            else:
                # No progress, increment counter
                self.no_progress_steps += 1
                if self.no_progress_steps >= NO_PROGRESS_THRESHOLD_STEPS:
                    reward += NO_PROGRESS_PENALTY
                    terminated = True

        if self.render_mode == "human": self.render()
        return np.array(state, dtype=np.float32), reward, terminated, False, {"is_success": success}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[box2d]"`'
            ) from e

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface(
            (VIEWPORT_W + max(0.0, self.scroll) * SCALE, VIEWPORT_H)
        )

        pygame.transform.scale(self.surf, (SCALE, SCALE))

        # 1. Draw the sky
        pygame.draw.polygon(
            self.surf,
            color=(215, 215, 255),
            points=[
                (self.scroll * SCALE, 0),
                (self.scroll * SCALE + VIEWPORT_W, 0),
                (self.scroll * SCALE + VIEWPORT_W, VIEWPORT_H),
                (self.scroll * SCALE, VIEWPORT_H),
            ],
        )

        # 2. Draw the clouds
        for poly, x1, x2 in self.cloud_poly:
            if x2 < self.scroll / 2:
                continue
            if x1 > self.scroll / 2 + VIEWPORT_W / SCALE:
                continue
            pygame.draw.polygon(
                self.surf,
                color=(255, 255, 255),
                points=[
                    (p[0] * SCALE + self.scroll * SCALE / 2, p[1] * SCALE) for p in poly
                ],
            )
            gfxdraw.aapolygon(
                self.surf,
                [(p[0] * SCALE + self.scroll * SCALE / 2, p[1] * SCALE) for p in poly],
                (255, 255, 255),
            )

        # 3. Draw the terrain, including friction visualization
        for i, (poly, fric) in enumerate(self.terrain_poly):
            if poly[1][0] < self.scroll:
                continue
            if poly[0][0] > self.scroll + VIEWPORT_W / SCALE:
                continue
            
            # --- Core change: set fill color based on terrain_properties ---
            # Default to green
            color = (102, 153, 76)
            
            if fric is not None:
                # Ice: fill with light blue
                if fric <= self.FRICTION_ICE + 0.01:
                    color = (180, 220, 255)
                # High-grip terrain: fill with dark brown
                elif fric >= self.FRICTION_GRIP - 0.01:
                    color = (100, 50, 10)
            # --------------------------------------------------

            scaled_poly = []
            for coord in poly:
                scaled_poly.append([coord[0] * SCALE, coord[1] * SCALE])
            
            pygame.draw.polygon(self.surf, color=color, points=scaled_poly)
            gfxdraw.aapolygon(self.surf, scaled_poly, color)

        # 4. Draw the lidar rays
        self.lidar_render = (self.lidar_render + 1) % 100
        i = self.lidar_render
        if i < 2 * len(self.lidar):
            single_lidar = (
                self.lidar[i]
                if i < len(self.lidar)
                else self.lidar[len(self.lidar) - i - 1]
            )
            if hasattr(single_lidar, "p1") and hasattr(single_lidar, "p2"):
                pygame.draw.line(
                    self.surf,
                    color=(255, 0, 0),
                    start_pos=(single_lidar.p1[0] * SCALE, single_lidar.p1[1] * SCALE),
                    end_pos=(single_lidar.p2[0] * SCALE, single_lidar.p2[1] * SCALE),
                    width=1,
                )

        # 5. Draw the rigid bodies (hull, legs, terrain edge lines)
        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color1,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color2,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )
                else:
                    path = [trans * v * SCALE for v in f.shape.vertices]
                    if len(path) > 2:
                        pygame.draw.polygon(self.surf, color=obj.color1, points=path)
                        gfxdraw.aapolygon(self.surf, path, obj.color1)
                        path.append(path[0])
                        pygame.draw.polygon(
                            self.surf, color=obj.color2, points=path, width=1
                        )
                        gfxdraw.aapolygon(self.surf, path, obj.color2)
                    else:
                        pygame.draw.aaline(
                            self.surf,
                            start_pos=path[0],
                            end_pos=path[1],
                            color=obj.color1,
                        )

        # 6. Draw the finish flag
        finish_i = min(TERRAIN_LENGTH, len(self.terrain_y) - 1)
        flagy1 = self.terrain_y[finish_i] * SCALE

        flagy2 = flagy1 + 50
        x = FINISH_X * SCALE
        pygame.draw.aaline(
            self.surf, color=(0, 0, 0), start_pos=(x, flagy1), end_pos=(x, flagy2)
        )
        f = [
            (x, flagy2),
            (x, flagy2 - 10),
            (x + 25, flagy2 - 5),
        ]
        pygame.draw.polygon(self.surf, color=(230, 51, 0), points=f)
        pygame.draw.lines(
            self.surf, color=(0, 0, 0), points=f + [f[0]], width=1, closed=False
        )

        self.surf = pygame.transform.flip(self.surf, False, True)

        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (-self.scroll * SCALE, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )[:, -VIEWPORT_W:]

# Example usage
if __name__ == "__main__":
    # Minimal self-test for stability and lidar hits on stairs.
    env = HybridBipedalWalker(render_mode=None, mode="climber")
    obs, _ = env.reset()
    lidar_min = []
    motor_speeds = []
    for _ in range(2000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        lidar_min.append(float(np.min(obs[-10:])))
        motor_speeds.append(env.joints[0].motorSpeed)
        if terminated or truncated:
            obs, _ = env.reset()
    print(f"lidar fraction min/mean: {min(lidar_min):.3f}/{np.mean(lidar_min):.3f}")
    print(f"motorSpeed range: {min(motor_speeds):.3f}..{max(motor_speeds):.3f}")
    env.close()
