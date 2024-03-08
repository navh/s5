import gymnasium as gym
from math import pi
import numpy as np


INITIAL_UNTRACKED_TARGETS = 10
INITIAL_TRACKED_TARGETS = 10
TARGET_ARRIVAL_LAMBDA_PER_SECOND = 0.05
SCHEDULING_FRAME_SECONDS = 0.200  # 200ms
MAX_TARGET_RANGE_METERS = 200_000  # 200km
MAX_TIME_DESIRED_START = 1.0  # 1 second from now
assert MAX_TIME_DESIRED_START >= SCHEDULING_FRAME_SECONDS
MAX_TIME_DWELL = 1.0
MAX_COST_DELAY_UTILITY_PER_SECOND = -100.0
MAX_COST_DROP_UTILITY = -1000.0
COST_DELAY_SEARCH = 12.0
COST_DROP_SEARCH = 345.0
COST_DELAY_TRACKED = 67.0
COST_DROP_TRACKED = 890.0
SEARCH_PATTERN_PERIOD_SECONDS = 0.5
SEARCH_PATTERN_DWELL_SECONDS = 0.2
assert SEARCH_PATTERN_DWELL_SECONDS / SEARCH_PATTERN_PERIOD_SECONDS <= 1.0
TRACK_TASK_EXPIRE_AFTER_SECONDS = 1.0
SEARCH_TASK_EXPIRE_AFTER_SECONDS = 2.0


SEQUENCE_OF_TASKS_SPACE = gym.spaces.Sequence(
    gym.spaces.Dict(
        {
            "time_desired_start": gym.spaces.Box(
                0.0,
                MAX_TIME_DESIRED_START,
                shape=(1,),
                dtype=np.float32,
            ),
            "time_dwell": gym.spaces.Box(
                0.0,
                MAX_TIME_DWELL,
                shape=(1,),
                dtype=np.float32,
            ),
            "cost_delay_utility_per_second": gym.spaces.Box(
                0.0,
                MAX_COST_DELAY_UTILITY_PER_SECOND,
                shape=(1,),
                dtype=np.float32,
            ),
            "cost_drop_utility": gym.spaces.Box(
                0.0,
                MAX_COST_DROP_UTILITY,
                shape=(1,),
                dtype=np.float32,
            ),
        }
    )
)


class Simulation(gym.Env):
    def __init__(
        self,
        seed: int | None = None,
        # render_mode: str | None = None, # This is for making the pretty animations
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.clock = 0.0
        self.untracked_targets = []
        self.tracked_targets = []
        self.search_pattern = None
        self.queue = []

    @property
    def observation_space(self) -> gym.spaces.Space:
        return SEQUENCE_OF_TASKS_SPACE

    @property
    def action_space(self) -> gym.spaces.Space:
        return SEQUENCE_OF_TASKS_SPACE

    # def close(self) -> None:
    #     # This was teardown code for pygame
    #     return None

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> []:  # should return observation type
        if seed is None:
            seed = self.rng.integers(2**31)
        super().reset(seed=seed)
        if seed:
            # I'm not sure if this is necessary, I can't keep track of rngs in python
            self.rng = np.random.default_rng(seed)

        self.clock = 0.0

        self.search_pattern = Sequential_Search_Pattern()

        self.untracked_targets = [
            Target(self.rng) for _ in range(INITIAL_UNTRACKED_TARGETS)
        ]

        temp_tracked_targets = [
            Target(self.rng) for _ in range(INITIAL_UNTRACKED_TARGETS)
        ]

        self.trackers = [Tracker(self.rng, target) for target in temp_tracked_targets]

        self.queue = []
        self.queue += [
            tracker.look_requests(SCHEDULING_FRAME_SECONDS) for tracker in self.trackers
        ]
        self.queue += self.search_pattern.look_requests(SCHEDULING_FRAME_SECONDS)

        observation = [
            look_request
            for look_request in self.queue
            if look_request.time_desired_start <= self.clock + SCHEDULING_FRAME_SECONDS
        ]

        info = {}
        return observation, info

    def step(self, action):
        # gym boilerplate
        truncated = False
        terminated = False
        info = {}

        ## initialize cost (to be returned as 'reward')
        cost = 0.0

        ## advance time by some duration
        # You could do always 200ms in some fixed "duration",
        # but this lets you have a jagged edge if tasks don't fill the 200ms buffer.
        # I think fixing to 200ms would just cause bizarre idle time even if queue is full
        duration = sum([task["time_dwell"] for task in action])
        self.clock = self.clock + duration

        ## 'execute' tasks chosen by the agent

        for task_index, scheduled_time in action:
            # Okay, so this doesn't work, because the 'tasks' coming back in your action space
            # aren't actually the tasks from the queue, worse,
            # I think you intend for your scheduler to assign them some sort of 'start time'
            # I think probably a better format would be to give each task an index
            # and then instead of handing back entire tasks, just give 'index, start time' pairs back
            # anyway, it'll look something like:
            task = self.queue.pop(task_index)  # pop will remove it from the queue
            cost += task.execute_at(scheduled_time)
            for target in self.untracked_targets:
                if self.rng.random() > target.p_d():
                    self.trackers.append(Target(self.rng, target))
                    # TODO: somehow pop the spotted target from untracked_targets
                    # I'm not sure how to do this in python the way I've looped over this list
                    # I've put a really ugly hack below, you can do better
            self.untracked_targets = [
                untracked_target
                for untracked_target in self.untracked_targets
                if untracked_target not in [tracker.target for tracker in self.trackers]
            ]

        ## 'update' everything
        self.queue = [task.update(duration) for task in self.queue]
        self.untracked_targets = [
            target.update(duration) for target in self.untracked_targets
        ]
        self.trackers = [tracker.update(duration) for tracker in self.trackers]

        ## 'tend' the queue
        cost += sum([task.cost_drop_utility for task in self.queue if task.expired()])
        self.queue = [task for task in self.queue if not task.expired()]

        self.queue += [
            tracker.look_requests(SCHEDULING_FRAME_SECONDS) for tracker in self.trackers
        ]
        self.queue += self.search_pattern.look_requests(SCHEDULING_FRAME_SECONDS)

        # generate new targets
        for _ in range(self.rng.poisson(TARGET_ARRIVAL_LAMBDA_PER_SECOND * duration)):
            self.untracked_targets.append(Target(self.rng))

        # kill off tracked targets?
        for _ in range(self.rng.poisson(TARGET_ARRIVAL_LAMBDA_PER_SECOND * duration)):
            # Nukes a random tracker, worried about array out of bounds errors here
            del self.trackers[self.rng.integers(len(self.trackers))]

        observation = self.queue_to_observation()

        # # End with gym boilerplate to generate animations
        # if self.render_mode == "human":
        #     self.render()
        return observation, cost, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="human")',
            )
            return
        # If you want to make a beautiful animation, this is where you'd do it
        # That said, I don't think any of your tasks currently have azimuth
        # and the way that you want multiple 'frames' from one 'step' will be a nightmare
        return

    def queue_to_observation(self):
        # I don't know what a 'gym.spaces.sequence' is, I'm guessing just a list...
        # I think a list of dicts will interface with the bonkers 'gym.spaces' thing I defined above
        # I'm not sure though, maybe you need some special 'sequence' from gym.
        # You then (I think) can just use it as a normal list on the other side?
        # (after half a dozen casts and asserts)
        # python people will saw off their own arm before they'll define a type... apparently this is easier?
        # I'm sure I've done it wrong, but this should be close enough for you to figure out.
        # good luck. I'm too dumb python type anarchy.
        return [
            {
                "time_desired_start": task.time_desired_start,
                "time_dwell": task.time_dwell,
                "cost_delay_utility_per_second": task.cost_delay_utility_per_second,
                "cost_drop_utility": task.cost_drop_utility,
            }
            for task in self.queue
            # I think it's possible, but unnecessary, to filter for schedulable tasks
            # if task.time_desired_start <= self.clock + SCHEDULING_FRAME_SECONDS
        ]


class Target:
    def __init__(self, rng):
        self.azimuth = rng.random() * 2 * pi
        self.range = rng.random() * MAX_TARGET_RANGE_METERS
        # initialize target kinematics here, would look something like
        # self.singer_alpha = rng.uniform(MIN_SINGER_ALPHA,MAX_SINGER_ALPHA)
        # self.singer_beta = rng.normal(SINGER_BETA_MU,SINGER_BETA_SIGMA))
        # self.singer_gamma = rng.poisson(SINGER_GAMMA_LAM))

    def update(self, time_elapsed):
        # self.range = self.singer_alpha * 42 + 3 or whatever, do targets actually move?
        return

    def p_d(dwell_time):
        # SNR = POWER * dwell_time / (BW *  (self.range**4))
        # pd = 1 - e**-SNR
        # work out the odds that the beam has hit the target
        return 0.5


class Tracker:
    def __init__(self, rng, target):
        self.target = target
        self.seconds_since_last_task = rng.random()  # Prevent big-bang start

    def update(self, time_elapsed):
        self.seconds_since_last_task -= time_elapsed
        self.target.update(time_elapsed)  # yuck, but can't figure out where to put this

    def tasks(self, time):
        # revisit_rate = self.target.singer_beta * 42 or whatever that math on slide 12 was
        revisit_rate = 0.42
        # dwell_time = self.target.singer_gamma / 123 # N_0 math?
        dwell = 0.123

        tasks = []
        while self.seconds_since_last_task < time:
            self.seconds_since_last_task += revisit_rate
            tasks.append(
                Task(
                    time_desired_start=self.seconds_since_last_task,
                    time_dwell=dwell,
                    cost_delay_utility_per_second=COST_DELAY_TRACKED,  # or sample some priority or whatever
                    cost_drop_utility=COST_DROP_TRACKED,
                )
            )
        return tasks


class Task:
    def __init__(
        self,
        time_desired_start,
        time_dwell,
        cost_delay_utility_per_second,
        cost_drop_utility,
        expire_after_seconds,
    ):
        assert time_desired_start <= MAX_TIME_DESIRED_START
        assert time_dwell <= MAX_TIME_DWELL
        assert cost_delay_utility_per_second <= MAX_COST_DELAY_UTILITY_PER_SECOND
        assert cost_drop_utility <= MAX_COST_DROP_UTILITY
        assert expire_after_seconds >= 0
        self.time_desired_start = time_desired_start
        self.time_dwell = time_dwell
        self.cost_delay_utility_per_second = cost_delay_utility_per_second
        self.cost_drop_utility = cost_drop_utility
        self.expire_after_seconds = expire_after_seconds

    def update(self, time_elapsed):
        self.time_desired_start -= time_elapsed
        self.expire_after_seconds -= time_elapsed

    def expired(self) -> bool:
        return self.expire_after_seconds >= 0

    def execute_at(self, scheduled_time) -> float:
        if scheduled_time >= self.expire_after_seconds:
            # TODO: I think this logic is wrong, but I think you need something like this
            return self.drop_cost_utility
        delay = scheduled_time - self.time_desired_start  # I've lost track of signs
        return self.cost_delay_utility_per_second * delay


class Sequential_Search_Pattern:
    def __init__(self):
        self.seconds_since_last_task = 0.0
        # self.last_azimuth = 0.0 # You could keep track of azimuth but I don't think it matters

    def tasks(self, time):
        tasks = []
        while self.seconds_since_last_look_request < time:
            self.seconds_since_last_look_request += SEARCH_PATTERN_PERIOD_SECONDS
            tasks.append(
                Task(
                    time_desired_start=self.seconds_since_last_look_request,
                    time_dwell=SEARCH_PATTERN_DWELL_SECONDS,
                    cost_delay_utility_per_second=COST_DELAY_SEARCH,  # or sample some priority or whatever
                    cost_drop_utility=COST_DROP_SEARCH,
                )
            )
        return tasks
