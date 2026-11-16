import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from src.simulation import microsim

TZ = microsim.TZ


def test_segments_are_chained_head_to_tail_whatever_their_direction():
    a = np.array([[10.0, 0.0], [0.0, 0.0]])
    b = np.array([[20.0, 0.0], [10.0, 0.0]])
    c = np.array([[20.0, 0.0], [20.0, 5.0]])
    line = microsim.chain([a, b, c])
    assert line.tolist() == [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [20.0, 5.0]]


def test_points_cuts_and_projections_along_a_polyline():
    line = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0]])
    point, direction = microsim.point_at(line, 150.0)
    assert point.tolist() == [100.0, 50.0]
    assert direction.tolist() == [0.0, 1.0]
    assert microsim.project(line, np.array([90.0, 3.0])) == pytest.approx(90.0)
    window = microsim.cut(line, 50.0, 120.0)
    assert window[0].tolist() == [50.0, 0.0]
    assert window[-1].tolist() == [100.0, 20.0]
    assert microsim.distances(window)[-1] == pytest.approx(70.0)


def test_waypoints_are_evenly_spaced_and_stop_short_of_the_end():
    line = np.array([[0.0, 0.0], [100.0, 0.0]])
    points = microsim.waypoints(line, 30.0)
    assert [p[0] for p in points] == [30.0, 60.0]


def test_a_crossing_is_counted_only_going_forward():
    assert microsim.crossed(-0.4, 0.2)
    assert not microsim.crossed(0.2, -0.4)
    assert not microsim.crossed(0.5, 1.0)


def test_blocks_and_summary_compare_each_source_with_the_reference():
    start = pd.Timestamp("2026-10-27 23:00", tz=TZ)
    times = pd.Series([start, start, start + pd.Timedelta(minutes=6)])
    blocks = microsim.to_blocks(times, 5)
    assert blocks.tolist() == [2.0, 1.0]
    index = [start, start + pd.Timedelta(minutes=5)]
    observed = pd.Series([4.0, 2.0], index=index)
    result = microsim.summarise(
        {"observed": observed, "jupedsim": blocks}, "observed"
    ).set_index("source")
    assert result.loc["observed", "total"] == 6
    assert result.loc["jupedsim", "peak_block"] == start
    assert result.loc["jupedsim", "correlation"] == pytest.approx(1.0)


class FakeJuPedSim:
    """Agents walk straight from stage to stage at their speed, with no crowding."""

    class CollisionFreeSpeedModel:
        pass

    class CollisionFreeSpeedModelAgentParameters:
        def __init__(self, journey_id, stage_id, position, desired_speed, radius):
            self.journey_id, self.position = journey_id, position
            self.speed = desired_speed

    class Transition:
        @staticmethod
        def create_fixed_transition(stage):
            return stage

    class JourneyDescription:
        def __init__(self, stages):
            self.stages = list(stages)

        def set_transition_for_stage(self, here, there):
            pass

    class Agent:
        def __init__(self, agent_id, position, targets, speed):
            self.id, self.position = agent_id, tuple(position)
            self.targets, self.speed = targets, speed

    class Simulation:
        def __init__(self, model, geometry, dt):
            self.dt, self.time = dt, 0.0
            self.stages, self.journeys, self.walking = {}, {}, {}
            self.next_id = 0

        def add_waypoint_stage(self, point, reach):
            self.stages[len(self.stages)] = (np.asarray(point, float), reach)
            return len(self.stages) - 1

        def add_exit_stage(self, area):
            centre = area.centroid
            self.stages[len(self.stages)] = (np.array([centre.x, centre.y]), 1.0)
            return len(self.stages) - 1

        def add_journey(self, journey):
            self.journeys[len(self.journeys)] = journey.stages
            return len(self.journeys) - 1

        def add_agent(self, params):
            targets = [self.stages[s] for s in self.journeys[params.journey_id]]
            self.next_id += 1
            self.walking[self.next_id] = FakeJuPedSim.Agent(
                self.next_id, params.position, targets, params.speed
            )

        def iterate(self, count=1):
            for _ in range(count):
                self.time += self.dt
                for key in list(self.walking):
                    agent = self.walking[key]
                    here = np.asarray(agent.position)
                    target, reach = agent.targets[0]
                    if np.linalg.norm(target - here) <= reach:
                        agent.targets.pop(0)
                        if not agent.targets:
                            del self.walking[key]
                            continue
                        target, reach = agent.targets[0]
                    gap = target - here
                    size = np.linalg.norm(gap)
                    step = min(size, agent.speed * self.dt)
                    agent.position = tuple(here + gap / size * step if size else here)

        def agents(self):
            return list(self.walking.values())

        def agent_count(self):
            return len(self.walking)

        def elapsed_time(self):
            return self.time


def test_every_agent_crosses_the_line_once_at_walking_speed():
    window = np.array([[50.0, 0.0], [250.0, 0.0]])
    entries = pd.DataFrame({"route": [0] * 10, "entry_s": np.arange(10) * 4.0})
    cfg = {
        "dt": 0.05,
        "check_s": 0.5,
        "max_minutes": 30,
        "seed": 1,
        "speed_mps": 1.3,
        "half_width_m": 3.0,
        "entry_inset_m": 0.0,
        "lateral_offsets_m": [0.0],
        "waypoint_spacing_m": 15.0,
        "waypoint_reach_m": 2.0,
        "exit_radius_m": 2.0,
        "agent_radius_m": 0.2,
    }
    seconds, most = microsim.simulate(
        FakeJuPedSim,
        box(40, -3, 260, 3),
        {0: window},
        entries,
        np.array([150.0, 0.0]),
        np.array([1.0, 0.0]),
        cfg,
    )
    assert len(seconds) == 10
    expected = entries["entry_s"] + 100 / 1.3
    assert np.sort(seconds) == pytest.approx(expected.to_numpy(), abs=1.0)
    assert most >= 1
