"""
Side-by-side pygame visualization of merge-v0 vs merge-v1.

Shows how connected-lane neighbour detection differs when the ego vehicle
is on one lane segment and the lead vehicle is on the next connected segment.

Run:
    python scripts/compare_merge_v0_v1.py
    python scripts/compare_merge_v0_v1.py --no-patch
    python scripts/compare_merge_v0_v1.py --validate
"""

import argparse
import math
import types
from contextvars import ContextVar
from functools import partial

import gymnasium as gym
import numpy as np
import pygame

from highway_env.road.graphics import WorldSurface
from highway_env.road.road import Road
from highway_env.vehicle import kinematics
from highway_env.vehicle.objects import Landmark, LaneIndex


STARTING_SEED = 42
PANEL_SIZE = 520
HEADER_HEIGHT = 96
FOOTER_HEIGHT = 96
GAP = 12
ENV_CONFIG = {
    "duration": 300,
    "other_vehicles": 1,
    "real_time_rendering": False,
    "offscreen_rendering": True,
    "screen_width": PANEL_SIZE,
    "screen_height": PANEL_SIZE,
    "simulation_frequency": 15,
    "policy_frequency": 5,
}
BOUNDARY_COLOR = (255, 210, 60)
BOUNDARY_LABEL_COLOR = (255, 245, 180)
BOUNDARY_LABEL_BG = (40, 40, 40)
MISSED_FRONT_COLOR = (255, 60, 60)
MISSED_REAR_COLOR = (255, 60, 60)

_should_update_seed: ContextVar[bool] = ContextVar("update_seed")


class DualEnvReplay:
    """Keep both merge envs in sync and support stepping, rewinding, and looping."""

    def __init__(
        self,
        env_v0,
        env_v1,
        seed: int,
        neutral_action: np.ndarray,
        patch_left: bool = False,
        patch_right: bool = False,
    ) -> None:
        self.env_v0 = env_v0
        self.env_v1 = env_v1
        self.seed = seed
        self.neutral_action = neutral_action
        self.patch_left = patch_left
        self.patch_right = patch_right
        self.cursor = 0
        self.actions: list[np.ndarray] = []
        self.loops_completed = 0
        self._reset_episode()

    def _maybe_patch(self, env, enabled: bool) -> None:
        if enabled:
            patch_original_neighbour_vehicles(env.unwrapped.road)

    def _reset_episode(self) -> None:
        self.seed += _should_update_seed.get()
        self.env_v0.reset(seed=self.seed)
        self.env_v1.reset(seed=self.seed)
        self._maybe_patch(self.env_v0, self.patch_left)
        self._maybe_patch(self.env_v1, self.patch_right)
        self.cursor = 0
        self.actions = []

    def complete_loop_and_reset(self) -> None:
        self.loops_completed += 1
        self._reset_episode()

    def _replay_to(self, target: int) -> None:
        self.env_v0.reset(seed=self.seed)
        self.env_v1.reset(seed=self.seed)
        self._maybe_patch(self.env_v0, self.patch_left)
        self._maybe_patch(self.env_v1, self.patch_right)
        for index in range(target):
            self.env_v0.step(self.actions[index])
            self.env_v1.step(self.actions[index])
        self.cursor = target

    def step_forward(self) -> None:
        action = self.neutral_action.copy()
        self.env_v0.step(action)
        self.env_v1.step(action)
        if self.cursor == len(self.actions):
            self.actions.append(action)
        else:
            self.actions = self.actions[: self.cursor] + [action]
        self.cursor += 1

    def rewind(self) -> bool:
        if self.cursor <= 0:
            return False
        self._replay_to(self.cursor - 1)
        return True

    def unwind(self) -> bool:
        if self.cursor < len(self.actions):
            self._replay_to(self.cursor + 1)
            return True
        return False

    @property
    def at_live_edge(self) -> bool:
        return self.cursor == len(self.actions)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare merge-v0 and merge-v1 neighbour detection."
    )
    parser.add_argument(
        "-f",
        "--fixed-seed",
        action="store_true",
        help="Disable seed update after each loop.",
    )
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=80,
        help="Simulation steps per loop before resetting.",
    )
    parser.add_argument(
        "-p",
        "--patch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Patch left merge-v0 with the pre-PR-667 neighbour_vehicles implementation "
            "(default: enabled). Ignored when --validate is set."
        ),
    )
    parser.add_argument(
        "-v",
        "--validate",
        action="store_true",
        help="Compare registered merge-v0 (left) against patched merge-v0 (right).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Rendering framerate.",
    )
    args = parser.parse_args()

    if args.validate:
        right_env_id = "merge-v0"
        patch_left = False
        patch_right = True
        window_caption = "Merge v0 vs patched v0 — validation"
        left_title = "merge-v0 (registered)"
        right_title = "merge-v0 (patched original neighbour_vehicles)"
        mode_label = "validate"
    else:
        right_env_id = "merge-v1"
        patch_left = args.patch
        patch_right = False
        window_caption = "Merge v0 vs v1 — neighbour_vehicles comparison"
        left_title = (
            "merge-v0 (original neighbour_vehicles)" if args.patch else "merge-v0"
        )
        right_title = "merge-v1 (connected-lane search)"
        mode_label = f"v0 patch {'on' if args.patch else 'off'}"

    pygame.init()
    frame_rate = args.fps
    window_width = PANEL_SIZE * 2 + GAP
    window_height = PANEL_SIZE + HEADER_HEIGHT + FOOTER_HEIGHT
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption(window_caption)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("menlo,consolas,dejavusansmono,monospace", 18)
    small_font = pygame.font.SysFont("menlo,consolas,dejavusansmono,monospace", 13)
    boundary_font = pygame.font.SysFont("menlo,consolas,dejavusansmono,monospace", 11)

    _make_env = partial(gym.make, render_mode="rgb_array", config=ENV_CONFIG)
    env_v0 = _make_env("merge-v0")
    env_v1 = _make_env(right_env_id)
    neutral_action = np.zeros(env_v0.action_space.shape, dtype=np.float32)
    _should_update_seed.set(not args.fixed_seed)
    replay = DualEnvReplay(
        env_v0,
        env_v1,
        STARTING_SEED - _should_update_seed.get(),
        neutral_action,
        patch_left,
        patch_right,
    )

    auto_play = True
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_SPACE:
                    auto_play = not auto_play
                elif event.key == pygame.K_LEFT:
                    if replay.rewind():
                        auto_play = False
                elif event.key == pygame.K_RIGHT:
                    if replay.unwind():
                        pass
                    elif replay.at_live_edge:
                        replay.step_forward()
                        if replay.cursor >= args.steps:
                            replay.complete_loop_and_reset()
                        auto_play = False

        if auto_play and replay.at_live_edge:
            replay.step_forward()
            if replay.cursor >= args.steps:
                replay.complete_loop_and_reset()

        screen.fill((20, 20, 20))

        panel_v0, status_v0 = _render_panel(env_v0, boundary_font)
        panel_v1, status_v1 = _render_panel(env_v1, boundary_font)

        _draw_header(
            screen,
            font,
            small_font,
            left_title,
            status_v0,
            0,
            PANEL_SIZE,
        )
        _draw_header(
            screen,
            font,
            small_font,
            right_title,
            status_v1,
            PANEL_SIZE + GAP,
            PANEL_SIZE,
        )

        screen.blit(panel_v0, (0, HEADER_HEIGHT))
        screen.blit(panel_v1, (PANEL_SIZE + GAP, HEADER_HEIGHT))

        footer_y = HEADER_HEIGHT + PANEL_SIZE
        _draw_footer(
            screen,
            small_font,
            footer_y,
            loop_text=(
                f"step {replay.cursor}/{args.steps} | "
                f"loop {replay.loops_completed + 1} | "
                f"{mode_label} | "
                f"{'live' if replay.at_live_edge else 'rewound'} | "
                f"{'playing' if auto_play and replay.at_live_edge else 'paused'} | "
                f"seed {replay.seed}"
            ),
        )

        pygame.display.flip()
        clock.tick(frame_rate)

    env_v0.close()
    env_v1.close()
    pygame.quit()


def original_neighbour_vehicles(
    self,
    vehicle: kinematics.Vehicle,
    lane_index: LaneIndex = None,
) -> tuple[kinematics.Vehicle | None, kinematics.Vehicle | None]:
    """
    Find the preceding and following vehicles of a given vehicle.

    :param vehicle: the vehicle whose neighbours must be found
    :param lane_index: the lane on which to look for preceding and following vehicles.
                    It doesn't have to be the current vehicle lane but can also be another lane, in which case the
                    vehicle is projected on it considering its local coordinates in the lane.
    :return: its preceding vehicle, its following vehicle
    """
    lane_index = lane_index or vehicle.lane_index
    if not lane_index:
        return None, None
    lane = self.network.get_lane(lane_index)
    s = self.network.get_lane(lane_index).local_coordinates(vehicle.position)[0]
    s_front = s_rear = None
    v_front = v_rear = None
    for v in self.vehicles + self.objects:
        if v is not vehicle and not isinstance(
            v, Landmark
        ):  # self.network.is_connected_road(v.lane_index,
            # lane_index, same_lane=True):
            s_v, lat_v = lane.local_coordinates(v.position)
            if not lane.on_lane(v.position, s_v, lat_v, margin=1):
                continue
            if s <= s_v and (s_front is None or s_v <= s_front):
                s_front = s_v
                v_front = v
            if s_v < s and (s_rear is None or s_v > s_rear):
                s_rear = s_v
                v_rear = v
    return v_front, v_rear


def patch_original_neighbour_vehicles(road: Road) -> None:
    """Replace Road.neighbour_vehicles with the pre-PR-667 implementation."""
    road.neighbour_vehicles = types.MethodType(original_neighbour_vehicles, road)
    setattr(road, "_uses_original_neighbour_vehicles", True)


def _draw_dashed_line(
    surface: WorldSurface,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    width: int = 2,
    dash_length: int = 8,
    gap_length: int = 5,
) -> None:
    """Draw a dashed line between two pixel coordinates."""
    (x0, y0), (x1, y1) = start, end
    dx, dy = x1 - x0, y1 - y0
    distance = max(math.sqrt(dx * dx + dy * dy), 1.0)
    direction = (dx / distance, dy / distance)
    position = 0.0
    draw = True
    while position < distance:
        segment = dash_length if draw else gap_length
        next_position = min(position + segment, distance)
        if draw:
            p0 = (
                int(x0 + direction[0] * position),
                int(y0 + direction[1] * position),
            )
            p1 = (
                int(x0 + direction[0] * next_position),
                int(y0 + direction[1] * next_position),
            )
            pygame.draw.line(surface, color, p0, p1, width)
        position = next_position
        draw = not draw


def _draw_segment_boundaries(
    surface: WorldSurface,
    road: Road,
    font: pygame.font.Font | None = None,
) -> None:
    """Draw perpendicular dashed lines at lane segment ends (road-network nodes)."""
    labeled_nodes: set[str] = set()

    for _from, to_dict in road.network.graph.items():
        for _to, lanes in to_dict.items():
            for lane in lanes:
                longitudinal = lane.length
                half_width = lane.width_at(longitudinal) / 2
                margin = 0.4
                left = lane.position(longitudinal, -(half_width + margin))
                right = lane.position(longitudinal, half_width + margin)
                start = surface.vec2pix(left)
                end = surface.vec2pix(right)
                _draw_dashed_line(surface, start, end, BOUNDARY_COLOR, width=3)

                if font is not None and _to not in labeled_nodes:
                    center = lane.position(longitudinal, 0)
                    center_pix = surface.vec2pix(center)
                    label = font.render(f"node {_to}", True, BOUNDARY_LABEL_COLOR)
                    label_rect = label.get_rect(
                        center=(center_pix[0], center_pix[1] - 14)
                    )
                    bg_rect = label_rect.inflate(6, 4)
                    pygame.draw.rect(surface, BOUNDARY_LABEL_BG, bg_rect)
                    surface.blit(label, label_rect)
                    labeled_nodes.add(_to)


def _vehicle_label(vehicle) -> str:
    if vehicle is None:
        return "None"
    lane = getattr(vehicle, "lane_index", None)
    if lane:
        return f"{lane[0]}->{lane[1]} lane {lane[2]}"
    return type(vehicle).__name__


def _has_same_segment_vehicle(road, ego, direction: str) -> bool:
    """Whether any vehicle sits ahead/behind ego on the current lane segment."""
    if ego is None or not ego.lane_index:
        return False

    lane = road.network.get_lane(ego.lane_index)
    ego_s = lane.local_coordinates(ego.position)[0]
    for vehicle in road.vehicles:
        if vehicle is ego:
            continue
        s_v, lat_v = lane.local_coordinates(vehicle.position)
        if not lane.on_lane(vehicle.position, s_v, lat_v, margin=1):
            continue
        if direction == "front" and s_v > ego_s:
            return True
        if direction == "rear" and s_v < ego_s:
            return True
    return False


def _reference_neighbours(road, ego):
    """Front/rear as returned by the library connected-lane implementation."""
    enabled = road.neighbour_vehicles_connected_lanes
    road.neighbour_vehicles_connected_lanes = True
    try:
        return Road.neighbour_vehicles(road, ego)
    finally:
        road.neighbour_vehicles_connected_lanes = enabled


def _find_connected_lane_vehicle(road, ego, direction: str = "front"):
    """Return the closest vehicle on a directly connected next/previous lane segment."""
    if ego is None or not ego.lane_index:
        return None

    _from, _to, _id = ego.lane_index
    ego_lane = road.network.get_lane(ego.lane_index)
    ego_s = ego_lane.local_coordinates(ego.position)[0]
    best = None
    best_distance = float("inf")

    if direction == "front":
        search_lanes = []
        for next_lane_list in road.network.graph.get(_to, {}).values():
            if _id < len(next_lane_list):
                search_lanes.append((next_lane_list[_id], ego_lane.length))
            elif next_lane_list:
                search_lanes.append((next_lane_list[0], ego_lane.length))

        for search_lane, offset in search_lanes:
            for vehicle in road.vehicles:
                if vehicle is ego:
                    continue
                s_v, lat_v = search_lane.local_coordinates(vehicle.position)
                if not search_lane.on_lane(vehicle.position, s_v, lat_v, margin=1):
                    continue
                distance = (offset + s_v) - ego_s
                if 0 < distance < best_distance:
                    best_distance = distance
                    best = vehicle
    else:
        search_lanes = []
        for to_dict in road.network.graph.values():
            if _from not in to_dict:
                continue
            prev_lanes = to_dict[_from]
            if _id < len(prev_lanes):
                search_lanes.append(prev_lanes[_id])
            elif prev_lanes:
                search_lanes.append(prev_lanes[0])

        for search_lane in search_lanes:
            offset = -search_lane.length
            for vehicle in road.vehicles:
                if vehicle is ego:
                    continue
                s_v, lat_v = search_lane.local_coordinates(vehicle.position)
                if not search_lane.on_lane(vehicle.position, s_v, lat_v, margin=1):
                    continue
                distance = ego_s - (s_v + offset)
                if 0 < distance < best_distance:
                    best_distance = distance
                    best = vehicle

    return best


def _missed_connected_neighbour(road, ego, detected, reference, direction: str) -> bool:
    """True when connected-lane search finds a neighbour that was not detected."""
    if reference is None or reference is detected:
        return False
    if ego is None:
        return False
    if detected is None and _has_same_segment_vehicle(road, ego, direction):
        return False
    return True


def _draw_missed_neighbour_cue(
    surface: WorldSurface,
    ego,
    vehicle,
    color: tuple[int, int, int],
) -> None:
    """Draw a dashed line and ring highlighting an undetected connected-lane neighbour."""
    ego_pix = surface.vec2pix(ego.position)
    target_pix = surface.vec2pix(vehicle.position)
    _draw_dashed_line(
        surface, ego_pix, target_pix, color, width=2, dash_length=12, gap_length=6
    )
    pygame.draw.circle(surface, color, target_pix, 10, 3)


def _draw_neighbour_overlay(surface: WorldSurface, road, ego) -> dict:
    """Draw neighbour-detection lines and return status for the HUD."""
    front, rear = road.neighbour_vehicles(ego)
    ref_front, ref_rear = _reference_neighbours(road, ego)
    connected_front = _find_connected_lane_vehicle(road, ego, "front")
    connected_rear = _find_connected_lane_vehicle(road, ego, "rear")
    missed_front = _missed_connected_neighbour(road, ego, front, ref_front, "front")
    missed_rear = _missed_connected_neighbour(road, ego, rear, ref_rear, "rear")
    show_missed_cues = not road.neighbour_vehicles_connected_lanes

    if ego is not None:
        ego_pix = surface.vec2pix(ego.position)

        if front is not None:
            front_pix = surface.vec2pix(front.position)
            pygame.draw.line(surface, (0, 220, 80), ego_pix, front_pix, 2)

        if rear is not None:
            rear_pix = surface.vec2pix(rear.position)
            pygame.draw.line(surface, (80, 160, 255), ego_pix, rear_pix, 2)

        if show_missed_cues and missed_front and ref_front is not None:
            _draw_missed_neighbour_cue(surface, ego, ref_front, MISSED_FRONT_COLOR)

        if show_missed_cues and missed_rear and ref_rear is not None:
            _draw_missed_neighbour_cue(surface, ego, ref_rear, MISSED_REAR_COLOR)

    return {
        "connected_lanes": road.neighbour_vehicles_connected_lanes,
        "uses_original_patch": getattr(
            road, "_uses_original_neighbour_vehicles", False
        ),
        "front": front,
        "rear": rear,
        "connected_front": connected_front,
        "connected_rear": connected_rear,
        "ref_front": ref_front,
        "ref_rear": ref_rear,
        "missed_connected_front": missed_front,
        "missed_connected_rear": missed_rear,
    }


def _render_panel(env, boundary_font: pygame.font.Font | None = None) -> pygame.Surface:
    env.render()
    viewer = env.unwrapped.viewer
    surface = viewer.sim_surface
    _draw_segment_boundaries(surface, env.unwrapped.road, boundary_font)
    status = _draw_neighbour_overlay(surface, env.unwrapped.road, env.unwrapped.vehicle)
    frame = viewer.get_image()
    panel = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    return panel, status


def _draw_legend_swatches(
    screen: pygame.Surface,
    x: int,
    y: int,
    items: list[tuple[tuple[int, int, int], bool, str]],
    font: pygame.font.Font,
    muted: tuple[int, int, int],
) -> int:
    """Draw a row of colour swatches; return x after the last item."""
    for color, dashed, label in items:
        if dashed and color in (MISSED_FRONT_COLOR, MISSED_REAR_COLOR):
            for offset in range(0, 18, 4):
                pygame.draw.line(
                    screen,
                    color,
                    (x + offset, y + 5),
                    (x + offset + 2, y + 5),
                    2,
                )
        elif dashed:
            _draw_dashed_line(screen, (x, y + 5), (x + 18, y + 5), color, 2)
        else:
            pygame.draw.line(screen, color, (x, y + 5), (x + 18, y + 5), 2)
        text = font.render(label, True, muted)
        screen.blit(text, (x + 24, y))
        x += 24 + text.get_width() + 20
    return x


def _draw_header(
    screen: pygame.Surface,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    title: str,
    status: dict,
    x: int,
    width: int,
) -> None:
    connected = "ON" if status["connected_lanes"] else "OFF"
    front_text = _vehicle_label(status["front"])
    rear_text = _vehicle_label(status["rear"])
    connected_front_text = _vehicle_label(status["connected_front"])
    connected_rear_text = _vehicle_label(status["connected_rear"])

    header_rect = pygame.Rect(x, 0, width, HEADER_HEIGHT)
    pygame.draw.rect(screen, (30, 30, 30), header_rect)

    screen.blit(font.render(title, True, (255, 255, 255)), (x + 12, 8))

    detail_color = (210, 210, 210)
    search_line = (
        "neighbour search: original implementation"
        if status["uses_original_patch"]
        else f"connected-lane search: {connected}"
    )
    row_y = 30
    screen.blit(small_font.render(search_line, True, detail_color), (x + 12, row_y))
    row_y += 16
    screen.blit(
        small_font.render(
            f"detected  front: {front_text}   rear: {rear_text}", True, detail_color
        ),
        (x + 12, row_y),
    )
    row_y += 16
    screen.blit(
        small_font.render(
            f"next seg front: {connected_front_text}   prev seg rear: {connected_rear_text}",
            True,
            detail_color,
        ),
        (x + 12, row_y),
    )
    row_y += 16
    if not status["connected_lanes"] and status["missed_connected_front"]:
        screen.blit(
            small_font.render(
                "missed front on connected segment", True, MISSED_FRONT_COLOR
            ),
            (x + 12, row_y),
        )
        row_y += 14
    if not status["connected_lanes"] and status["missed_connected_rear"]:
        screen.blit(
            small_font.render(
                "missed rear on connected segment", True, MISSED_REAR_COLOR
            ),
            (x + 12, row_y),
        )


def _draw_footer(
    screen: pygame.Surface,
    font: pygame.font.Font,
    y: int,
    loop_text: str,
) -> None:
    """Draw playback status, colour legend, and controls at the bottom."""
    muted = (180, 180, 180)
    screen.blit(font.render(loop_text, True, muted), (12, y + 4))

    legend_rows = [
        [
            ((0, 220, 80), False, "detected front"),
            (MISSED_FRONT_COLOR, True, "missed front (connected)"),
        ],
        [
            ((80, 160, 255), False, "detected rear "),
            (MISSED_REAR_COLOR, True, "missed rear  (connected)"),
        ],
        [
            (BOUNDARY_COLOR, True, "segment boundary"),
        ],
    ]

    legend_y = y + 20
    for row in legend_rows:
        _draw_legend_swatches(screen, 12, legend_y, row, font, muted)
        legend_y += 16

    screen.blit(
        font.render(
            "← rewind | → unwind / step | Space pause/play | Q quit",
            True,
            muted,
        ),
        (12, legend_y + 6),
    )


if __name__ == "__main__":
    main()
