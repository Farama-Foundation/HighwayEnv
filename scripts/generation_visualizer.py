import math

import numpy as np
import pygame

from highway_env.road.generation.engine import (
    check_lanes_type_validity,
    combine_nodes,
    correct_junction_boundaries,
    generate_lane_boundaries,
    generate_road_network_skeleton,
    get_all_intersection_points,
    get_invalid_lanes,
    get_junction_pos,
    get_nodeset,
    get_radially_sorted_endpoints,
    kill_lanes,
    prune_intersecting_lanes,
    rectify_short_lanes,
    remove_disjoint_clusters,
    remove_identical_reference_lanes,
    seal_dead_end,
    split_lanes,
    twist_optimize,
)
from highway_env.road.generation.spatial_hash import lanes_spatial_hash


"""
Visualizes the step-by-step process of procedural road network generation for debugging.
"""


# Generation Parameters -------------------------------------
params = {
    "target_num_endpoints": 300,
    "forward_speed": 30,
    "age_of_maturity": 4,
    "lane_width": 20,
    "perlin_variation_params": {
        "jitteriness": {"upper": 0.1, "lower": 0.0},
        "max_turn_speed": {"upper": 4.0, "lower": 0.01},
        "replication_chance": {"upper": 0.7, "lower": 0.0},
        "spontaneous_death_chance": {"upper": 0.0, "lower": 0.0},
    },
    "disable_prints": False,
}

seed = 2

rng = np.random.default_rng(seed)

merge_radius = params["forward_speed"] * 2
prevent_replication_radius = params["age_of_maturity"] * params["forward_speed"]

twist_iterations = 2 * params["forward_speed"]
twist_step = 0.0002 / params["forward_speed"]

disable_prints = params["disable_prints"]


# Drawing -------------------------------------

GRID_COLOR = (35, 35, 35)
ORIGIN_COLOR = (80, 80, 80)
BG_COLOR = (10, 10, 10)
LABEL_COLOR = (150, 150, 150)

LANE_COLOR = (180, 180, 180)  # road polyline
LANE_COLORS = {"left_points": (180, 180, 180), "right_points": (180, 180, 180)}
NODE_COLOR = (255, 200, 60)  # node label text
NODE_BG_COLOR = (30, 30, 30)  # small background pill behind label
NODE_RADIUS_COLOR = (60, 120, 180)  # thin merge-radius circle
CENTER_POINTS_COLOR = (255, 0, 0)
JUNCTION_POINT_COLOR = (0, 255, 0)
MERGE_RADIUS = merge_radius  # must match generator value
DRAW_MERGE_RADII = False
DRAW_ENDPOINTS = True

PAN_SPEED = (
    5.0  # world units per frame at zoom=1 (scaled by 1/zoom so speed feels constant)
)
ZOOM_SPEED = 0.05
MIN_ZOOM = 0.05
MAX_ZOOM = 50.0


class Camera:
    def __init__(self, screen_w, screen_h):
        self.x = 0.0
        self.y = 0.0
        self.zoom = 1.0
        self.screen_w = screen_w
        self.screen_h = screen_h

    def world_to_screen(self, wx, wy):
        sx = (wx - self.x) * self.zoom + self.screen_w / 2
        sy = (wy - self.y) * self.zoom + self.screen_h / 2
        return (int(sx), int(sy))

    def world_to_screen_f(self, wx, wy):
        """Float version, useful for distance calculations."""
        sx = (wx - self.x) * self.zoom + self.screen_w / 2
        sy = (wy - self.y) * self.zoom + self.screen_h / 2
        return (sx, sy)

    def scale(self, world_length):
        """Convert a world-unit length to pixels."""
        return world_length * self.zoom


def draw_circle(surface, camera, color, wx, wy, world_radius, width=0):
    sx, sy = camera.world_to_screen(wx, wy)
    r = max(1, int(camera.scale(world_radius)))
    pygame.draw.circle(surface, color, (sx, sy), r, width)


def draw_rect(surface, camera, color, wx, wy, world_w, world_h, width=0):
    """Draw a rectangle centred at (wx, wy) in world space."""
    sx, sy = camera.world_to_screen(wx - world_w / 2, wy - world_h / 2)
    pw = max(1, int(camera.scale(world_w)))
    ph = max(1, int(camera.scale(world_h)))
    pygame.draw.rect(surface, color, pygame.Rect(sx, sy, pw, ph), width)


def draw_polyline(surface, camera, color, world_points, width=1):
    if len(world_points) < 2:
        return
    screen_points = [camera.world_to_screen(x, y) for x, y in world_points]
    pygame.draw.lines(
        surface, color, False, screen_points, max(1, int(width * camera.zoom))
    )


def draw_line(surface, camera, color, wx1, wy1, wx2, wy2, width=1):
    p1 = camera.world_to_screen(wx1, wy1)
    p2 = camera.world_to_screen(wx2, wy2)
    pygame.draw.line(surface, color, p1, p2, max(1, int(width)))


def draw_grid(surface, camera, spacing=50):
    """Draw a light world-space grid."""
    w, h = surface.get_size()

    # world bounds visible on screen
    left = camera.x - (w / 2) / camera.zoom
    right = camera.x + (w / 2) / camera.zoom
    top = camera.y - (h / 2) / camera.zoom
    bottom = camera.y + (h / 2) / camera.zoom

    # snap to grid spacing
    x = math.floor(left / spacing) * spacing
    while x <= right:
        sx, _ = camera.world_to_screen(x, 0)
        color = ORIGIN_COLOR if x == 0 else GRID_COLOR
        pygame.draw.line(surface, color, (sx, 0), (sx, h), 1 if x != 0 else 2)
        x += spacing

    y = math.floor(top / spacing) * spacing
    while y <= bottom:
        _, sy = camera.world_to_screen(0, y)
        color = ORIGIN_COLOR if y == 0 else GRID_COLOR
        pygame.draw.line(surface, color, (0, sy), (w, sy), 1 if y != 0 else 2)
        y += spacing


def draw_lanes(surface, camera, lanes, node_font, stage):
    """
    Draw all lanes as polylines, then overlay node ID labels.
    """

    # --- Pass 1: road polylanes ---
    sides = ["left_points", "right_points"]
    for lane in lanes:
        if stage <= 8:
            pts = lane.points
            screen_pts = [camera.world_to_screen(x, y) for x, y in pts]
            width = max(1, int(5 * camera.zoom))
            if len(screen_pts) >= 2:
                pygame.draw.lines(surface, LANE_COLOR, False, screen_pts, width)
        else:
            for side in sides:
                pts = getattr(lane, side)
                if len(pts) < 2:
                    continue
                screen_pts = [camera.world_to_screen(x, y) for x, y in pts]
                width = max(1, int(1 * camera.zoom))
                pygame.draw.lines(surface, LANE_COLORS[side], False, screen_pts, width)

    if DRAW_MERGE_RADII:
        # --- Pass 2: merge-radius circles at every node ---
        for lane in lanes:
            pts = lane.points
            if not pts:
                continue
            for wx, wy in (pts[0], pts[-1]):
                sx, sy = camera.world_to_screen(wx, wy)
                r = max(1, int(camera.scale(MERGE_RADIUS)))
                pygame.draw.circle(surface, NODE_RADIUS_COLOR, (sx, sy), r, 1)
        # """

    # --- Pass 2: node labels ---
    if DRAW_ENDPOINTS:
        node_labels = []
        for lane in lanes:
            pts = lane.points
            if not pts:
                continue
            # start node
            sx, sy = camera.world_to_screen(*pts[0])
            node_labels.append((lane.start, sx, sy))
            # end node
            sx, sy = camera.world_to_screen(*pts[-1])
            node_labels.append((lane.end, sx, sy))

        for node_id, sx, sy in node_labels:
            text_surf = node_font.render(node_id, True, NODE_COLOR)
            tw, th = text_surf.get_size()
            pad = 3
            bg_rect = pygame.Rect(
                sx - tw // 2 - pad,
                sy - th // 2 - pad,
                tw + pad * 2,
                th + pad * 2,
            )
            pygame.draw.rect(surface, NODE_BG_COLOR, bg_rect, border_radius=3)
            surface.blit(text_surf, (sx - tw // 2, sy - th // 2))


def draw_interest_points(lanes, points, junction_to_pos, surface, camera):
    r = 2
    for point in points:
        sx, sy = camera.world_to_screen(point[0], point[1])

        pygame.draw.circle(surface, (255, 0, 0), (sx, sy), r, 0)

    for lane in lanes:
        for point in lane.points:
            sx, sy = camera.world_to_screen(point[0], point[1])
            pygame.draw.circle(surface, CENTER_POINTS_COLOR, (sx, sy), r, 0)

    for point in junction_to_pos.values():
        sx, sy = camera.world_to_screen(point[0], point[1])
        pygame.draw.circle(surface, JUNCTION_POINT_COLOR, (sx, sy), r, 0)


def draw_hud(surface, font, camera):
    lines = [
        "WASD: pan",
        "Q/E: zoom in/out",
        "T: progress generator",
    ]
    y = 10
    for line in lines:
        img = font.render(line, True, LABEL_COLOR)
        surface.blit(img, (10, y))
        y += 18


# Main -------------------------------------
def main():
    pygame.init()
    SCREEN_W, SCREEN_H = 1280, 720
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), pygame.RESIZABLE)
    pygame.display.set_caption("Road network generation visualizer")

    hud_font = pygame.font.SysFont("monospace", 14)
    node_font = pygame.font.SysFont("monospace", 12)

    print("Generating road network...")
    lanes = generate_road_network_skeleton(
        target_num_endpoints=max(2, params["target_num_endpoints"]),
        forward_speed=params["forward_speed"],
        merge_radius=merge_radius,
        prevent_replication_radius=prevent_replication_radius,
        age_of_maturity=params["age_of_maturity"],
        perlin_variation_params=params["perlin_variation_params"],
        disable_prints=disable_prints,
        rng=rng,
    )
    print(f"Number of lanes: {len(lanes)}")

    camera = Camera(SCREEN_W, SCREEN_H)
    running = True
    stage = 1
    stage_names = {
        1: "Removing short lanes",
        2: "Splitting lanes",
        3: "Removing short lanes (again)",
        4: "Combining nodes",
        5: "Removing identical reference frames",
        6: "Removing crossing lanes",
        7: "Twist optimization",
        8: "Generate lane boundaries",
        9: "Correcting junction boundaries",
        10: "Labeling interest points",
        11: "Detecting and removing blocked lanes",
        12: "Removing disjoint clusters",
        13: "GENERATION FINISHED",
    }

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.VIDEORESIZE:
                camera.screen_w, camera.screen_h = event.w, event.h
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    print(f"Executing stage {stage}: {stage_names[stage]}")
                    match stage:
                        case 1:
                            rectify_short_lanes(lanes)
                        case 2:
                            conjoined_nodes = combine_nodes(
                                lanes,
                                merge_radius,
                                mark=True,
                                disable_prints=disable_prints,
                            )
                            split_lanes(
                                lanes,
                                conjoined_nodes,
                                merge_radius=merge_radius,
                                forward_speed=params["forward_speed"],
                                disable_prints=disable_prints,
                            )
                        case 3:
                            rectify_short_lanes(lanes)
                        case 4:
                            combine_nodes(
                                lanes,
                                merge_radius,
                                disable_prints=disable_prints,
                            )
                        case 5:
                            remove_identical_reference_lanes(lanes)
                        case 6:
                            prune_intersecting_lanes(
                                lanes, disable_prints=disable_prints
                            )
                        case 7:
                            twist_optimize(
                                lanes,
                                iterations=twist_iterations,
                                step=twist_step,
                                lane_width=params["lane_width"],
                                disable_prints=disable_prints,
                            )
                            nodeset = get_nodeset(lanes)
                        case 8:
                            generate_lane_boundaries(lanes, params["lane_width"])
                        case 9:
                            for node in get_nodeset(lanes):
                                correct_junction_boundaries(lanes, node)
                                seal_dead_end(lanes, node)
                        case 10:
                            lane_to_grid, grid_to_lanes = lanes_spatial_hash(
                                lanes, gridsize=params["lane_width"] * 2
                            )
                            intersecting_points = get_all_intersection_points(
                                lanes, lane_to_grid, grid_to_lanes
                            )

                            junction_to_pos = dict()
                            for node in nodeset:
                                junction_to_pos[node] = get_junction_pos(
                                    lanes,
                                    get_radially_sorted_endpoints(lanes, node),
                                )

                        case 11:
                            invalids = get_invalid_lanes(
                                lanes,
                                params["forward_speed"],
                                disable_prints=disable_prints,
                                rng=rng,
                            )
                            if not disable_prints:
                                print(f"Removing {len(invalids)} obstructed lanes")
                            kill_lanes(lanes, invalids)

                        case 12:
                            remove_disjoint_clusters(lanes)
                            assert check_lanes_type_validity(lanes)
                            junction_to_pos = dict()
                            for node in get_nodeset(lanes):
                                junction_to_pos[node] = get_junction_pos(
                                    lanes,
                                    get_radially_sorted_endpoints(lanes, node),
                                )
                        case _:
                            continue
                    print(f"Finished executing stage {stage}.")
                    stage += 1

        # --- Camera movement ---
        keys = pygame.key.get_pressed()
        pan = PAN_SPEED / camera.zoom

        if keys[pygame.K_w]:
            camera.y -= pan
        if keys[pygame.K_s]:
            camera.y += pan
        if keys[pygame.K_a]:
            camera.x -= pan
        if keys[pygame.K_d]:
            camera.x += pan

        if keys[pygame.K_q]:
            camera.zoom = min(MAX_ZOOM, camera.zoom * (1 + ZOOM_SPEED))
        if keys[pygame.K_e]:
            camera.zoom = max(MIN_ZOOM, camera.zoom * (1 - ZOOM_SPEED))

        # --- Rendering ---
        screen.fill(BG_COLOR)
        draw_grid(screen, camera, spacing=50)
        draw_lanes(screen, camera, lanes, node_font, stage)
        draw_hud(screen, hud_font, camera)
        if stage >= 11:
            draw_interest_points(
                lanes, intersecting_points, junction_to_pos, screen, camera
            )
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
