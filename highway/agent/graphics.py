import matplotlib as mpl
import matplotlib.cm as cm
import pygame

from highway.agent.mcts import MCTSAgent
from highway.agent.ttc_vi import TTCVIAgent
from highway.mdp.road_mdp import RoadMDP  # TODO: Replace with Generic MDP


class MCTSGraphics(object):
    RED = (255, 0, 0)
    BLACK = (0, 0, 0)

    @classmethod
    def display(cls, agent, surface):
        cell_size = (surface.get_width() // agent.mcts.max_depth, surface.get_height())
        pygame.draw.rect(surface, cls.BLACK, (0, 0, surface.get_width(), surface.get_height()), 0)
        cls.display_node(agent.mcts.root, surface, (0, 0), cell_size, depth=0, selected=True)

    @classmethod
    def display_node(cls, node, surface, origin, size, depth=0,
                     selected=False,
                     display_exploration=False,
                     display_count=False,
                     display_text=True):
        # Display node value
        cmap = cm.jet_r
        norm = mpl.colors.Normalize(vmin=-15, vmax=30)
        color = cmap(norm(node.value), bytes=True)
        pygame.draw.rect(surface, color, (origin[0], origin[1], size[0], size[1]), 0)

        # Add exploration bonus
        if display_exploration:
            norm = mpl.colors.Normalize(vmin=-15, vmax=30)
            color = cmap(norm(node.selection_strategy()), bytes=True)
            pygame.draw.polygon(surface, color, [(origin[0], origin[1] + size[1]),
                                                 (origin[0] + size[0], origin[1]),
                                                 (origin[0] + size[0], origin[1] + size[1])], 0)

        # Add node count
        if display_count and depth < 3:
            norm = mpl.colors.Normalize(vmin=0, vmax=100)
            color = cmap(norm(node.count), bytes=True)
            pygame.draw.rect(surface, color, (origin[0], origin[1], 5, 5), 0)

        # Add selection display
        if selected:
            pygame.draw.rect(surface, cls.RED, (origin[0], origin[1], size[0], size[1]), 1)

        if display_text and depth < 3:
            font = pygame.font.Font(None, 15)
            text = font.render("{:.1f} / {:.1f} / {}".format(node.value, node.selection_strategy(), node.count),
                               1, (10, 10, 10), (255, 255, 255))
            text_pos = text.get_rect(centerx=origin[0]+20, centery=origin[1]+5)
            surface.blit(text, text_pos)

        # Recursively display children nodes
        best_action = node.select_action(temperature=0)
        for a in RoadMDP.ACTIONS:
            if a in node.children:
                action_selected = (selected and (a == best_action))
                cls.display_node(node.children[a], surface,
                                 (origin[0]+size[0], origin[1]+a*size[1]/len(RoadMDP.ACTIONS)),
                                 (size[0], size[1]/len(RoadMDP.ACTIONS)),
                                 depth=depth+1, selected=action_selected)


class TTCVIGraphics(object):
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)

    @classmethod
    def display(cls, agent, surface):
        norm = mpl.colors.Normalize(vmin=-15, vmax=30)
        cmap = cm.jet_r
        cell_size = (surface.get_width() // agent.T, surface.get_height() // (agent.L * agent.V))
        velocity_size = surface.get_height() // agent.V

        for h in range(agent.V):
            for i in range(agent.L):
                for j in range(agent.T):
                    color = cmap(norm(agent.value[h, i, j]), bytes=True)
                    pygame.draw.rect(surface, color, (
                        j * cell_size[0], i * cell_size[1] + h * velocity_size, cell_size[0], cell_size[1]), 0)
            pygame.draw.line(surface, cls.BLACK, (0, h * velocity_size), (agent.T * cell_size[0], h * velocity_size), 1)
        path, actions = agent.pick_trajectory()
        for (h, i, j) in path:
            pygame.draw.rect(surface, cls.RED,
                             (j * cell_size[0], i * cell_size[1] + h * velocity_size, cell_size[0], cell_size[1]), 1)


class AgentGraphics(object):
    @classmethod
    def display(cls, agent, surface):
        if isinstance(agent, TTCVIAgent):
            TTCVIGraphics.display(agent, surface)
        elif isinstance(agent, MCTSAgent):
            MCTSGraphics.display(agent, surface)
