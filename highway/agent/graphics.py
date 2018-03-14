from __future__ import division, print_function
import matplotlib as mpl
import matplotlib.cm as cm
import pygame

from highway.agent.mcts import MCTSAgent
from highway.agent.ttc_vi import TTCVIAgent
from highway.mdp.road_mdp import RoadMDP  # TODO: Replace with Generic MDP


class AgentGraphics(object):
    """
        Graphical visualization of any Agent implementing AbstractAgent.
    """
    @classmethod
    def display(cls, agent, surface):
        """
            Display an agent visualization on a pygame surface.

        :param agent: the agent to be displayed
        :param surface: the pygame surface on which the agent is displayed
        :return:
        """
        if isinstance(agent, TTCVIAgent):
            TTCVIGraphics.display(agent, surface)
        elif isinstance(agent, MCTSAgent):
            MCTSGraphics.display(agent, surface)
        else:
            raise NotImplementedError()


class MCTSGraphics(object):
    """
        Graphical visualization of the MCTSAgent tree.
    """
    RED = (255, 0, 0)
    BLACK = (0, 0, 0)

    @classmethod
    def display(cls, agent, surface):
        """
            Display the whole tree of an MCTSAgent.

        :param agent: the MCTSAgent to be displayed
        :param surface: the pygame surface on which the agent is displayed
        """
        cell_size = (surface.get_width() // agent.mcts.max_depth, surface.get_height())
        pygame.draw.rect(surface, cls.BLACK, (0, 0, surface.get_width(), surface.get_height()), 0)
        cls.display_node(agent.mcts.root, surface, (0, 0), cell_size, depth=0, selected=True)

        actions = agent.mcts.get_plan()
        font = pygame.font.Font(None, 15)
        text = font.render('-'.join(map(str, actions)), 1, (10, 10, 10), (255, 255, 255))
        surface.blit(text, (5, surface.get_height()-15))


    @classmethod
    def display_node(cls, node, surface, origin, size, depth=0,
                     selected=False,
                     display_exploration=False,
                     display_count=False,
                     display_text=True):
        """
            Display an MCTS node at a given position on a surface.

        :param node: the MCTS node to be displayed
        :param surface: the pygame surface on which the node is displayed
        :param origin: the location of the node on the surface [px]
        :param size: the size of the node on the surface [px]
        :param depth: the depth of the node in the tree
        :param selected: whether the node is within a selected branch of the tree
        :param display_exploration: display the exploration bonus
        :param display_count: display the visitation count of the node
        :param display_text: display a text showing the value and visitation count of the node
        """
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
            surface.blit(text, (origin[0]+2, origin[1]+2))

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
    """
        Graphical visualization of the TTCVIAgent value function.
    """
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)

    @classmethod
    def display(cls, agent, surface):
        """
            Display the computed value function of an agent.

        :param agent: the agent to be displayed
        :param surface: the surface on which the agent is displayed.
        """
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
