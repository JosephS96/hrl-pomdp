from gym_minigrid.minigrid import Grid


def get_subview(env, pos, view_size=7, rgb=False):
    """
    Gets the subview of the grid, putting the given position in the center
    returns: encoded array of the view (7, 7, 3)
    """
    top_offset = int(view_size / 2)

    topX = pos[0] - top_offset
    topY = pos[1] - top_offset

    # print(env.agent_pos)

    grid = env.grid.slice(topX, topY, width=view_size, height=view_size)
    grid = grid.encode()

    if rgb:
        grid, mask = Grid.decode(grid)
        agent_pos = to_relative_pos(env.agent_pos, topX, topY, view_size)

        rgb_img = grid.render(
            8,
            agent_pos=agent_pos,
            agent_dir=env.agent_dir,
            highlight_mask=mask
        )
        return rgb_img

    return grid


def to_relative_pos(agent_pos, topX, topY, size):
    """
    Translates absolute coordinates into relative coordinates in
    relation to the given starting point.
    return: New relative coordinates, if the original position is not within
    the given subview, then None
    """

    relative_x = agent_pos[0] - topX
    relative_y = agent_pos[1] - topY

    valid_x = 0 <= relative_x < size
    valid_y = 0 <= relative_y < size

    if valid_x and valid_y:
        relative_pos = (relative_x, relative_y)
        return relative_pos

    return None
