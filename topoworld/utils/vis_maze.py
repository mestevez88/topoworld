import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


def plot_state_function(state_function: np.ndarray, adjacency: np.ndarray, n_tiles_per_state: int = 15,
                        ax: plt.Axes = None, cmap: str = "Reds"):
    n = n_tiles_per_state
    single_state_tiles = np.arange(n * n).reshape(n, n)
    i, j = state_function.shape

    a_shape = adjacency.shape
    assert len(a_shape) == 2
    assert a_shape[0] == a_shape[1]

    assert i*j == a_shape[0]

    total_mask = np.pad(np.zeros([n * i, n * j]) != 0, 1, constant_values=True)

    for state in range(i * j):
        mask = np.zeros([n, n]) != 0
        mask[0, -1] = True
        mask[0, 0] = True
        mask[-1, 0] = True
        mask[-1, -1] = True
        if not ((state % j != 0) and adjacency[state, state - 1]):
            mask = mask | (single_state_tiles % n == 0)

        if not ((state <= (i * j - j - 1)) and adjacency[state, state + j]):
            mask = mask | (single_state_tiles > (n * n - n - 1))

        if not ((state % j < (j - 1)) and adjacency[state, state + 1]):
            mask = mask | (single_state_tiles % n == (n - 1))

        if not ((state >= j) and adjacency[state, state - j]):
            mask = mask | (single_state_tiles < n)

        x = state // j
        y = state % j

        total_mask[1 + n * x:1 + n * x + n, 1 + n * y:1 + n * y + n] = mask

    func_enlarged = np.pad(np.repeat(np.repeat(state_function, n, axis=0), n, axis=1), 1, mode="edge")

    masked_func = np.ma.masked_where(total_mask, func_enlarged)

    pcm = ax.imshow(masked_func, cmap=cmap)

    ax.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,  # ticks along the bottom edge are off
        right=False,  # ticks along the top edge are off
        labelbottom=False,
        labelleft=False)  # labels along the bottom edge are off

    ax.set_facecolor('grey')

    return pcm


def draw_maze(i, j, adjacency, ax):
    return plot_state_function(np.ones([i, j]), adjacency, ax=ax)


def draw_maze_collection(maze_collection):
    n_mazes = maze_collection["info"]["n_mazes"]
    lx = maze_collection["info"]["lx"]
    lz = maze_collection["info"]["lz"]
    pis = maze_collection["info"]["pis"]
    _, axs = plt.subplots(n_mazes, len(pis), figsize=(2*n_mazes, 2*len(pis)))
    for j, pi in enumerate(pis):
        axs[0, j].set_title(f"pi_1={pi}")
        for i, (ax, maze) in enumerate(zip(axs[:, j], maze_collection["mazes"][pi])):
            draw_maze(lx, lz, maze, ax=ax)
    plt.show()


def draw_trajectory(i, j, adjacency, trajectory, n_tiles_per_state: int = 15):
    # import networkx as nx
    # distance_to_goal = nx.shortest_path_length(nx.from_numpy_array(adjacency), target=i*j-1)
    # color_states = np.array([distance_to_goal[state] for state in range(i*j)]).reshape(i, j)
    color_states = np.ones([i, j])

    fig, ax = plt.subplot_mosaic("""AAABF;AAACF;AAADF;AAAEF""")

    plot_state_function(color_states, adjacency, ax=ax["A"], n_tiles_per_state=n_tiles_per_state, cmap="Reds")

    where_to_split = [0]

    for s in list(np.where(trajectory == np.max(trajectory))[0] + 1) + [5000]:
        while s - where_to_split[-1] > 100:
            where_to_split.append(where_to_split[-1] + 100)
        where_to_split.append(s)
    where_to_split = where_to_split[1:-1]

    episodes = np.array_split(trajectory, where_to_split)

    tile_center = n_tiles_per_state // 2
    tiling = lambda state: [tile_center + (state % j) * n_tiles_per_state + 1,
                            tile_center + (state // j) * n_tiles_per_state + 1]

    cmap = mpl.cm.magma
    colors = cmap(np.linspace(0, 1, len(episodes)))
    alphas = np.linspace(0, 1, len(episodes))
    linewidths = np.linspace(20, 1, len(episodes))

    max_eps_len = max([len(episode) for episode in episodes])

    for i, episode in enumerate(episodes):
        tile_indexes = np.array(list(map(tiling, episode))).transpose() + 2*np.random.rand(2, len(episode))
        x = tile_indexes[0]
        y = tile_indexes[1]

        ax["A"].plot(x, y, color=colors[i], alpha=alphas[i], linewidth=1)

    norm = mpl.colors.Normalize(vmin=0, vmax=len(episodes))

    cb1 = mpl.colorbar.ColorbarBase(ax["F"], cmap=cmap, norm=norm, orientation='vertical')
    cb1.set_label('Episode')
    fig.show()

    # TODO: plot progression of state occupancy
    plot_state_function(color_states, adjacency, ax=ax["B"], n_tiles_per_state=n_tiles_per_state, cmap="Reds")
    plot_state_function(color_states, adjacency, ax=ax["C"], n_tiles_per_state=n_tiles_per_state, cmap="Reds")
    plot_state_function(color_states, adjacency, ax=ax["D"], n_tiles_per_state=n_tiles_per_state, cmap="Reds")
    plot_state_function(color_states, adjacency, ax=ax["E"], n_tiles_per_state=n_tiles_per_state, cmap="Reds")
