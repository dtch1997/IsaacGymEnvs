import numpy as np
from isaacgym.terrain_utils import SubTerrain

def ramp_terrain(terrain: SubTerrain, start_x: float = 0.0, stop_x: float = 1.0, start_y: float = 0.0, stop_y: float = 1.0, max_height: float = 1.0) -> SubTerrain:
    start_x = int(start_x * terrain.width)
    stop_x = int(stop_x * terrain.width)
    start_y = int(start_y * terrain.length)
    stop_y = int(stop_y * terrain.length)
    terrain.height_field_raw[:,:] = 0

    max_height = int(max_height / terrain.vertical_scale)
    heights = np.linspace(max_height, 0, (stop_x - start_x)).reshape(-1,1)
    terrain.height_field_raw[start_x : stop_x, start_y : stop_y] = heights
    return terrain

def platform_terrain(terrain: SubTerrain, start_x: float = 0.0, stop_x: float = 1.0, start_y: float = 0.0, stop_y: float = 1.0, platform_height: float = 1.0):
    """ generate a platform terrain """
    start_x = int(start_x * terrain.width)
    stop_x = int(stop_x * terrain.width)
    start_y = int(start_y * terrain.length)
    stop_y = int(stop_y * terrain.length)
    terrain.height_field_raw[:,:] = 0
    terrain.height_field_raw[start_x : stop_x , start_y : stop_y] = int(platform_height / terrain.vertical_scale)
    return terrain