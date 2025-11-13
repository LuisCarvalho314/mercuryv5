import numpy as np


class Agent:
    """
    Class containing functionality for agent interaction within the maze environment.
    Is distinct from control scripts for agents.
    Contains information on agents location and impact of actions that is not available to the agent control scripts.
    """

    def __init__(self, initial_position,maze, sensors=None):
        if sensors is None:
            sensors = {"sensor": "cartesian"}
        self.position = initial_position
        self.action_dict = {
            "north": (self.go_north, [1, 0, 0, 0]),
            "east": (self.go_east, [0, 1, 0, 0]),
            "south": (self.go_south, [0, 0, 1, 0]),
            "west": (self.go_west, [0, 0, 0, 1]),
        }
        self.sensors = sensors
        self.observation_dict = {
            "cartesian": self.cartesian_obs,
            "cardinal distance": self.cardinal_distance,
            "floor": self.floor,
        }
        self.observation = self.make_observation(maze)
        self.action = None

    def go_west(self, delta=1):
        new_row = self.position[0]
        new_col = self.position[1] - delta
        return new_row, new_col

    def go_east(self, delta=1):
        new_row = self.position[0]
        new_col = self.position[1] + delta
        return new_row, new_col

    def go_north(self, delta=1):
        new_row = self.position[0] - delta
        new_col = self.position[1]
        return new_row, new_col

    def go_south(self, delta=1):
        new_row = self.position[0] + delta
        new_col = self.position[1]
        return new_row, new_col

    def make_observation(self, maze):
        observation = self.observation_dict[self.sensors["sensor"]](maze)
        self.observation = observation
        return observation

    def cartesian_obs(self, maze):
        return self.position

    def cardinal_distance(self, maze):
        sensor_range = self.sensors["range"]
        if sensor_range is not None:
            return self.limited_range_sensor(sensor_range=sensor_range, maze=maze)
        return self.unlimited_range_sensor(maze=maze)

    def floor(self, maze):

        return np.array([maze[self.position[0], self.position[1]],
                maze[self.position[0], self.position[1]]])

    def limited_range_sensor(self, sensor_range, maze):
        distances = {"north": sensor_range, "east": sensor_range, "south": sensor_range, "west": sensor_range}
        for delta in np.arange(0, sensor_range + 1):
            for direction in distances.keys():
                distances[direction] = int(self.cardinal_direction(
                    direction=direction, distances=distances, maze=maze,
                    delta=delta, sensor_range=sensor_range))

        return tuple(distances.values())

    def unlimited_range_sensor(self, maze):
        sensor_range = max(maze.shape[0], maze.shape[1])
        distances = {"north": sensor_range, "east": sensor_range, "south": sensor_range, "west": sensor_range}
        for delta in np.arange(0, sensor_range):
            for direction in distances.keys():
                distances[direction] = int(self.cardinal_direction(
                    direction=direction, distances=distances, maze=maze,
                    delta=delta, sensor_range=sensor_range))

        return tuple(distances.values())


    def cardinal_direction(self, direction, distances, maze, delta, sensor_range):
        if distances[direction] < sensor_range:
            return distances[direction]
        new_pos = self.action_dict[direction][0](delta=delta)
        is_wall = maze[new_pos]
        if is_wall:
            return delta - 1
        return sensor_range

    def take_action(self, maze, action):
        if action in self.action_dict:
            do_action, action_hot = self.action_dict[action]
            new_position = do_action()
            collision = self.valid_position(maze=maze, new_position=new_position)
            if not collision:
                self.position = new_position
            return self.position, action_hot ,collision

    # @staticmethod
    # def valid_position(maze, new_position):
    #     if maze[new_position[0]][new_position[1]] == 0:
    #         return False
    #     # print("COLLISION")
    #     return True
    @staticmethod
    def valid_position(maze, new_position):
        return maze[new_position[0]][new_position[1]]