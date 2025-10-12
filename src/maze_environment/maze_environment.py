import time
import turtle
import math
from LEVEL import *
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

# Define maze dimensions in pixels
RES = 24
HEIGHT = 25 * RES
WIDTH = 25 * RES
COLLISION_TOL = 0.1 * RES


# Initialise window for maze
wn = turtle.Screen()
wn.bgcolor("#7a0404")
wn.title("A maze navigation testbed")
wn.setup(HEIGHT, WIDTH)
wn.tracer(0)


# Register shapes
wn.register_shape("./images/Mars_Rover_Right.gif")
wn.register_shape("./images/Mars_Rover_Left.gif")
wn.register_shape("./images/Mars_Rover_Up.gif")
wn.register_shape("./images/Mars_Rover_Down.gif")
wn.register_shape("./images/Mars.gif")
wn.register_shape("./images/treasure_24.gif")


class Pen(turtle.Turtle):
    def __init__(self):
        turtle.Turtle.__init__(self)
        self.shape("./images/Mars.gif")
        self.color("#99391c")
        self.penup()
        self.speed(0)


class Player(turtle.Turtle):
    def __init__(self):
        turtle.Turtle.__init__(self)
        self.shape("./images/Mars_Rover_Right.gif")
        self.color("blue")
        self.penup()
        self.speed(0)
        self.gold = 0

    def is_collision(self, other):
        """Detect if agent has hit a wall"""
        a = self.xcor() - other.xcor()
        b = self.ycor() - other.ycor()
        distance = math.sqrt(a**2 + b**2)
        if distance < COLLISION_TOL:
            print("BUMP")
            return True
        return False

    """Movement functions"""

    def go_west(self):
        """Subtracts STEP from xcor"""
        move_to_x = self.xcor() - RES
        move_to_y = self.ycor()
        self.shape("./images/Mars_Rover_Left.gif")
        if (move_to_x, move_to_y) not in walls:
            self.goto(move_to_x, move_to_y)
        pass

    def go_east(self):
        """Adds STEP to xcor"""
        move_to_x = self.xcor() + RES
        move_to_y = self.ycor()
        self.shape("./images/Mars_Rover_Right.gif")
        if (move_to_x, move_to_y) not in walls:
            self.goto(move_to_x, move_to_y)
        pass

    def go_north(self):
        """Adds STEP to ycor"""
        move_to_x = self.xcor()
        move_to_y = self.ycor() + RES
        self.shape("./images/Mars_Rover_Up.gif")
        if (move_to_x, move_to_y) not in walls:
            self.goto(move_to_x, move_to_y)

        pass

    def go_south(self):
        """Subtracts STEP from ycor"""
        move_to_x = self.xcor()
        move_to_y = self.ycor() - RES
        self.shape("./images/Mars_Rover_Down.gif")
        pen.shape("./images/Mars_Rover_Down.gif")
        if (move_to_x, move_to_y) not in walls:
            self.goto(move_to_x, move_to_y)
        pass

    def sensory_input(self, walls, sensory_method=None):
        """Provides sensory input via cartesian/polar cordinates, or distance
        to walls in cardinal directions"""

        if sensory_method == "LIDAR":
            [
                self.dist_north,
                self.dist_east,
                self.dist_south,
                self.dist_west,
            ] = (-1, -1, -1, -1)
            MAX_OBS = max(HEIGHT, WIDTH)
            [
                delta_N,
                delta_E,
                delta_S,
                delta_W,
            ] = (MAX_OBS, MAX_OBS, MAX_OBS, MAX_OBS)

            for delta in range(0, MAX_OBS, RES):
                if (
                    self.xcor(),
                    self.ycor() + delta,
                ) in walls and delta < delta_N:
                    self.dist_north = delta
                    delta_N = delta

                if (
                    self.xcor(),
                    self.ycor() - delta,
                ) in walls and delta < delta_S:
                    self.dist_south = delta
                    delta_S = delta

                if (
                    self.xcor() + delta,
                    self.ycor(),
                ) in walls and delta < delta_E:
                    self.dist_east = delta
                    delta_E = delta

                if (
                    self.xcor() - delta,
                    self.ycor(),
                ) in walls and delta < delta_W:
                    self.dist_west = delta
                    delta_W = delta
            self.obs = (
                np.array(
                    [
                        self.dist_north,
                        self.dist_east,
                        self.dist_south,
                        self.dist_west,
                    ]
                )
                / RES
            )

        elif sensory_method == "POLAR":
            a = self.xcor()
            b = self.ycor()
            self.dist_centre = np.sqrt(a**2 + b**2) / RES

            self.angle = np.degrees(np.arctan2(self.xcor(), self.ycor()))
            if self.angle < 0:
                self.angle += 360
            self.obs = np.array([self.dist_centre, self.angle])

        else:
            self.obs = np.array([player.xcor(), player.ycor()]) / RES

        # print(f"obs = {self.obs}")
        return self.obs


class Treasure(turtle.Turtle):
    def __init__(self, x, y):
        turtle.Turtle.__init__(self)
        self.shape("./images/treasure_24.gif")
        self.color("gold")
        self.gold = 100
        self.penup()
        self.speed()
        self.goto(x, y)
        pass

    def destroy(self):
        self.goto(2000, 2000)
        self.hideturtle()
        pass


treasures = []
goal = None


def setup_maze(level):
    """Builds maze from csv file"""
    for y, _ in enumerate(level):
        for x, _ in enumerate(level[y]):
            character = level[y][x]

            screen_x = -RES + (x * RES)
            screen_y = RES - (y * RES)

            if character == "X":
                pen.goto(screen_x, screen_y)
                pen.stamp()
                walls.append((screen_x, screen_y))

            """
            else:
                pen2.goto(screen_x,screen_y)
                pen2.stamp()
                empty.append(((screen_x,screen_y)))"""

            if character == "P":
                player.goto(screen_x, screen_y)
    pass


pen = Pen()
player = Player()

walls = []
empty = []
setup_maze(levels[15])


def main():
    action_dict = {
        "North": [1, 0, 0, 0],
        "East": [0, 1, 0, 0],
        "South": [0, 0, 1, 0],
        "West": [0, 0, 0, 1],
    }
    action_list = ["North", "East", "South", "West"]
    action = "Down"
    # action = None
    wn.tracer(0)

    Niter = 100
    Data = np.zeros((Niter, 3))
    SD = 0

    epsilon = 1
    exp_T = 500
    # valueClass = Value(model.W)

    bump = []
    BUMP = []
    score = []
    SCORE = []
    move_list = []
    states = []
    actions = []
    x = player.sensory_input(sensory_method="LIDAR", walls=walls)
    x_init = np.array([player.xcor(), player.ycor()])

    def random_action(action_list):
        rand_num = np.random.random()
        divisor = len(action_list)
        for i in range(divisor):
            if rand_num >= i / divisor and rand_num < (i + 1) / divisor:
                action_index = i
        action = action_list[action_index]

        return action

    action = random_action(action_list)
    steps = 20000
    moves = 0
    for n in range(steps):
        x1 = x

        action_list_temp = action_list
        action_prev = action

        if np.random.random() < 0.3:
            action = random_action(action_list)
            # print("NO BUMP CHANGED DIRECTION")
        else:
            action = action_prev
            # print("NO BUMP")
        # print(action)
        if action == "North":
            player.go_north()
            a = [1, 0, 0, 0]
        elif action == "East":
            player.go_east()
            a = [0, 1, 0, 0]
        elif action == "South":
            player.go_south()
            a = [0, 0, 1, 0]
        elif action == "West":
            player.go_west()
            a = [0, 0, 0, 1]

        # pen2.goto(player.xcor(), player.ycor())
        # pen2.stamp()
        # empty.append((player.xcor(), player.ycor()))

        actions.append(a)

        x = player.sensory_input(sensory_method="LIDAR", walls=walls)

        states.append(x)

        if (x == x1).all():
            # print(action_prev)
            # print(action_list_temp)
            action_list_temp = [j for j in action_list if j != action_prev]
            action = random_action(action_list_temp)
            # print(f"{x1} == {x}")

        # print(x-1)
        moves += 1

        if moves % 1000 == 0:
            print(moves)
        wn.update()
        # time.sleep(1)
    np.savetxt(
        "simple_maze_question_mark_LIDAR_data_2023_10_18.csv",
        states,
        delimiter=",",
    )
    np.savetxt(
        "simple_maze_question_mark_LIDAR_actions_2023_10_18.csv",
        actions,
        delimiter=",",
    )


hyperparameter_defaults = dict(nDim=6, Ni=6)

# wandb.init(project="TMGWR_LIDAR_Colour", name="test")

main()
