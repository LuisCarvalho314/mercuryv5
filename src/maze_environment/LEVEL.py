levels = [""]
level_1 = [
    "XXXXXXXXXXX",
    "XXP  XXXT X",
    "XXX XXXX XX",
    "X   XXXX XX",
    "X XXXXX  XX",
    "X XXXXX XXX",
    "X        XX",
    "XXXXXXXX XX",
    "XXXXXXXX TX",
    "XXXXXXXXXXX",
]

level_2 = [
    "XXXXXXXXXXXXXXXXX",
    "XP              X",
    "XXX XXXX X XX X X",
    "X X    X X    X X",
    "X XXXXXX XXXXXX X",
    "X        X      X",
    "XXXXXXXXXXX X XXX",
    "X           X   X",
    "X X X X XXX X XXX",
    "X   XXX X X X   X",
    "XX X  X X X XXXXX",
    "X  XX X X X     X",
    "XX    X X   XXX X",
    "X  XXXXXX X X X X",
    "X     X       X X",
    "X XTXXXX XXXX XXX",
    "XXXXXXXXXXXXXXXXX",
]

level_3 = [
    "XXXXXXX XXXXX",
    "XP    X X   X",
    "X     CRE K X",
    "X     X X   X",
    "XK    X X   X",
    "X     L XCXLX",
    "X     X  D   ",
    "XXEXXXX XEXXX",
    "  U     X  KX",
    "XXCXXXX X   X",
    "X     X X   X",
    "X     EAC   L",
    "X     X X   X",
    "XK    X X   X",
    "XXXXLXX XXXXX",
]
level_4 = [
    "XXXXXXXXX",
    "XP      X",
    "X   X   X",
    "X       X",
    "X       X",
    "X   X   X",
    "X   X   X",
    "X       X",
    "XXXXXXXXX",
]
level_5 = [
    "XXXXXXXXXXXXXXXXX",
    "XP    X         X",
    "X     X XXX   X X",
    "X     X       X X",
    "X XX XXX XXXXXX X",
    "X    X          X",
    "X    XXXXX  XXXXX",
    "X  X            X",
    "X  X  X X X   X X",
    "X  X  X XXX     X",
    "XXXX  X   X  XXXX",
    "X  X           TX",
    "X  X XXXXX  X   X",
    "X  X   X    X   X",
    "X      X  XXXXXXX",
    "X      X        X",
    "XXXXXXXXXXXXXXXXX",
]
level_6 = [
    "XXXXXXXXXXXXXXXX",
    "XP             X",
    "X              X",
    "X              X",
    "X              X",
    "X              X",
    "X              X",
    "X              X",
    "X              X",
    "X             EX",
    "XXXXXXXXXXXXXXXX",
]

level_7 = [
    "XXXXXXX",
    "X    TX",
    "XX XXXX",
    "X     X",
    "XX XXXX",
    "XP    X",
    "XXXXXXX",
]

# from openpyxl import load_workbook

# workbook = load_workbook(filename="TMGWR\MAZE_LAYOUT.xlsx")
# workbook2 = load_workbook(filename="TMGWR\MAZE_LAYOUT_A.xlsx")
# workbook3 = load_workbook(filename="TMGWR\MAZE_LAYOUT_KEY_DOOR.xlsx")

# maze = workbook.active
# maze2 = workbook2.active
# maze3 = workbook3.active

level_8 = []
level_9 = []
level_10 = []
level_11 = [
    "XXXXX",
    "X X X",
    "X P X",
    "XXXXX",
]

level_12 = [
    "XXXXXXX",
    "X XXX X",
    "X  P  X",
    "XXXXXXX",
]

level_13 = [
    "XXXXX",
    "X P X",
    "X X X",
    "X   X",
    "XXXXX",
]

level_14 = [
    "XXXXXXX",
    "X X X X",
    "X  P  X",
    "XXXXXXX",
]

level_15 = [
    "XXXXXX",
    "X P XX",
    "X X XX",
    "X X  X",
    "XXXXXX",
]

level_16 = [
    "XXXXXXXXXXXXXXXXXX",
    "XP               X",
    "XXXXXXXXXXXXXXXXXX"
]

level_17 = [
    "XXXXXXX",
    "XP    X",
    "XXXXXXX"
]


# for value in maze.iter_rows(
#     min_row=1, max_row=17, min_col=1, max_col=17, values_only=True
# ):
#     level_8.append(value)

# for value in maze2.iter_rows(
#     min_row=1, max_row=10, min_col=1, max_col=10, values_only=True
# ):
#     level_9.append(value)

# for value in maze3.iter_rows(
#     min_row=1, max_row=17, min_col=1, max_col=17, values_only=True
# ):
#     level_10.append(value)

levels.append(level_1)
levels.append(level_2)
levels.append(level_3)
levels.append(level_4)
levels.append(level_5)
levels.append(level_6)
levels.append(level_7)
levels.append(level_8)
levels.append(level_9)
levels.append(level_10)
levels.append(level_11)
levels.append(level_12)
levels.append(level_13)
levels.append(level_14)
levels.append(level_15)
levels.append(level_16)
levels.append(level_17)
