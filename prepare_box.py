import random as rd
import numpy as np


def prepare_box(board_cellsize, x1, y1, x2, y2):
    lower_list = []
    upper_list = []
    for i in range(x1, x2, 2):
        for j in range(y1, y2, 2):
            randomH = rd.randrange(-1, -4, -1)
            box_lower = board_cellsize * \
                np.array([[i, j,  0], [i+1, j,  0], [i+1, j+1,  0], [
                    i, j+1,  0]])
            lower_list.append(box_lower)
            box_upper = board_cellsize * \
                np.array([[i, j, randomH], [i+1, j, randomH], [i+1, j+1, randomH],
                          [i, j+1, randomH]])  # 카메라 방향으로의 z값이 음수
            upper_list.append(box_upper)
    return lower_list, upper_list
