import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def renderEnv(goal):
    sizeX = 10
    sizeY = 10

    hasObstacle = None
    goal1_x = goal[0]
    goal1_y = goal[1]

    agent_x = 4
    agent_y = 4

    partial = 5

    a = np.zeros([sizeX, sizeY, 3], dtype='uint8')
    a[:, :, :] = 128
    a[1:-1, 1:-1, :] = 255

    if (hasObstacle is not None):
        a[1:3, 5, :] = 128
        a[4:7, 5, :] = 128
        a[8:10, 5, :] = 128
        a[5, 1:3, :] = 128
        a[5, 4:7, :] = 128
        a[5, 8:10, :] = 128

    # goal1 is blue
    a[goal1_x, goal1_y, 0] = 0
    a[goal1_x, goal1_y, 1] = 0
    a[goal1_x, goal1_y, 2] = 255

    # agent in black
    a[agent_x, agent_y, 0] = 0
    a[agent_x, agent_y, 1] = 0
    a[agent_x, agent_y, 2] = 0

    # total = cv2.resize(a, (44, 44), interpolation=cv2.INTER_NEAREST)
    total = a# cv2.resize(a, (44, 44), interpolation=cv2.INTER_NEAREST)

    # total[0, :, :] = 0
    # total[7, :, :] = 0
    # total[13, :, :] = 0
    # total[19, :, :] = 0
    # total[26, :, :] = 0
    # total[32, :, :] = 0
    # total[39, :, :] = 0
    # total[45, :, :] = 0
    # total[52, :, :] = 0
    # total[58, :, :] = 0
    # total[65, :, :] = 0
    # total[71, :, :] = 0
    # total[78, :, :] = 0
    # total[83, :, :] = 0
    #
    # total[:, 0, :] = 0
    # total[:, 7, :] = 0
    # total[:, 13, :] = 0
    # total[:, 19, :] = 0
    # total[:, 26, :] = 0
    # total[:, 32, :] = 0
    # total[:, 39, :] = 0
    # total[:, 45, :] = 0
    # total[:, 52, :] = 0
    # total[:, 58, :] = 0
    # total[:, 65, :] = 0
    # total[:, 71, :] = 0
    # total[:, 78, :] = 0
    # total[:, 83, :] = 0

    if partial is not None:
        if (partial == 3):
            a = a[agent_x - 1:agent_x + 2, agent_y - 1:agent_y + 2, :]
        else:
            partial_image = np.zeros([5, 5, 3], dtype='uint8')
            xmin = agent_x - 2
            xmax = agent_x + 3
            ymin = agent_y - 2
            ymax = agent_y + 3

            if (xmin < 0):
                if (ymin < 0):
                    partial_image[1:5, 1:5, :] = a[0:4, 0:4, :]
                elif (ymax > sizeY):
                    partial_image[1:5, 0:4, :] = a[0:4, sizeY - 4:sizeY, :]
                else:
                    partial_image[1:5, 0:5, :] = a[0:4, ymin:ymax, :]
            elif (xmax > sizeX):
                if (ymin < 0):
                    partial_image[0:4, 1:5, :] = a[sizeX - 4:sizeX, 0:4, :]
                elif (ymax > sizeY):
                    partial_image[0:4, 0:4, :] = a[sizeX - 4:sizeX, sizeY - 4:sizeY, :]
                else:
                    partial_image[0:4, 0:5, :] = a[sizeX - 4:sizeX, ymin:ymax, :]
            else:
                if (ymin < 0):
                    partial_image[0:5, 1:5, :] = a[xmin:xmax, 0:4, :]
                elif (ymax > sizeY):
                    partial_image[0:5, 0:4, :] = a[xmin:xmax, sizeY - 4:sizeY, :]
                else:
                    partial_image[0:5, 0:5, :] = a[xmin:xmax, ymin:ymax, :]
            a = partial_image
    else:
        partial_image = np.zeros([sizeX, sizeY, 3], dtype='uint8')
        xmin = agent_x - 5
        xmax = agent_x + 6
        ymin = agent_y - 5
        ymax = agent_y + 6

        if (xmin < 0):
            if (ymin < 0):
                partial_image[-xmin:sizeX, -ymin:sizeY, :] = a[0:sizeX + xmin, 0:sizeY + ymin, :]
            elif (ymax > sizeY):
                partial_image[-xmin:sizeX, 0:sizeY + sizeY - ymax, :] = a[0:sizeX + xmin,
                                                                                       ymax - sizeY:sizeY, :]
            else:
                partial_image[-xmin:sizeX, 0:sizeY, :] = a[0:sizeX + xmin, ymin:ymax, :]
        elif (xmax > sizeX):
            if (ymin < 0):
                partial_image[0:sizeX + sizeX - xmax, -ymin:sizeY, :] = a[xmax - sizeX:sizeX,
                                                                                       0:sizeY + ymin, :]
            elif (ymax > sizeY):
                partial_image[0:sizeX + sizeX - xmax, 0:sizeY + sizeY - ymax, :] = a[
                                                                                                       xmax - sizeX:sizeX,
                                                                                                       ymax - sizeY:sizeY,
                                                                                                       :]
            else:
                partial_image[0:sizeX + sizeX - xmax, 0:sizeY, :] = a[xmax - sizeX:sizeX,
                                                                                   ymin:ymax, :]
        else:
            if (ymin < 0):
                partial_image[0:sizeX, -ymin:sizeY, :] = a[xmin:xmax, 0:sizeY + ymin, :]
            elif (ymax > sizeY):
                partial_image[0:sizeX, 0:sizeY + sizeY - ymax, :] = a[xmin:xmax,
                                                                                   ymax - sizeY:sizeY, :]
            else:
                partial_image[0:sizeX, 0:sizeY, :] = a[xmin:xmax, ymin:ymax, :]
        a = partial_image

    # patial = cv2.resize(a, (44, 44), interpolation=cv2.INTER_NEAREST)

    return total, a


def getImage(state):
    sizeX = 10
    sizeY = 10

    hasObstacle = None
    goal1_x = state[0]
    goal_y = state[1]

    agent_x = 4
    agent_y = 4

    partial = 5

    a = np.zeros([sizeX, sizeY, 3], dtype='uint8')
    a[:, :, :] = 128
    a[1:-1, 1:-1, :] = 255

    if (hasObstacle is not None):
        a[1:3, 5, :] = 128
        a[4:7, 5, :] = 128
        a[8:10, 5, :] = 128
        a[5, 1:3, :] = 128
        a[5, 4:7, :] = 128
        a[5, 8:10, :] = 128


    a[goal1_x, goal_y, 0] = 0
    a[goal1_x, goal_y, 1] = 0
    a[goal1_x, goal_y, 2] = 255

    # agent in black
    a[agent_x, agent_y, 0] = 0
    a[agent_x, agent_y, 1] = 0
    a[agent_x, agent_y, 2] = 0

    if partial is not None:
        if (partial == 3):
            a = a[state[0] - 1:state[0] + 2, state[1] - 1:state[1] + 2, :]
        else:
            partial_image = np.zeros([5, 5, 3], dtype='uint8')
            xmin = state[0] - 2
            xmax = state[0] + 3
            ymin = state[1] - 2
            ymax = state[1] + 3

            if (xmin < 0):
                if (ymin < 0):
                    partial_image[1:5, 1:5, :] = a[0:4, 0:4, :]
                elif (ymax > sizeY):
                    partial_image[1:5, 0:4, :] = a[0:4, sizeY - 4:sizeY, :]
                else:
                    partial_image[1:5, 0:5, :] = a[0:4, ymin:ymax, :]
            elif (xmax > sizeX):
                if (ymin < 0):
                    partial_image[0:4, 1:5, :] = a[sizeX - 4:sizeX, 0:4, :]
                elif (ymax > sizeY):
                    partial_image[0:4, 0:4, :] = a[sizeX - 4:sizeX, sizeY - 4:sizeY, :]
                else:
                    partial_image[0:4, 0:5, :] = a[sizeX - 4:sizeX, ymin:ymax, :]
            else:
                if (ymin < 0):
                    partial_image[0:5, 1:5, :] = a[xmin:xmax, 0:4, :]
                elif (ymax > sizeY):
                    partial_image[0:5, 0:4, :] = a[xmin:xmax, sizeY - 4:sizeY, :]
                else:
                    partial_image[0:5, 0:5, :] = a[xmin:xmax, ymin:ymax, :]
            a = partial_image
    else:
        partial_image = np.zeros([sizeY, sizeY, 3], dtype='uint8')
        xmin = state[0] - 5
        xmax = state[0] + 6
        ymin = state[1] - 5
        ymax = state[1] + 6

        if (xmin < 0):
            if (ymin < 0):
                partial_image[-xmin:sizeX, -ymin:sizeY, :] = a[0:sizeX + xmin, 0:sizeY + ymin,
                                                                       :]
            elif (ymax > sizeY):
                partial_image[-xmin:sizeX, 0:sizeY + sizeY - ymax, :] = a[0:sizeX + xmin,
                                                                                       ymax - sizeY:sizeY,
                                                                                       :]
            else:
                partial_image[-xmin:sizeX, 0:sizeY, :] = a[0:sizeX + xmin, ymin:ymax, :]
        elif (xmax > sizeX):
            if (ymin < 0):
                partial_image[0:sizeX + sizeX - xmax, -ymin:sizeY, :] = a[
                                                                                       xmax - sizeX:sizeX,
                                                                                       0:sizeY + ymin, :]
            elif (ymax > sizeY):
                partial_image[0:sizeX + sizeX - xmax, 0:sizeY + sizeY - ymax, :] = a[
                                                                                                       xmax - sizeX:sizeX,
                                                                                                       ymax - sizeY:sizeY,
                                                                                                       :]
            else:
                partial_image[0:sizeX + sizeX - xmax, 0:sizeY, :] = a[xmax - sizeX:sizeX,
                                                                                   ymin:ymax, :]
        else:
            if (ymin < 0):
                partial_image[0:sizeX, -ymin:sizeY, :] = a[xmin:xmax, 0:sizeY + ymin, :]
            elif (ymax > sizeY):
                partial_image[0:sizeX, 0:sizeY + sizeY - ymax, :] = a[xmin:xmax,
                                                                                   ymax - sizeY:sizeY, :]
            else:
                partial_image[0:sizeX, 0:sizeY, :] = a[xmin:xmax, ymin:ymax, :]

        a = partial_image

    # This transforms the grid of (5, 5, 3) to an rgb image of (44, 44, 3)
    # patial = cv2.resize(a, (44, 44), interpolation=cv2.INTER_NEAREST)

    return a # patial

if __name__ == "__main__":
    goal = (6, 6)
    goal_state = getImage(goal)
    full, partial = renderEnv(goal)
    print(partial.shape)

    concat = np.concatenate([goal_state, partial], axis=2)
    tf_concat = tf.concat([goal_state, partial], axis=2)
    print(concat.shape)
    print(tf_concat.shape)

    # plt.imshow(concat, aspect='auto')
    # plt.show()

    # new_state = np.reshape(my_image, [5808]) / 255.0