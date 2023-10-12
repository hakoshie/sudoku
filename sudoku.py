import cv2
import numpy as np
import numpy.typing as npt


# def solve(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.int32]:

#     return np.zeros((9, 9), dtype="int32")
from itertools import product
import recognize 

def solve(image):

    problem = np.array(recognize.recognize(image))
    stk = []
    stk.append(problem.copy())
    solution = np.ones((9, 9), dtype="int32") 

    while len(stk) != 0:
        P = stk.pop()
        success = True
        for i, j in product(range(P.shape[0]), range(P.shape[1])):
            if P[i, j] == 0:
                success = False
                row_nums = P[i, :]
                col_nums = P[:, j]
                k = 3 * (i // 3)
                l = 3 * (j // 3)
                blk_nums = P[k : k + 3, l : l + 3].flatten()

                used_nums = np.concatenate([row_nums, col_nums, blk_nums])
                unused_nums = [n for n in range(10) if not n in used_nums]
                for n in unused_nums:
                    new_P = P.copy()
                    new_P[i, j] = n
                    stk.append(new_P)

                # if the first empty cell is found, then,
                # we don't need to see the following cells anymore.
                break

        if success:
            solution = P.copy()
            break
    # print(solution)

    return np.array(solution,dtype="int32")