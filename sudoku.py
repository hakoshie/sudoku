import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
# def solve(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.int32]:
#     return np.zeros((9, 9), dtype="int32")
import recognize 
import sudoku_solver
import time

def count_violations(board):
    violations = 0
    # 行の制約をチェック
    for i in range(9):
        row = board[i, :]
        for j in range(1, 10):
            if np.count_nonzero(row == j) > 1:
                violations += 1
    # 列の制約をチェック
    for j in range(9):
        col = board[:, j]
        for i in range(1, 10):
            if np.count_nonzero(col == i) > 1:
                violations += 1
    # ボックスの制約をチェック
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            box = board[i:i+3, j:j+3].flatten()
            for k in range(1, 10):
                if np.count_nonzero(box == k) > 1:
                    violations += 1
    return violations
def solve(image):
    # start_time = time.time()

    # 処理を実行する
    # ...
    problem=np.array(recognize.recognize(image))
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print("処理時間: {:.3f}秒".format(elapsed_time))
    cnt=0
    ans=None
    try:
        for solution in sudoku_solver.solve_sudoku((3,3),problem):
            cnt+=1
            ans=np.array(solution)
            if cnt>1:
                break
    except:
        return np.ones((9,9),dtype=np.int32)
    if cnt==0:
        return np.ones((9,9),dtype=np.int32)
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print("処理時間: {:.3f}秒".format(elapsed_time))
    return np.array(ans,dtype=np.int32)