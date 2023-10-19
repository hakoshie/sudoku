import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
# def solve(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.int32]:
#     return np.zeros((9, 9), dtype="int32")
import recognize 
import sudoku_solver
import time


def count_constraint_violations(board):
    violations = []  # (i, j, 違反回数) のタプルを格納するリスト

    for i in range(9):
        for j in range(9):
            num = board[i][j]
            violation_count = 0  # 制約に違反している回数

            if num != 0:
                # 同じ列に同じ数字があるか確認
                for k in range(9):
                    if k != i and board[k][j] == num:
                        violation_count += 1

                # 同じ行に同じ数字があるか確認
                for k in range(9):
                    if k != j and board[i][k] == num:
                        violation_count += 1

                # 同じ3x3のブロックに同じ数字があるか確認
                block_start_i = (i // 3) * 3
                block_start_j = (j // 3) * 3
                for x in range(3):
                    for y in range(3):
                        if (block_start_i + x != i or block_start_j + y != j) and board[block_start_i + x][block_start_j + y] == num:
                            violation_count += 1
            if violation_count>0:
                violations.append((i, j, violation_count))

    return violations

def solve(image):
    # start_time = time.time()

    # 処理を実行する
    # ...
    problem=np.array(recognize.recognize(image))
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print("処理時間: {:.3f}秒".format(elapsed_time))
        # 違反回数で降順ソート
    violations = count_constraint_violations(problem)
    sorted_violations = sorted(violations, key=lambda x: x[2], reverse=True)
    
    # 上位5つのタプルを取得
    top_5_violations = sorted_violations[:10]

    candidates=[[0, 1, 9],
    [1, 3, 2],
    [2, 7, 3],
    [3, 2, 1],
    [4, 6, 1],
    [5, 3, 6],
    [6, 8, 5],
    [7, 1, 2],
    [8, 6, 2],
    [9, 6, 3]]
    cnt=0
    ans=None
    try:
        for solution in sudoku_solver.solve_sudoku((3,3),problem):
            if cnt>0:
                break
            cnt+=1
            ans=np.array(solution)
 
    except:
        cnt=0
        # return np.ones((9,9),dtype=np.int32)
        K=min(5,len(top_5_violations))
        for i in range(3**K):
            candidate_idx = [(i // 3**j) % 3 for j in range(K)]
            problem_tmp=problem.copy()
            for j,idx in enumerate(candidate_idx):
                i_x,j_y,_=top_5_violations[j]
                num_ij=problem[i_x][j_y]
                problem_tmp[i_x][j_y]=candidates[num_ij][idx]
            try:
                for solution in sudoku_solver.solve_sudoku((3,3),problem_tmp):
                    if cnt>0:
                        break
                    cnt+=1
                    ans=np.array(solution)

                if cnt>0:
                    break
            except:
                cnt=0
                continue
    if cnt==0:
        return np.ones((9,9),dtype=np.int32)
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print("処理時間: {:.3f}秒".format(elapsed_time))
    return np.array(ans,dtype=np.int32)