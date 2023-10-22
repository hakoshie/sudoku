import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
# def solve(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.int32]:
#     return np.zeros((9, 9), dtype="int32")
import recognize 
import recog_l3
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
            if violation_count>=0:
                violations.append((i, j, violation_count))

    return violations
def count_violations(board):
    violations = 0
    # 行の制約をチェック
    for i in range(9):
        row = board[i, :]
        for j in range(1, 10):
            j_cnt=np.count_nonzero(row == j)
            if  j_cnt> 1:
                violations += j_cnt
    # 列の制約をチェック
    for j in range(9):
        col = board[:, j]
        for i in range(1, 10):
            i_cnt=np.count_nonzero(col == i)
            if i_cnt> 1:
                violations += i_cnt
    # ボックスの制約をチェック
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            box = board[i:i+3, j:j+3].flatten()
            for k in range(1, 10):
                k_cnt=np.count_nonzero(box == k)
                if k_cnt > 1:
                    violations += k_cnt
    return violations
def violation_check(board):
    # O(10^3)
    for i in range(9):
        for j in range(9):
            num = board[i][j]
            if num != 0:
                # 同じ列に同じ数字があるか確認
                for k in range(9):
                    if k != i and board[k][j] == num:
                        return False

                # 同じ行に同じ数字があるか確認
                for k in range(9):
                    if k != j and board[i][k] == num:
                        return False

                # 同じ3x3のブロックに同じ数字があるか確認
                block_start_i = (i // 3) * 3
                block_start_j = (j // 3) * 3
                for x in range(3):
                    for y in range(3):
                        if (block_start_i + x != i or block_start_j + y != j) and board[block_start_i + x][block_start_j + y] == num:
                            return False    
    return True

def solve(image):
    try:
        problem1=np.array(recog_l3.recognize(image,pass_image=True),dtype=np.int32)
        if problem1.shape!=(9,9):
            problem1=np.ones((9,9),dtype=np.int32)
    except:
        problem1=np.ones((9,9),dtype=np.int32)  
    # print(problem1.shape)
    problem2=np.array(recognize.recognize(image))
    # 違反回数で降順ソート
    if count_violations(problem1)>=count_violations(problem2):
        problem=problem2
    else:
        problem=problem1
    violations=count_constraint_violations(problem)
    sorted_violations = sorted(violations, key=lambda x: x[2], reverse=True)
    # 上位5つのタプルを取得
    potential_miss = sorted_violations[:10]

    candidates=[[0, 5, 0],
                [1, 4, 0],
                [2, 3, 0],
                [3, 1, 0],
                [4, 9, 0],
                [5, 9, 0],
                [6, 8, 0],
                [7, 2, 0],
                [8, 6, 0],
                [9, 5, 0]]
    ans=None
    try:
        for solution in sudoku_solver.solve_sudoku((3,3),problem):
            if ans is not None:
                break
            ans=np.array(solution)
    except:
        # return np.ones((9,9),dtype=np.int32)

        
        ### 候補3つ
        # K=min(9,len(potential_miss))
        # max_trial=200
        # n_trial=0
        # for i in range(3**K):
        #     if n_trial>=max_trial:
        #         break
        #     candidate_idx = [(i // 3**j) % 3 for j in range(K)]
        #     problem_tmp=problem.copy()
        #     for j,idx in enumerate(candidate_idx):
        #         i_x,j_y,_=potential_miss[j]
        #         num_ij=problem[i_x][j_y]
        #         problem_tmp[i_x][j_y]=candidates[num_ij][idx]
        #     if violation_check(problem_tmp):
        #         n_trial+=1
        #         try:
        #             for solution in sudoku_solver.solve_sudoku((3,3),problem_tmp):
        #                 if ans is not None:
        #                     break
        #                 ans=np.array(solution)
        #             if ans is not None:
        #                 break
        #         except:
        #             continue
    if ans is None:
        return np.ones((9,9),dtype=np.int32)
    
    return np.array(ans,dtype=np.int32)