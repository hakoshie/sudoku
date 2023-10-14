import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
# def solve(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.int32]:
#     return np.zeros((9, 9), dtype="int32")
from itertools import product
import recognize 
import re
import time
# Knuth's Algorithm X
def dfs(df: pd.DataFrame):
    # 使い勝手のためにNumPy配列に変換
    m = df.to_numpy()
    if m.size == 0:
        return []

    # 1の数が最小の行を探す
    sums = np.sum(m, axis=0)
    sc = np.argmin(sums)
    if sums[sc] == 0:
        return None

    sel_rows = np.where(m[:, sc])[0]
    for sr in sel_rows:
        mask_cols = np.where(m[sr, :])[0]
        mask_rows = [np.where(m[:, c])[0] for c in mask_cols]
        mask_rows = np.unique(np.concatenate(mask_rows))

        new_df = df.drop(index=df.index[mask_rows]).drop(columns=df.columns[mask_cols])
        ret = dfs(new_df)
        if ret is not None:
            return ret + [df.index[sr]]

    return None
def solve(image):
    start_time = time.time()

# 処理を実行する
# ...
    problem=np.array(recognize.recognize(image))
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("処理時間: {:.3f}秒".format(elapsed_time))
    index = [
        "R{:d}C{:d}#{:d}".format(i, j, n)
        for i in range(1, 10)
        for j in range(1, 10)
        for n in range(1, 10)
    ]
    cond1 = ["R{:d}C{:d}".format(i, j) for i in range(1, 10) for j in range(1, 10)]
    cond2 = ["R{:d}#{:d}".format(i, n) for i in range(1, 10) for n in range(1, 10)]
    cond3 = ["C{:d}#{:d}".format(j, n) for j in range(1, 10) for n in range(1, 10)]
    cond4 = ["B{:d}#{:d}".format(b, n) for b in range(1, 10) for n in range(1, 10)]
    conds = cond1 + cond2 + cond3 + cond4

    df = pd.DataFrame(index=index, columns=conds, dtype="bool")
    df.iloc[:] = 0
    for i in range(9):
        for j in range(9):
            for n in range(1, 10):
                r = i + 1
                c = j + 1
                b = (i // 3) * 3 + (j // 3) + 1
                op = "R{:d}C{:d}#{:d}".format(r, c, n)
                cd1 = "R{:d}C{:d}".format(r, c)
                cd2 = "R{:d}#{:d}".format(r, n)
                cd3 = "C{:d}#{:d}".format(c, n)
                cd4 = "B{:d}#{:d}".format(b, n)
                df.loc[op, [cd1, cd2, cd3, cd4]] = 1
    for i in range(9):
        for j in range(9):
            if problem[i, j] != 0:
                r = i + 1
                c = j + 1
                b = (i // 3) * 3 + (j // 3) + 1
                n = problem[i, j]
                # print(n)
                del_op1 = ["R{:d}C{:d}#{:d}".format(r, c, n_) for n_ in range(1, 10) if n != n_]
                del_op2 = ["R{:d}C{:d}#{:d}".format(r_, c, n) for r_ in range(1, 10) if r != r_]
                del_op3 = ["R{:d}C{:d}#{:d}".format(r, c_, n) for c_ in range(1, 10) if c != c_]
                del_op4 = [
                    "R{:d}C{:d}#{:d}".format(r_, c_, n)
                    for r_ in range(i // 3 * 3 + 1, i // 3 * 3 + 4)
                    for c_ in range(j // 3 * 3 + 1, j // 3 * 3 + 4)
                    if r != r_ or c != c_
                ]

                del_op = list(set(del_op1 + del_op2 + del_op3 + del_op4))
                df.drop(index=[id for id in del_op if id in df.index], inplace=True)
    solution=np.ones((9,9),dtype="int32")
    ans=dfs(df)
    if ans:
        for a in ans:
            # print(a)
            _,r,c,n= re.split("[RC#]", a)
            solution[int(r) - 1, int(c) - 1] = int(n)
    return solution
def solve_back(image):
    
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