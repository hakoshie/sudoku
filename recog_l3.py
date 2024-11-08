import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os 

def is_square(cnt,sides):
    
    # 輪郭の面積と外接矩形の面積を比較
    area = cv2.contourArea(cnt)
    rect_area = sides[0]*sides[1]
    side_ratio=(sides[1]+sides[3])/(sides[0]+sides[2])
    # if side_ratio>2 or side_ratio<0.5:
    #     return False
    if area / rect_area < 0.3 or area / rect_area > 3:
        return False
    return True
def count_violations(board):
    violations = 0
    if np.count_nonzero(board) < 17:
        return 1000
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

def recognize(image,clf=None,scaler=None,pixel=20,ret_img=False,n_open=0,n_close=0,prior_close=False,trim_percentage=0.007,mean_white_axis=0,arc_epsilon=5e-2,erase_line=1,white_thres=245,otsu_times=1.05,clf_f_name="SVC",pixel_f=150,clf_f=None,scaler_f=None,sigmaColor=2,sigmaSpace=2,ret_num=False,clipLimit2=.46, tileGridSize2=7,n_dilate=4,n_erode=3,plt_res2=0,first_clahe=False,clipLimit1=.5,tileGridSize1=95,bilateral=1,mean_denoise=1,clahe_time1=1,clahe_time2=2,pass_image=0,plt_res3=0,plt_res1=0):

    if not pass_image:
        image = cv2.imread(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if scaler is None or clf is None:
        # model_name="MLPC_numbers_mix_v4"
        # model_name="MLPC_numbers_mix_v3"
        # model_name="MLPC_numbers_mix_v2"
        # model_name="MLPC_numbers_mix_line3_v2_m"
        # model_name="Rand_numbers_mix_l2"
        # model_name="MLPC_numbers_mix"
        model_name="ensemble_MLPC"
        scaler = pd.read_pickle(f'./models/{model_name}_scaler.pickle')        
        clf = pd.read_pickle(f'./models/{model_name}_clf.pickle')
    trim_percentage=0.001
    height, width, channels = image.shape[:3]
    trim_width = int(width * trim_percentage)
    trim_height = int(height * trim_percentage)
    image = image[trim_height:height - trim_height, trim_width:width - trim_width]


    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    gray= cv2.bilateralFilter(gray, 11, 2,2)
    # gray_gb = cv2.GaussianBlur(gray, None, 2.0)

    # ヒストグラム平坦化
    clahe = cv2.createCLAHE(clipLimit=clipLimit1, tileGridSize=(tileGridSize1,tileGridSize1))
    for i in range(clahe_time1):
        gray = clahe.apply(gray)
    # gray = cv2.equalizeHist(gray)
    # gray_gb =cv2.medianBlur(gray_gb, 5)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    # 大津の二値化
    thr, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    new_thr = min(int(thr * otsu_times), 255)
    _, binary = cv2.threshold(gray, new_thr, 255, cv2.THRESH_BINARY)
    # # plt.imshow(binary, cmap="gray")
    # # plt.title("Otsu's binarization (threshold={:d})".format(int(thr)))
    # # plt.show()
    edge = cv2.Canny(binary, 100, 200)
    edge = cv2.dilate(edge, np.ones((5, 5), dtype=edge.dtype),iterations=1)
    edge = cv2.erode(edge, np.ones((3, 3), dtype=edge.dtype),iterations=1)
    # res_close = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), dtype=binary.dtype))
    # plt.imshow(edge, cmap="gray")
    # plt.title("After morphology operation".format(thr))
    # plt.show()
    contours, _ = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result = image.copy()
    # cv2.drawContours(result, contours, -1, (0,0,0), 3, cv2.LINE_AA)
    # plt.imshow(result)
    # plt.show()
    # x, y, w, h = cv2.boundingRect(contours[1])
    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # # plt.imshow(image)
    # # plt.show()
    largest_cnt = None
    max_length = 0.0
    max_area = 0.0
    # result = image.copy()
    epsilon=3e-2
    for cnt in contours:
        # 輪郭線の長さを計算
        arclen = cv2.arcLength(cnt, True)
        approx=cv2.approxPolyDP(cnt, arclen *epsilon, True)
        area=cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(approx)
        internal_area_ratio = area / (w * h)
        if len(approx)<10 and len(approx) >= 4 and internal_area_ratio>0.5:
            # # print(len(approx), x,y,w,h, arclen,internal_area_ratio)
            # cv2.drawContours(result, [approx], -1, (0,0,0), 3, cv2.LINE_AA)
            # # plt.imshow(result)
            # # plt.show()
            sides = []
            points = []
            for i in range(4):
                x1, y1 = approx[i][0]
                x2, y2 = approx[(i + 1) % 4][0]
                side = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                points.append([x1, y1])
                sides.append(side)
            if  max_area < area and is_square(cnt,sides):
                max_area = area
                max_length = arclen
                largest_cnt = cnt
    # result = image.copy()

    # # plt.imshow(result)
    # # plt.show()
    # # print(len(longest_cnt))
    arclen = cv2.arcLength(largest_cnt, True)
    approx = cv2.approxPolyDP(largest_cnt, arclen * epsilon, True) 
    # cv2.drawContours(result, [approx], -1, (0,0,0), 3, cv2.LINE_AA)
    cv2.drawContours(result, [largest_cnt], -1, (0,0,0), 3, cv2.LINE_AA)
    if plt_res1:
        plt.imshow(result)
        plt.title("Red region has {:d} corners".format(len(approx)))
        plt.show()
    # print(len(approx))
    # 輪郭線の重心を計算する
    M = cv2.moments(approx)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # 各点に対して、重心からの距離を計算し、1.2倍した点を取得する
    new_approx = []
    for point in approx:
        x, y = point[0]
        dist = np.sqrt((cx - x)**2 + (cy - y)**2)
        dist_new = dist * 1.2
        x_new = int(cx - dist_new * (cx - x) / dist)
        y_new = int(cy - dist_new * (cy - y) / dist)
        new_approx.append([[x_new, y_new]])

    # 新しい近似を描画する
    new_approx = np.array(new_approx)
    # cv2.drawContours(result, [np.array(new_approx)], -1, (0, 0, 255), 3, cv2.LINE_AA)
    # plt.imshow(result)
    # plt.title("Red region has {:d} corners".format(len(new_approx)))
    # plt.show()
    ##################################
    ## さっきのところを切り取って射影変換
    ##################################    
    x, y, w, h = cv2.boundingRect(new_approx)
    # cropped = image[y:y+h, x:x+w]
    h,w= result.shape[:2]
    # パディングを追加する
    padding=150
    color = [255, 255, 255] #白
    result= cv2.copyMakeBorder(result, padding,padding,padding,padding, cv2.BORDER_CONSTANT, value=color)
    cropped = result[y+padding:y+padding+h, x+padding:x+padding+w]
    # plt.imshow(cropped)
    # plt.show()
    # # print(approx)
    new_approx=new_approx-(x,y)
    src_pts = new_approx.reshape((-1, 2)).astype("float32")
    center = np.mean(src_pts, axis=0)
    # ４点を重心からの角度でソートする
    # 時計回りになる
    src_pts = np.array(sorted(src_pts, key=lambda p: np.arctan2(p[1]-center[1], p[0]-center[0])))

    # 結果を表示する
    # print(src_pts)
    # 縦横比の計算
    w = np.linalg.norm(src_pts[3] - src_pts[0])
    h = np.linalg.norm(src_pts[1] - src_pts[0])
    aspect = abs(w) / abs(h)

    # 新しい画像サイズを設定
    new_w = int(1000*aspect)
    new_w = 800
    new_h = 800
    # dst_pts = np.array([(0, 0), (0, new_h), (new_w, new_h), (new_w, 0)], dtype="float32")
    dst_pts = np.array([(0, 0), (new_w, 0), (new_w, new_h), (0, new_h)], dtype="float32")

    # 射影変換を計算して、パースをキャンセルする
    try:
        warp = cv2.getPerspectiveTransform(src_pts, dst_pts)
        result = cv2.warpPerspective(cropped, warp, (new_w, new_h))
        result=cv2.resize(result, (1000,1000), interpolation=cv2.INTER_AREA)
    except:
        return -1
    


    n_close,n_open=2,2
    # n_dilate,n_erode=4,3
    #############################
    # 正方形を検出する
    ########################
    # 画像をグレースケールに変換する

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    gray= cv2.bilateralFilter(gray, 11, 2,2)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clipLimit2, tileGridSize=(tileGridSize2,tileGridSize2))
    for i in range(clahe_time2):
        gray = clahe.apply(gray)

    # Apply histogram equalization
    # gray = cv2.equalizeHist(gray)
    kernel = np.ones((2,2), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel,iterations=n_dilate)
    gray = cv2.morphologyEx(gray, cv2.MORPH_ERODE, kernel,iterations=n_erode)

    # Apply denoising
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

    # 大津の二値化を適用
    thresh, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    # new_thr = min(int(thresh * otsu_times), 255)
    # _, binary = cv2.threshold(gray, new_thr, 255, cv2.THRESH_BINARY)
    edge = cv2.Canny(binary, 100, 200)
    edge = cv2.dilate(edge, np.ones((11, 11), dtype=edge.dtype))
    edge = cv2.erode(edge, np.ones((9, 9), dtype=edge.dtype))
    # 輪郭を検出する
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # plt.imshow(binary, cmap="gray")
    # plt.show()
    # 正方形を検出する
    res2 = result.copy()
    count=0
    # contours=sorted(contours, key=lambda x:x[0][0][0])
    contours=sorted(contours, key=lambda c: (cv2.boundingRect(c)[0], cv2.boundingRect(c)[1]))
    # contours=sorted(contours, key=lambda x:x[0][0][1])
    squares=[]
    for cnt in contours:
        # 輪郭線の近似を行う
        arclen = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, arclen * 7e-2, True)
        if len(approx) == 4:

            # 各辺の長さを計算する
            sides = []
            points = []
            for i in range(4):
                x1, y1 = approx[i][0]
                x2, y2 = approx[(i + 1) % 4][0]
                side = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                points.append([x1, y1])
                sides.append(side)
            points=sorted(points, key=lambda x:x[0])
            if points[0][1]<points[1][1]:
                points[0],points[1]=points[1],points[0]
            
            # 各辺の長さがほぼ等しい場合、正方形とみなす
            # if np.std(sides) < np.mean(sides) :
            if is_square(cnt,sides):
                if max(sides) >= result.shape[0] / 2 or min(sides) <= result.shape[0] / 9:
                # if min(sides) <= result.shape[0] / 9:
                    continue
                cv2.drawContours(res2, [approx], -1, (255,0,255), 2)
                # put number
                count+=1
                squares.append(approx)
                cv2.putText(res2, str(count), (points[0][0],points[0][1]), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3, cv2.LINE_AA)
    if ret_num:
        return count
    if plt_res2:
        plt.imshow(res2)
        plt.show()
    if count !=9:
        print("count is not 9 but",count)
        return None
    # print(count)

    
    # contoursのソート
    # 左上の輪郭を取得する
    top_left = None
    # 重心
    centers = []
    # 重心と輪郭の対応
    cen2contours={}

    for cnt in squares:
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        centers.append((cx, cy))
        cen2contours[(cx, cy)]=cnt
    not_visited=set(center for center in centers)

    # 左上の輪郭を取得する
    try:
        top_left = min(centers, key=lambda s: s[0] + s[1])
    except:
        return -1
    not_visited.remove(top_left)
    squares=[[top_left]]
    # 左上の輪郭の中でy軸方向に近い順に2つの輪郭を選択する
    current_pos = top_left
    for i in range(2):
        min_dist = float('inf')
        next_square = None
        for cnt in not_visited:
            x,y=cnt
            dist= np.sqrt((x - current_pos[0])**2 + (y - current_pos[1])**2)
            if dist < min_dist and y > current_pos[1]+5 and x<current_pos[0]+50:
                min_dist = dist
                next_square = cnt
        if next_square is not None:
            squares.append([next_square])
            current_pos = next_square
            not_visited.remove(next_square)
    # 各輪郭に対してx軸方向に近い順に2つの輪郭を選択する
    res3 = result.copy()
    count = 0
    # # print(squares)

    for i in range(3):
        left_most = squares[i][0]
        # print(left_most)
        current_pos = left_most
        for j in range(2):
            min_dist = float('inf')
            next_square = None
            for cnt in not_visited:
                x, y=cnt
                # dist = abs(y - current_pos[1])
                dist= np.sqrt((x - current_pos[0])**2 + (y - current_pos[1])**2)
                if dist < min_dist and x > current_pos[0]+5 and y<current_pos[1]+50:
                    min_dist = dist
                    next_square = cnt
            if next_square is not None:
                squares[i].append(next_square)
                current_pos = next_square
                not_visited.remove(next_square)
    for i in range(3):
        for j in range(3):
            try:
                cx,cy= squares[i][j]
            except IndexError:
                # print(i,j)
                continue
            cnt=cen2contours[(cx,cy)]
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.drawContours(res3, [cnt], -1, (255,0,255), 2)
            count += 1
            cv2.putText(res3, str(count), (x, y+h), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 2, cv2.LINE_AA)
    if plt_res3:
        plt.imshow(res3)
        plt.show()
    problems=[np.ones((9,9)) for _ in range(3)]
    pixel=20
    # white_thres=250

    for i in range(3):
        for j in range(3):
            cnt=cen2contours[squares[i][j]]
            x, y, w, h = cv2.boundingRect(cnt)
            cropped = result[y:y+h, x:x+w]
            # # plt.imshow(cropped)
            # # plt.show()
            # # print(approx)
            cnt=cnt-(x,y)
            src_pts = cnt.reshape((-1, 2)).astype("float32")
            center = np.mean(src_pts, axis=0)
            # ４点を重心からの角度でソートする
            # 時計回りになる
            src_pts = np.array(sorted(src_pts, key=lambda p: np.arctan2(p[1]-center[1], p[0]-center[0])))

            # 結果を表示する
            # # print(src_pts)
            # 縦横比の計算
            w = np.linalg.norm(src_pts[3] - src_pts[0])
            h = np.linalg.norm(src_pts[1] - src_pts[0])
            aspect = abs(w) / abs(h)

            # 新しい画像サイズを設定
            # new_w = int(1000*aspect)
            new_w = 500
            new_h = 500
            # dst_pts = np.array([(0, 0), (0, new_h), (new_w, new_h), (new_w, 0)], dtype="float32")
            dst_pts = np.array([(0, 0), (new_w, 0), (new_w, new_h), (0, new_h)], dtype="float32")

            # 射影変換を計算して、パースをキャンセルする
            warp = cv2.getPerspectiveTransform(src_pts, dst_pts)
            res4 = cv2.warpPerspective(cropped, warp, (new_w, new_h))
            res4=cv2.cvtColor(res4, cv2.COLOR_RGB2GRAY)
            thr, res4 = cv2.threshold(res4, 0, 255, cv2.THRESH_OTSU)
            res4=cv2.resize(res4, (pixel*3,pixel*3), interpolation=cv2.INTER_AREA)
          
            for k in range(3):
                for l in range(3):
                    digit_square = res4[k*pixel:(k+1)*pixel,l*pixel:(l+1)*pixel]
                    problem_i=i*3+k
                    problem_j=j*3+l
                    for m in range(3):
                        # 回転させる
                        digit_square_rot=np.rot90(digit_square,m-1)
                        if m==2:
                            t_i=9-problem_j-1
                            t_j=problem_i
                        elif m==0:
                            t_i=problem_j
                            t_j=9-problem_i-1
                        else:
                            t_i=problem_i
                            t_j=problem_j
                        if np.mean(digit_square_rot)>=white_thres:
                            problems[m][t_i][t_j]=0
                            continue
                        digit_square_rot = digit_square_rot.reshape(1, -1)/255.0
                        digit_square_rot=scaler.transform(digit_square_rot)
                        prediction = clf.predict(digit_square_rot)
                        # predicted_digit = np.argmax(prediction)
                        problems[m][t_i][t_j]=prediction[0]
            # if j%9==3:
            #     # plt.imshow(res4)
            #     # plt.show()
    violations=[]
    for i in range(3):
        violations.append(count_violations(problems[i]))
    # violationの最小のものを返す
    problem=problems[np.argmin(violations)]
    # print(problem,violations[np.argmin(violations)])
    return np.array(problem,dtype=np.int32)
