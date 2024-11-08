import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

def recognize(path,clf=None,scaler=None,pixel=20,ret_img=False,n_open=3,n_close=0,prior_close=False,trim_percentage=0.007,mean_white_axis=0,arc_epsilon=5e-2,erase_line=1,white_thres=255,otsu_times=1.22,clf_f_name="SVClinear",clf_f=None,scaler_f=None,sigmaColor=2,sigmaSpace=2,pixel_f=120,detect_rotate=1):

    try:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
    except Exception as e:
        print("cannot read image",e)
        return
    if pixel is None:
        pixel=60
    if scaler is None:
            # scaler = pd.read_pickle('./models/Rand_numbers_mix_l2_scaler.pickle')
            scaler = pd.read_pickle('./models/MLPC_numbers_mix_scaler.pickle')
    if clf is None:
        # clf = pd.read_pickle('./models/Rand_numbers_mix_l2_clf.pickle')
        clf=pd.read_pickle('./models/MLPC_numbers_mix_clf.pickle')
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 外縁を切る
    trim_percentage=0.001
    height, width, channels = image.shape[:3]
    trim_width = int(width * trim_percentage)
    trim_height = int(height * trim_percentage)
    image = image[trim_height:height - trim_height, trim_width:width - trim_width]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # print(image.shape)
    # ぼかし処理
    # gray_gb = cv2.GaussianBlur(gray, None, 3.0)
    gray_gb= cv2.bilateralFilter(gray, 11, sigmaColor, sigmaSpace)
    ##########################
    ## エッジ検出、輪郭抽出
    ############################
    thr, binary = cv2.threshold(gray_gb, 0, 255, cv2.THRESH_OTSU)

    ########################################
    ## level 2ではこれをすると死ぬ
    #######################################
    new_thr = min(int(thr * otsu_times), 255)
    _, binary = cv2.threshold(gray_gb, new_thr, 255, cv2.THRESH_BINARY)

    # # plt.imshow(binary, cmap="gray")
    # plt.title("Otsu's binarization (threshold={:d})".format(int(thr)))
    # # plt.show()

    edge = cv2.Canny(binary, 100, 200)
    edge = cv2.dilate(edge, np.ones((5, 5), dtype=edge.dtype),iterations=1)
    edge = cv2.erode(edge, np.ones((3, 3), dtype=edge.dtype),iterations=1)
    # res_close = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), dtype=binary.dtype))
    # # plt.imshow(edge, cmap="gray")
    # plt.title("After morphology operation".format(thr))
    # # plt.show()
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = image.copy()
    cv2.drawContours(result, contours, -1, (255, 0, 0), 3, cv2.LINE_AA)
    # # plt.imshow(result)
    # # plt.show()
    longest_cnt = None
    max_area = 0.0
    result = image.copy()
    for cnt in contours:
        # 輪郭線の長さを計算
        arclen = cv2.arcLength(cnt, True)
        approx=cv2.approxPolyDP(cnt, arclen * arc_epsilon, True)
        area=cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(approx)
        internal_area_ratio = area / (w * h)
        if len(approx) == 4 and internal_area_ratio>0.5:
            # print(len(approx), x,y,w,h, arclen,internal_area_ratio)
            cv2.drawContours(result, [approx], -1, (255, 0, 0), 3, cv2.LINE_AA)
            # # plt.imshow(result)
            # # plt.show()
            if  max_area < area:
                max_area = area
                longest_cnt = cnt
                    

    result = image.copy()

    # # plt.imshow(result)
    # # plt.show()
    # print(len(longest_cnt))
    arclen = cv2.arcLength(longest_cnt, True)
    approx = cv2.approxPolyDP(longest_cnt, arclen * arc_epsilon, True)

    cv2.drawContours(result, [approx], -1, (255, 0, 0), 3, cv2.LINE_AA)
    # plt.imshow(result)
    # plt.title("Red region has {:d} corners".format(len(approx)))
    # plt.show()
    # len(contours[0])
    
    x, y, w, h = cv2.boundingRect(approx)
    cropped = image[y:y+h, x:x+w]
    # plt.imshow(cropped)
    # plt.show()
    # print(approx)
    approx=approx-(x,y)
    src_pts = approx.reshape((-1, 2)).astype("float32")
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
    # new_w = 1000
    new_h = 1000
    # dst_pts = np.array([(0, 0), (0, new_h), (new_w, new_h), (new_w, 0)], dtype="float32")
    dst_pts = np.array([(0, 0), (new_w, 0), (new_w, new_h), (0, new_h)], dtype="float32")

    # 射影変換を計算して、パースをキャンセルする
    warp = cv2.getPerspectiveTransform(src_pts, dst_pts)
    result = cv2.warpPerspective(cropped, warp, (new_w, new_h))
    # print(result.shape)
    # result = cv2.warpPerspective(binary, warp, (new_w, new_h))

    #　反転させたリストを作成

    # for i, result in enumerate(results):

        # plt.subplot(2, 2, i+1)
        # # plt.imshow(result)
        # plt.title("result {:d}".format(i))

    # plt.imshow(result, cmap="gray")
    # plt.show()
    # print(result.shape)
    gray_result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    thresh, binary_image = cv2.threshold(gray_result, 0, 255, cv2.THRESH_OTSU)
    # 白い領域の平均値を計算
    white_region = result.copy()
    idx=binary_image!=0
    # mean_white = np.median(white_region[idx],axis=0)
    # 赤にするとなぜかうまくいく, 0の閾値の関係で、0ありで訓練したモデルを使っているとき
    # mean_white = np.mean(white_region[idx],axis=None)
    mean_white = np.mean(white_region[idx],axis=mean_white_axis)
    
    # print(mean_white)
    # plt.imshow(binary_image, cmap="gray")
    # plt.imshow(binary_image!=0, cmap="gray")
    # white

    height, width, channels = result.shape[:3]
    trim_width = int(width * trim_percentage)
    trim_height = int(height * trim_percentage)
    result = result[trim_height:height - trim_height, trim_width:width - trim_width]
    # plt.imshow(result)
    if erase_line:
        gray=cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        # Threshold the image to create a binary image
        # Smooth the image
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect edges using Canny edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect lines using Hough line detection
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

        # Plot the detected lines on the image
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(result, (x1, y1), (x2, y2), (mean_white), 15)

        # Display the image with the detected lines
        # plt.imshow(result, cmap="gray")
        # plt.show()
        # Display the result
        # Load the image
        image = result

        # Define the size of the lines
        line_size = int(min(image.shape[:2]) / 9)

        # Define the spacing between the lines
        line_spacing = line_size
        # print(line_size, line_spacing)
        # Create a copy of the image to draw the lines on
        result = image.copy()

        # Draw the vertical lines
        for x in range(0, image.shape[1], line_spacing):
            cv2.line(result, (x, 0), (x, image.shape[0]), (mean_white), thickness=25)

        # Draw the horizontal lines
        for y in range(0, image.shape[0], line_spacing):
            cv2.line(result, (0, y), (image.shape[1], y), (mean_white), thickness=25)

    # Display the image with the lines
    # plt.imshow(result, cmap="gray")
    # plt.show()

    # scaler_f=pd.read_pickle("./pickle/knn_scaler_flip.pickle")
    # clf_f=pd.read_pickle("./pickle/knn_clf_flip.pickle")

    # scaler_f=pd.read_pickle("./pickle/rf_flip_scaler.pickle")
    # clf_f=pd.read_pickle("./pickle/rf_clf_flip.pickle")

    #########################################
    ## classify flipped
    #########################################
    if detect_rotate:
        if clf_f is None:
            scaler_f = pd.read_pickle(f'./pickle/{clf_f_name}_flip_scaler.pickle')
            clf_f=pd.read_pickle(f'./pickle/{clf_f_name}_flip_clf.pickle')

        results=[result,np.rot90(result,1),np.rot90(result,3)]
        if pixel_f is None:
            pixel_f=200
        proba=[]
        for res in results:
            res=cv2.resize(res,(pixel_f,pixel_f),interpolation=cv2.INTER_AREA)
            res_gr=cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
            

            try:
                prob=clf_f.predict_proba(scaler_f.transform(res_gr.reshape(1,-1)/255.0))
            # print(prob)
                proba.append(prob[0][1])
            except:
                prob=clf_f.predict(scaler_f.transform(res_gr.reshape(1,-1)/255.0))
                proba.append(prob[0])
                # print(prob)
        res_idx=np.argmax(proba)
        result=results[res_idx]
    if ret_img:
        return result
    

    cropped = result.copy()
    h,w,_=cropped.shape
    cropped_rs=cv2.resize(cropped,(max(h,w),max(h,w)),interpolation=cv2.INTER_AREA)
    cropped_gr=cv2.cvtColor(result, cv2.COLOR_RGB2GRAY) # 白地


    # # plt.imshow(cropped_rs)
    # # plt.show()
    # 内側の細い線を塗りつぶす
    if prior_close:
        # closing
        cropped_cl = cv2.dilate(cropped_rs, np.ones((2, 2), dtype=edge.dtype),iterations=n_close)
        cropped_cl= cv2.erode(cropped_cl, np.ones((2, 2), dtype=edge.dtype),iterations=n_close)
        # opening
        cropped_cl = cv2.erode(cropped_rs, np.ones((2, 2), dtype=edge.dtype),iterations=n_open)
        cropped_cl = cv2.dilate(cropped_cl, np.ones((2, 2), dtype=edge.dtype),iterations=n_open)
    else:
        # opening
        cropped_cl = cv2.erode(cropped_rs, np.ones((2, 2), dtype=edge.dtype),iterations=n_open)
        cropped_cl = cv2.dilate(cropped_cl, np.ones((2, 2), dtype=edge.dtype),iterations=n_open)
        # closing
        cropped_cl = cv2.dilate(cropped_cl, np.ones((2, 2), dtype=edge.dtype),iterations=n_close)
        cropped_cl= cv2.erode(cropped_cl, np.ones((2, 2), dtype=edge.dtype),iterations=n_close)
        
    # # plt.imshow(cropped_cl,cmap="gray")
    # # plt.show()

    cropped_cl_gr=cv2.cvtColor(cropped_cl, cv2.COLOR_RGB2GRAY)
    thr, binary = cv2.threshold(cropped_cl_gr, 0, 255, cv2.THRESH_OTSU)
    # new_thr = min(int(thr * 1.3), 255)
    # _, binary = cv2.threshold(cropped_region_gr, new_thr, 255, cv2.THRESH_BINARY)

    binary=cv2.resize(binary,(pixel*9,pixel*9),interpolation=cv2.INTER_AREA)

    # binary=cv2.medianBlur(binary,3)
    # plot i,j th component
    # i,j=0,3
    # # plt.imshow(binary[i*pixel:(i+1)*pixel, j*pixel:(j+1)*pixel], cmap='gray')

    # # plt.show()
    # results=[result,np.rot90(result,1),np.rot90(result,3)]
    # pixel_f=200
    # stds=[]
    # for res in results:
    #     res=cv2.resize(res,(pixel_f,pixel_f),interpolation=cv2.INTER_AREA)
    predicted_numbers = []
    for i in range(9):
        for j in range(9):
            digit_square = binary[i*pixel:(i+1)*pixel, j*pixel:(j+1)*pixel]
            if np.mean(digit_square)>=white_thres:
                predicted_numbers.append(0)
                continue
            digit_square = digit_square.reshape(1, -1)/255.0
            digit_square=scaler.transform(digit_square)
            prediction = clf.predict(digit_square)
            predicted_digit = np.argmax(prediction)
            predicted_numbers.append(prediction[0])
    problem=[]
    for i in range(0, len(predicted_numbers), 9):
        problem.append(predicted_numbers[i:i+9])
        # print(predicted_numbers[i:i+9])

    # stdが低いものがよさそう
    # for problem in problems:
    #     nonzeros=np.count_nonzero(problem)
    #     print(nonzeros,np.std(problem),np.mean(problem))
    # problem=problems[0]
    return problem
