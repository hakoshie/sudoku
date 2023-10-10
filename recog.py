import cv2
import numpy as np
import matplotlib.pyplot as plt


def recognize(path="./data/level_1.jpg"):
    try:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
    except:
        print("Error: reading image")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # ぼかし処理
    gray_gb = cv2.GaussianBlur(gray, None, 3.0)

    # 大津の二値化
    thr, binary = cv2.threshold(gray_gb, 0, 255, cv2.THRESH_OTSU)
    edge = cv2.Canny(binary, 100, 200)
    edge = cv2.dilate(edge, np.ones((11, 11), dtype=edge.dtype))
    edge = cv2.erode(edge, np.ones((9, 9), dtype=edge.dtype))

    # 外側のcontoursを取得
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # result = image.copy()
    # cv2.drawContours(result, contours, -1, (255, 0, 0), 3, cv2.LINE_AA)

    # 内側の細い線を塗りつぶす
    # # closing
    image_closed = cv2.dilate(image, np.ones((2, 2), dtype=edge.dtype),iterations=4)
    image_closed = cv2.erode(image_closed, np.ones((2, 2), dtype=edge.dtype),iterations=4)
    # opening
    # tmp = cv2.erode(cropped_region, np.ones((1, 1), dtype=edge.dtype),iterations=2)
    # tmp = cv2.dilate(tmp, np.ones((1, 1), dtype=edge.dtype))


    x, y, w, h = cv2.boundingRect(contours[0])
    # 画像から領域を切り出す
    cropped_region = image_closed[y:y+h, x:x+w].copy()


    # plt.imshow(cropped_region,cmap="gray")
    # cropped_region.shape
    
    # 内側の太いcontoursを黒で塗りつぶすために検出
    edge = cv2.Canny(cropped_region, 100, 200)
    edge = cv2.dilate(edge, np.ones((11, 11), dtype=edge.dtype))
    edge = cv2.erode(edge, np.ones((9, 9), dtype=edge.dtype))
    contours, hierarchy= cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    draws=[]
    for i,contour in enumerate(contours):
        #親のindexが1以下のものを描画して塗りつぶすためにdrawsに追加
        if hierarchy[0,i][3]<=1:
            draws.append(contour)
    cv2.drawContours(cropped_region, draws, -1, (255,255,255), 15, cv2.FILLED)
    cropped_region=cv2.bitwise_not(cropped_region)
    # 枠の分をトリミング
    height, width, _ = cropped_region.shape
    trim_percentage = 0.01  # 外側を1%トリミング
    trim_width = int(width * trim_percentage)
    trim_height = int(height * trim_percentage)
    cropped_region = cropped_region[trim_height:height-trim_height, trim_width:width-trim_width]

    cropped_region=cv2.cvtColor(cropped_region, cv2.COLOR_RGB2GRAY)
    # plt.imshow(cropped_region)
    # plt.show()

    return cropped_region
