import cv2
import numpy as np

def isWhiteRegion(roi, white_threshold=200, white_pixel_percentage=0.7):
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
    total_pixels = gray_roi.size  # 区域内的像素总数

    # 白色像素的数量，需将布尔值转换为uint8类型
    white_pixels = cv2.countNonZero((gray_roi > white_threshold).astype(np.uint8))

    # 计算白色像素占比
    white_pixel_ratio = white_pixels / total_pixels

    # 如果白色像素的比例超过设定阈值，则认为是白色区域
    return white_pixel_ratio >= white_pixel_percentage

def isTouchingBorder(contour, image_shape):
    # 获取图像的尺寸
    img_height, img_width = image_shape[:2]
    
    # 计算轮廓的边界矩形
    x, y, w, h = cv2.boundingRect(contour)
    
    # 检查轮廓是否与图像边界相接触
    return x <= 0 or y <= 0 or (x + w) >= img_width or (y + h) >= img_height

def extractWhiteRegions(image, block_size=50, white_threshold=200, white_pixel_percentage=0.7, min_area=1000):
    height, width = image.shape[:2]
    white_region_mask = np.zeros((height, width), dtype=np.uint8)  # 创建一个空掩膜

    # 遍历图像，将其划分为大小为 block_size x block_size 的块
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            # 确保不超出图像边界
            block = image[y:y + block_size, x:x + block_size]

            # 检查该块是否为白色区域
            if isWhiteRegion(block, white_threshold, white_pixel_percentage):
                # 如果是白色区域，掩膜中该区域设为白色
                white_region_mask[y:y + block_size, x:x + block_size] = 255

    # 找到白色区域的轮廓
    contours, _ = cv2.findContours(white_region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个新的空图像来保存大于最小面积的白色区域
    filtered_white_regions = np.zeros_like(image)

    for contour in contours:
        # 计算轮廓的面积
        area = cv2.contourArea(contour)
        
        # 过滤小于最小面积的区域以及与边界相连接的区域
        if area >= min_area and not isTouchingBorder(contour, image.shape):
            cv2.drawContours(filtered_white_regions, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    return filtered_white_regions

def main(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    image = cv2.resize(image, (1000, 800))

    # 提取白色区域
    filtered_white_regions = extractWhiteRegions(image, block_size=3, white_threshold=120, white_pixel_percentage=0.1, min_area=50000)

    # 显示结果
    cv2.imshow('Original Image', image)
    cv2.imshow('Filtered White Regions', filtered_white_regions)

    # 将过滤后的白色区域应用到原图中
    result_image = cv2.bitwise_and(image, image, mask=cv2.cvtColor(filtered_white_regions, cv2.COLOR_BGR2GRAY))

    # 保存结果图像
    cv2.imwrite('result.png', result_image)

    # 显示保存的结果图像
    cv2.imshow('result image', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main('eye_image.jpg')
