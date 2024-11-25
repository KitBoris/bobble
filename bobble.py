import cv2
import numpy as np
import matplotlib.pyplot as plt


def fail(frame_data: List[int]) -> None:
    "Сохранение в файл"
    with open('frame_data.txt', 'w') as f:
        for frame_info in frame_data:
            frame_num = frame_info[0]
            center_x = frame_info[1]
            center_y = frame_info[2]
            width = frame_info[3]
            height = frame_info[4]
            f.write(f"{frame_num}\t\t{center_x}\t\t{center_y}\t\t{width}\t\t{height}\n")

def chart(frame_data: int, frame_width: int, frame_height: int) -> None:
    "Выводим траекторию движения"
    all_x = []
    all_y = []
    for _, cx, cy, _, _ in frame_data:
        all_x.append(cx)
        all_y.append(cy)

    plt.figure(figsize=(10, 6))
    plt.scatter(all_x, all_y, s=1)
    plt.xlabel('Ширина кадра')
    plt.ylabel('Высота кадра')
    plt.xlim(0, frame_width)
    plt.ylim(frame_height, 0)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def video(cap: cv2.VideoCapture,min_w_h: int, max_w_h: int, frame_data: int) -> List[int]:
    "Обработка видео"
    frame_number :int = 0 # номер кадра
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(blurred)
        edges = cv2.Canny(cl1, 100, 200)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_image = frame.copy()

        for i in range(len(contours)):
            if hierarchy[0][i][3] == -1:
                area = cv2.contourArea(contours[i])
                x, y, w, h = cv2.boundingRect(contours[i])
                if min_w_h <= w <= max_w_h and min_w_h <= h <= max_w_h:
                    center_x = x + w // 2
                    center_y = y + h // 2

                    if area < 600:
                        a = int((w * h * 3) / (4.15))
                        area_text = f"{a} pxp"
                    else:
                        area_text = f"{int(area)} px"
                    cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(contour_image, area_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 1)

                    frame_data.append((frame_number, center_x, center_y, w, h))

        contour_image = cv2.resize(contour_image, None, fx=0.7, fy=0.7)
        cv2.imshow('Video', contour_image)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    return frame_data

if __name__ == "__main__":
    cap = cv2.VideoCapture('0%_3_ml_min_200_mm_35_gr.MOV') # Загружаем видео
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # Выделяем ширину кадра для постраения графика
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Выделяем высоту кадра для постраения графика
    min_w_h : int = 25  # Минимальный размер объекта (максимальная высота и ширина объекта)
    max_w_h :int = 85  # Максимальный размер объекта (максимальная высота и ширина объекта)
    frame_data :int = [] # Список для хранения данных с ролика

    frame_data1= video(cap,min_w_h, max_w_h, frame_data)
    fail(frame_data1)
    chart(frame_data1,frame_width, frame_height)