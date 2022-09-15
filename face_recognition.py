import numpy as np
import cv2

def face_recognition():
    cap = cv2.VideoCapture(0)  # 0 - делает запрос к веб-камере, иначе указываем путь до файла
    cap.set(3, 500)  # устанавливаем ширину
    cap.set(4, 500)  # устанавливаем высоту
    faces = cv2.CascadeClassifier('faces.xml')  # СascadeClassifier берет файл и вытягивает этот файл как натренированную модель. В faces хранится модель

    while True:
        success, image = cap.read()  # читаем видео
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = faces.detectMultiScale(gray_image, scaleFactor=10, minNeighbors=4)  # за счет detectMultiScale находим координаты всех найденных объектов. scaleFactor означает, что мы можем находить лица в два раза больше чем те, на которых была натренирована наша модель

        for (x, y, w, h) in results:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)
            cv2.putText(image, 'Face detected', (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), thickness=2)

        cv2.imshow('Result', image)

        if cv2.waitKey(1) and 0xFF == ord('q'):  # каждая картинка в видео отображается одну миллисекунду, далее другая картинка
            break  # если нажимаем q, то выходим из видео




if __name__ == '__main__':
    face_recognition()
