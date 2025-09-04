import cv2
import mediapipe as mp
import numpy as np
import time

# TIPAGEM ========================
confidence = float
webcam_image = np.ndarray
rgb_tuple = tuple[int, int, int]

# CLASSE ========================
class Detector:
    def __init__(self,
                 mode: bool = False,
                 number_hands: int = 2,
                 model_complexity: int = 1,
                 min_detection_confidence: confidence = 0.5,
                 min_tracking_confidence: confidence = 0.5):
        #Parametros necessarios para inicializar o hands
        self.mode = mode
        self.max_number_hands = number_hands
        self.complexity = model_complexity
        self.detection_con = min_detection_confidence
        self.tracking_con = min_tracking_confidence

        # Inicializar o Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode,
                                         self.max_number_hands,
                                         self.complexity,
                                         self.detection_con,
                                         self.tracking_con)
        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]

    def find_hands(self, img: webcam_image, draw_hands: bool = True):
        # Correção de cor
        img_RBG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Coletar resultados do processo das hands e analisar
        self.results = self.hands.process(img_RBG)

        if self.results.multi_hand_landmarks and draw_hands:
            for hand in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand, self.mp_hands.HAND_CONNECTIONS)

        return img

    def find_position(self, img: webcam_image, hand_number: int = 0):
        self.required_landmark_list = []

        if self.results.multi_hand_landmarks:
            height, width, _ = img.shape
            my_hand = self.results. multi_hand_landmarks[hand_number]
            for id, lm in enumerate(my_hand.landmark):
                center_x, center_y = int(lm.x * width), int(lm.y * height)

                self.required_landmark_list.append([id, center_x, center_y])

        return self.required_landmark_list

# Teste de classe ====================
if __name__ == "__main__":
    Detect = Detector()

    # Captura da imagem
    capture = cv2.VideoCapture(0)

    while True:
        # Captura do frame
        _, img = capture.read()

        # Manipulação de frame
        img = Detect.find_hands(img)
        landmark_list = Detect.find_position(img)
        if landmark_list:
            print(landmark_list[8])

        # Mostrando o frame
        cv2.imshow("Image", img)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
