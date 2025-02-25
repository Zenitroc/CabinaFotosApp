import cv2
import time
import numpy as np

def overlay_logo(background, logo, pos):
    x, y = pos
    h_logo, w_logo = logo.shape[:2]

    if y + h_logo > background.shape[0]:
        h_logo = background.shape[0] - y
        logo = logo[:h_logo, :]
    if x + w_logo > background.shape[1]:
        w_logo = background.shape[1] - x
        logo = logo[:, :w_logo]

    roi = background[y:y+h_logo, x:x+w_logo]

    if logo.shape[2] == 4:
        logo_bgr = logo[:, :, :3]
        alpha = logo[:, :, 3] / 255.0
        for c in range(3):
            roi[:, :, c] = (alpha * logo_bgr[:, :, c] + (1 - alpha) * roi[:, :, c]).astype(np.uint8)
    else:
        roi[:, :, :] = logo

def add_black_bars(frame, bar_height_ratio=0.05):
    h, w = frame.shape[:2]
    bar_height = int(h * bar_height_ratio)
    black_bar = np.zeros((bar_height, w, 3), dtype=np.uint8)
    frame_with_bars = np.vstack([black_bar, frame, black_bar])
    return frame_with_bars

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    cv2.namedWindow("Webcam")
    foto_contador = 1

    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar el frame inicial.")
        return

    logo = cv2.imread("logo.png", cv2.IMREAD_UNCHANGED)
    if logo is None:
        print("No se pudo cargar el logo.")
        return

    scale_factor = 0.1  
    new_width = int(logo.shape[1] * scale_factor)
    new_height = int(logo.shape[0] * scale_factor)
    logo = cv2.resize(logo, (new_width, new_height), interpolation=cv2.INTER_AREA)

    capturando = False
    fotos_restantes = 0
    next_capture_time = 0
    flash_end_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo capturar el frame.")
            break

        if logo is not None:
            pos = (10, frame.shape[0] - logo.shape[0] - 10)
            overlay_logo(frame, logo, pos)

        frame = add_black_bars(frame)  # Agregar las franjas negras arriba y abajo

        if capturando:
            remaining_time = next_capture_time - time.time()
            seconds_left = int(np.ceil(remaining_time))
            if seconds_left > 0:
                text = str(seconds_left)
                font = cv2.FONT_HERSHEY_DUPLEX
                base_scale = 3
                max_scale = 5
                thickness = 8
                elapsed_time = 4 - remaining_time  

                scale_factor = base_scale + (max_scale - base_scale) * (elapsed_time / 4)
                text_size = cv2.getTextSize(text, font, scale_factor, thickness)[0]
                x_text = (frame.shape[1] - text_size[0]) // 2
                y_text = (frame.shape[0] + text_size[1]) // 2

                overlay = frame.copy()
                cv2.putText(overlay, text, (x_text, y_text), font, scale_factor, (255, 255, 255), thickness, cv2.LINE_AA)
                alpha = max(0, min(1, remaining_time / 4))
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            else:
                nombre_archivo = f"foto_{foto_contador}.jpg"
                cv2.imwrite(nombre_archivo, frame)
                print(f"Foto {foto_contador} guardada como {nombre_archivo}")
                foto_contador += 1
                fotos_restantes -= 1
                flash_end_time = time.time() + 0.2

                if fotos_restantes > 0:
                    next_capture_time = time.time() + 4
                else:
                    capturando = False

        if time.time() < flash_end_time:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (255, 255, 255), -1)
            alpha = 0.4
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        cv2.imshow("Webcam", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  
            break
        if key in [10, 13] and not capturando:
            print("Iniciando la captura de 5 fotos con cuenta regresiva y logo...")
            capturando = True
            fotos_restantes = 5
            next_capture_time = time.time() + 4

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
