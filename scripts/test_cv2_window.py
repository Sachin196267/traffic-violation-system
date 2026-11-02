import cv2
import numpy as np

img = np.zeros((300, 500, 3), dtype=np.uint8)
cv2.putText(img, "Click this window, then press any key", (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

cv2.namedWindow("test")
cv2.imshow("test", img)

print("Window shown. Waiting for a keypress (should NOT print until you press a key)...")
key = cv2.waitKey(0)  # 0 = wait forever, no timeout
print(f"Got key: {key}")

cv2.destroyAllWindows()
