import cv2

cap = cv2.VideoCapture(0)  # try 0 first

if not cap.isOpened():
    print("❌ Could not open camera")
    exit()

print("✅ Camera opened successfully")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
