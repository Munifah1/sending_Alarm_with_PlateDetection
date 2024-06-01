import cv2
import pytesseract
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

# Specify the path to the tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path based on your Tesseract installation

frameWidth = 640
frameHeight = 480
minArea = 250
color = (255, 0, 255)
count = 0

# Email configuration
sender_email = "sender@gmail.com"
receiver_email = "receiver_email@gmail.com"  # Replace with the recipient's email address
password = os.getenv('EMAIL_PASSWORD_generte key password from gmail app password')  # Retrieve the password from environment variables
subject = "Alarm Notification"

# Ensure the password is retrieved successfully
if not password:
    raise ValueError("No password set for EMAIL_PASSWORD environment variable")

# Load the pre-trained classifier
nPlateCascade = cv2.CascadeClassifier("Resources/haarcascade_russian_plate_number.xml")
if nPlateCascade.empty():
    print("Error loading cascade classifier.")
    exit()

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)  # Set width
cap.set(4, frameHeight) # Set height
cap.set(10, 150)        # Set brightness

def send_email(plate_text):
    # Create a MIMEText object
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    # Email body
    body = f"This is an alarm notification! Detected Number Plate: {plate_text}"
    message.attach(MIMEText(body, "plain"))

    # Establish a connection to the SMTP server over SSL
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, password)  # Login to the email server
            server.sendmail(sender_email, receiver_email, message.as_string())  # Send email
            print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect number plates
    vehicleNumberPlates = nPlateCascade.detectMultiScale(imgGray, 1.1, 10)

    for (x, y, w, h) in vehicleNumberPlates:
        area = w * h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, "Vehicle number Plate", (x, y - 5),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
            imgRoi = img[y:y + h, x:x + w]
            cv2.imshow("ROI", imgRoi)

            # Use OCR to read text from the ROI
            plate_text = pytesseract.image_to_string(imgRoi, config='--psm 8')
            print("Detected Number Plate:", plate_text)

            # Send email with the detected number plate
            send_email(plate_text)

    cv2.imshow("Detecting", img)

    # Save the detected plate region
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite(f"Resources/Scanned/noVehicleNumberPlate{count}.jpg", imgRoi)
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Scan Saved", (150, 265), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 2)
        cv2.imshow("Vehicle number Plate Detection", img)
        cv2.waitKey(500)
        count += 1

    # Exit the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()