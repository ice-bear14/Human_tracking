from flask import Flask, request, jsonify
import time
import RPi.GPIO as GPIO

#set GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

#Set Pin Servo
SERVO_X_PIN = 24 #HORIZONTAL
SERVO_Y_PIN = 13 #VERTIKAL

GPIO.setup(SERVO_X_PIN, GPIO.OUT)
GPIO.setup(SERVO_Y_PIN, GPIO.OUT)

#Set pwm 50hz
servo_x = GPIO.PWM(SERVO_X_PIN, 50)
servo_y = GPIO.PWM(SERVO_Y_PIN, 50)

servo_x.start(0)
servo_y.start(0)

#Konversi sudut
def angel_to_duty(angel):
    return 2 + (angel / 18.0)

@app.route("/pwm", methods=["POST"])
def receive_pwm():
    data = request.get_json()
    try:
        pwm_x = int(data.get("pwm_x", 90))
        pwm_y = int(data.get("pwm_y", 90))
        
        #Batas Sudut 0-180
        pwm_x = max(0, min(180, pwm_x))
        pwm_y = max(0, min(180, pwm_y))
        
        duty_x = angel_to_duty(pwm_x)
        duty_y = angel_to_duty(pwm_y)
        
        print(f"Servo X: {pwm_x} Derajat ({duty_x:.2f}%), Servo Y: {pwm_y} Derajat ({duty_y:.2f}%)")
        
        #PWM ke Servo
        servo_x.ChangeDutyCycle(duty_x)
        servo_y.ChangeDutyCycle(duty_y)
        
        time.sleep(0.3)
        
        #Matikan Sinyal PWM agar tidak jitter
        servo_x.ChangeDutyCycle(0)
        servo_y.ChangeDutyCycle(0)
        
        return jsonify({"status": "OK", "servo_x": pwm_x, "servo_y": pwm_y})
    
    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"status": "error", "message": ster(e)}), 400
    
if __name__ == "__main__":
    try:
        print("== Raspberry Pi Servo Receiver ==")
        print("Listening on http://0.0.0.0:5000/pwm ...")
        app.run(host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("Dihentikan Manual.")
    finally:
        servo_x.stop()
        servo_y.stop()
        GPIO.cleanup()