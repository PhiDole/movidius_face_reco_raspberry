from gpiozero import Button
from time import sleep 

button1 = Button(14)
button2 = Button(15)

while True:
	if button1.is_pressed:
		print("button 1")

	if button2.is_pressed:
		print("button 2 ")


