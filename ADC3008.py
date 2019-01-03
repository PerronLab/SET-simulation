# ADC3008 

import RPi.GPIO as IO
import time
#import matplotlib.pyplot as plt

#Define GPIO Pins for (CS, CLK, MISO & MOSI)
SPICLK  = 2
SPIMISO = 3 # Dout
SPIMOSI = 4 # Din
SPICS   = 17

#Set SPI pins as Input/Outout
def setup():
	IO.setmode(IO.BCM)
	IO.setwarnings(False)
	IO.setup(SPICLK,IO.OUT)
	IO.setup(SPIMISO, IO.IN)
	IO.setup(SPIMOSI,IO.OUT)
	IO.setup(SPICS,IO.OUT, initial = IO.HIGH)

def read(channel):
	#Determine if allowed channel
	if channel > 7 | channel < 0:
		print("error")
		return -1


	#Initialize communication when CS is low
	IO.output(SPICLK,IO.LOW)
	IO.output(SPICS,IO.LOW)

	control  = channel
	if channel == 0:
		control |=0b00011000

	elif channel == 1:
		control |=0b00011001
	elif channel == 2:
		control |=0b00011010
	elif channel == 3:
		control |=0b00011011
	elif channel == 4:
		control |=0b00011100
	elif channel == 5:
		control |=0b00011101
	elif channel == 6:
		control |=0b00011110
	elif channel == 7:
		control |=0b00011111

	# intialize 5 bits before data output
	#print(control)
	for i in range(5):
		if control & 0x10:
			#print("High\n")
			IO.output(SPIMOSI, IO.HIGH)
		else:
			#print("Low\n")
			IO.output(SPIMOSI, IO.LOW)
		control <<= 1
		#print(control)
		IO.output(SPICLK,IO.HIGH)
		IO.output(SPICLK,IO.LOW)
	# 10 bit read w/ null @ beginning
	char = 0
	for i in range(11):
		IO.output(SPICLK,IO.HIGH)
		IO.output(SPICLK,IO.LOW)
		char <<=1
		if IO.input(SPIMISO):
			char |= 0x1
	# stop communication
	IO.output(SPICS,IO.HIGH)
	return char

def mapVoltage(value, bitMin,bitMax,voltMin,voltMax):
	bitTotal  = (bitMax+1)  - bitMin
	voltTotal = voltMax - voltMin
	valueScaled = float(value-bitMin)/float(bitTotal)
	return voltMin + (valueScaled*voltTotal)


def roundValue(value):
	return ("{0:.2f}".format(value))


def saveFile(time,gen,pv):
	with open('data.txt','a') as file:
		file.write(str(time))
		file.write(',')
		file.write(str(gen))
		file.write(',')
		file.write(str(pv))
		#file.write(',')
		#file.write(str(data2))
		#file.write(',')
		#file.write(str(data3))
		file.write("\n")


setup()
vgain = 4.138
v_in = 119.8
v_out = 3.331
vgen = v_in/v_out
vpA = .066
gain =3/2
offset = 483
#577

#voltage peak to peak: 118.3
Vcc = 5
fudge = 8.28
while 1:
	char0 = read(0)-offset
	#char1 = read(0)
	#char2 = (read(2)-offset)
	char3 = read(3)
	data0 = mapVoltage(char0,0,1023,0,3.3)*(gain/vpA) #current measurement (Generator)
	#data1 = mapVoltage(char1,0,1023,0,3.3)*vgen #Voltage measurement (Solar Panel)
	#data2 = mapVoltage(char2,0,1023,0,3.3)/vpA #Current measurement (Solar Panel)
	data3 = mapVoltage(char3,0,1023,0,3.3)*(12/2.98) #voltage measurement (Generator)
	power = data0*data3
	print(time.strftime("%Y-%m-%d %H:%M:%S"),"|",roundValue(data3),"V", roundValue(data0), "I",roundValue(power),"W")
	#saveFile(time.strftime("%Y-%m-%d %H:%M:%S"),roundValue(data1), roundValue(data2))
	print(char0)	
	time.sleep(1)
