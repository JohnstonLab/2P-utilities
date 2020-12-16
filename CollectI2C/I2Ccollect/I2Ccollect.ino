
///////i2c communication
//SCB-19: PFI0: SDA   PFI1: SCL
//mega : data: 20, clock : 21

//arduino: red: A4   black: A5


//COMMENT MOTION_SENSOR IF NOT USING THE OPTICAL MICE
//COMMENT _DEBUG_ IF NOT DEBUGGING

#include "Arduino.h"
#include <Wire.h>

//#define _DEBUG_ // debug conditional compiling
//#define MOTION_SENSOR //conditional compiling if using optical mice. COMMENT IF NOT USING THE MICE


/*TO ADD A NEW DATA:
   add the corresponding pin in the code:
      int myDataPin = "my pin number"

   add an acquisition value variable
      int myDataValue;

   add an I2C array to transmit this value.
   This array should have the size of the number of characters you want to send +1.
   i.e. to send 327, the array size is 4, because each char is one byte and we want to transmit bytes.
      int myDataI2cArray["my required size"];

   add the pinMode in Setup()
      pinMode(myDataPin, INPUT);

   in the acquisition part (dataAcquisition() function)
   add your data reading
    if it is a digital value:
      myDataValue = digitalRead(myDataPin)

    if it is an analog value:
      myDataValue = analogRead(myDataPin);


   add the value in plot:
       Serial.print("yourData: "); Serial.print(yourData); Serial.print(" ");
     Make sure that the last line of ArduinoPlotting is Serial.print(\n");

   add the data in the bytes conversion function i2cDataTransform
      ((String)myDataValue).toCharArray(myDataI2cArray, my array size);

   add the i2C communication line inbetween Wire.beginTransmission(0x00); and Wire.endTransmission();
      Wire.write(myDataI2cArray); Wire.write(" ");
*/

// ===============================
// =====         GPIO         ====
// ===============================

//motionSensor
int clockMouse1Pin = 6;
int dataMouse1Pin = 5;
int clockMouse2Pin = 3;
int dataMouse2Pin = 2;
#include "MotionSensorAcq.h"

//analog channel2
int analog2 = A5;

//Lick sensor
int lickPin = 8;
int lickValveActivationPin = 9;
int lickDrainValveActivationPin = 10;

//strain gaugea1
int strainGaugePin = A4;

//camera trigger and stimulus
int syncPin = 15; //
int stimulusPin = 16; //aurora 206 final valve

// swithI2C mode
int I2CSwitch = 21;

// ===============================
// =====  Acquistion Values   ====
// ===============================
int motionSensorValuesArray[3] = {0};
int analog2Value;
int lickValue;
int lickValveActivationValue;
int lickDrainValveActivationValue;
int strainGaugeValue;
int cameraTriggerValue; // == 1 when triggered
int stimulusValue;

// ===============================
// =====         I2C          ====
// ===============================
int DELAY_I2C = 1;
int i2cError = 0;

//I2C message
char strainGaugeI2cArray[5];
char cameraTriggerI2cArray[2];
char stimulusI2cArray[2];
//Motion sensor data
char motionXI2cArray[8];
char motionYI2cArray[8];
char motionZI2cArray[8];
//therm resp
char analog2I2cArray[5]; 
//lick sensor
char lickValueI2cArray[2];
char lickValveActivationI2cArray[2];
char lickDrainValveActivationI2cArray[2];

// the setup routine runs once when you press reset:
void setup() {
  pinMode(20, INPUT_PULLUP);
  pinMode(21, INPUT_PULLUP); //sets i2C bus with internal pullup
  pinMode(syncPin, INPUT);
  pinMode(stimulusPin, INPUT_PULLUP);
  pinMode(analog2, INPUT_PULLUP);
  pinMode(lickPin, INPUT_PULLUP);
  pinMode(lickValveActivationPin, INPUT_PULLUP);
  pinMode(lickDrainValveActivationPin, INPUT_PULLUP);
  pinMode(strainGaugePin, INPUT_PULLUP);
  pinMode(I2CSwitch, INPUT);
  Wire.begin();
  //Wire.setClock(400000); //400kHz i2C freq change. must be left to default (100kHz) for scanImage to handle the data!
  Serial.begin(500000);
#ifdef MOTION_SENSOR
  motionSensorSetup();
#endif
}

void loop() {
  dataAcquisition();
  ArduinoPlot();
  i2cDataTransform();

  ////////////////////////////////////////////////////////////////////
  /*OVERWRITE cameraTriggerValue so constant i2c communication without trigger*/
  ////////////////////////////////////////////////////////////////////
 // cameraTriggerValue = 1;
  if(digitalRead(I2CSwitch) == HIGH){
    cameraTriggerValue = 1;
  }
  if(digitalRead(I2CSwitch) == LOW){
    cameraTriggerValue = 0;
  }

  if (cameraTriggerValue == 1) {
#ifdef _DEBUG_
    Serial.println("I2C ON, cameraTriggerValue == 1");
#endif
    i2cCommunication();

#ifdef _DEBUG_
    Serial.print("i2C error: "); Serial.println(i2cError);
#endif
    delay(DELAY_I2C);
  }
#ifdef _DEBUG_
  else {
    Serial.println("I2C OFF, cameraTriggerValue != 1");
  }
#endif
}

//-----------------------------------------------------------------------------------

// ===============================
// =====    Transmission      ====
// ===============================
void i2cDataTransform() {
  ((String)strainGaugeValue).toCharArray(strainGaugeI2cArray, 5);
  ((String)cameraTriggerValue).toCharArray(cameraTriggerI2cArray, 2);
  ((String)stimulusValue).toCharArray(stimulusI2cArray, 2);
  ((String)motionSensorValuesArray[0]).toCharArray(motionXI2cArray, 8);
  ((String)motionSensorValuesArray[1]).toCharArray(motionYI2cArray, 8);
  ((String)motionSensorValuesArray[2]).toCharArray(motionZI2cArray, 8);
  ((String)analog2Value).toCharArray(analog2I2cArray, 5);
  ((String)lickValue).toCharArray(lickValueI2cArray, 2);
  ((String)lickValveActivationValue).toCharArray(lickValveActivationI2cArray, 2);
  ((String)lickDrainValveActivationValue).toCharArray(lickDrainValveActivationI2cArray, 2);
}

void i2cCommunication() {
  Wire.beginTransmission(0x00); // transmits to device with address 0
  Wire.write(strainGaugeI2cArray); Wire.write(" "); // sends strain gauge value
  Wire.write(cameraTriggerI2cArray); Wire.write(" ");// sends trigger, should be always 1
  Wire.write(stimulusI2cArray); Wire.write(" "); // sends final valve
  Wire.write(motionXI2cArray); Wire.write(" ");
  Wire.write(motionYI2cArray); Wire.write(" ");
  Wire.write(motionZI2cArray); Wire.write(" ");
  Wire.write(analog2I2cArray); Wire.write(" ");
  Wire.write(lickValueI2cArray); Wire.write(" ");
  Wire.write(lickValveActivationI2cArray); Wire.write(" ");
  Wire.write(lickDrainValveActivationI2cArray); Wire.write(" ");
  i2cError = Wire.endTransmission();
}

// ===============================
// =====   Arduino plotting   ====
// ===============================
void ArduinoPlot() {
  Serial.print("motionX: "); Serial.print(motionSensorValuesArray[0]); Serial.print(" ");
  Serial.print("motionY: "); Serial.print(-motionSensorValuesArray[1]); Serial.print(" "); //note the negative
  Serial.print("motionZ: "); Serial.print(motionSensorValuesArray[2]); Serial.print(" ");
  Serial.print("analog2: "); Serial.print(analog2Value); Serial.print(" ");
  Serial.print("lickValue: "); Serial.print(lickValue * 5000); Serial.print(" ");
  Serial.print("lickValveActivationValue: "); Serial.print(lickValveActivationValue * 5000); Serial.print(" ");
  Serial.print("lickDrainValveActivationValue: "); Serial.print(lickDrainValveActivationValue * 5000); Serial.print(" ");
  Serial.print("cameraTriggerValue: "); Serial.print(cameraTriggerValue * 5000); Serial.print(" ");
  Serial.print("stimulusValue: "); Serial.print(stimulusValue * 5000); Serial.print(" ");
  Serial.print("strainGaugeValue: "); Serial.print(strainGaugeValue); Serial.print(" ");

  Serial.print("\n"); //required to plot each time
}

// ===============================
// =====     Acquisition      ====
// ===============================
void dataAcquisition() {
#ifdef MOTION_SENSOR
  motionSensorAcq(&motionSensorValuesArray[0]);
#endif
  analog2Value = analogRead(analog2);
  analog2Value *= (5000 / 1023.0);
  lickValue = digitalRead(lickPin);
  lickValveActivationValue = digitalRead(lickValveActivationPin);
  lickDrainValveActivationValue = digitalRead(lickDrainValveActivationPin);
  cameraTriggerValue = digitalRead(syncPin);
  stimulusValue = digitalRead(stimulusPin);
  strainGaugeValue = analogRead(strainGaugePin);
  strainGaugeValue *= (5000 / 1023.0); //in mV
}
