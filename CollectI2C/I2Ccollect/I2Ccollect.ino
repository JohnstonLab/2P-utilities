// /!\ OVERWRITE cameraTriggerValue for constant i2c communication without trigger
//SCB-19: red: SDA   black: SCL
//arduino: red: A4   black: A5
//mega : data: 20, clock : 21

//COMMENT MOTION_SENSOR IF NOT USING THE OPTICAL MICE
//COMMENT _DEBUG_ IF NOT DEBUGGING

#include "Arduino.h"
#include <Wire.h>

//#define _DEBUG_ // debug conditional compiling
//#define MOTION_SENSOR //conditional compiling if using optical mice. COMMENT IF NOT USNG THE MICE


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

//thermocamera
int thermRespirationPin = A5;

//Lick sensor
int lickPin = 8;
int lickValveActivationPin = 9;
int lickDrainValveActivationPin = 10;

//strain gaugea1
int strainGaugePin = A4;

//camera trigger and stimulus
int cameraTriggerPin = 15; //
int stimulusPin = 16; //aurora 206 final valve

// swithI2C mode
int I2CSwitch = 13;

//olf220A
int cleanAirPin = A0;
int OdourFlowPin = A1;
int DilutionFlowPin = A2;
int OdSig1Pin = 31;
int OdSig2Pin = 32;
int OdSig3Pin = 33;
int OdSig4Pin = 34;
int OdSig5Pin = 35;

// ===============================
// =====  Acquistion Values   ====
// ===============================
int motionSensorValuesArray[3] = {0};
int thermRespValue;
int lickValue;
int lickValveActivationValue;
int lickDrainValveActivationValue;
int strainGaugeValue;
int cameraTriggerValue; // == 1 when triggered
int stimulusValue;
int cleanAir;
int OdourFlow;
int DilutionFlow;
int OdSig1;
int OdSig2;
int OdSig3;
int OdSig4;
int OdSig5;
int OdourVial;

// ===============================
// =====         I2C          ====
// ===============================
int DELAY_I2C = 3;
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
char thermRespI2cArray[5]; 
//lick sensor
char lickValueI2cArray[2];
char lickValveActivationI2cArray[2];
char lickDrainValveActivationI2cArray[2];
//olf220A
char cleanAirI2cArray[5];
char OdourFlowI2cArray[5];
char DilutionFlowI2cArray[5];
char OdourVialI2cArray[5];

// the setup routine runs once when you press reset:
void setup() {
  pinMode(20, INPUT_PULLUP);
  pinMode(21, INPUT_PULLUP); //sets i2C bus with internal pullup
  pinMode(cameraTriggerPin, INPUT_PULLUP);
  pinMode(stimulusPin, INPUT);
  pinMode(thermRespirationPin, INPUT);
  pinMode(lickPin, INPUT);
  pinMode(lickValveActivationPin, INPUT);
  pinMode(lickDrainValveActivationPin, INPUT);
  pinMode(strainGaugePin, INPUT);
  pinMode(I2CSwitch, INPUT_PULLUP);
  pinMode(cleanAirPin, INPUT);
  pinMode(OdourFlowPin, INPUT);
  pinMode(DilutionFlowPin, INPUT);
  pinMode(OdSig1Pin, INPUT);
  pinMode(OdSig2Pin, INPUT);
  pinMode(OdSig3Pin, INPUT);
  pinMode(OdSig4Pin, INPUT);
  pinMode(OdSig5Pin, INPUT);
  Wire.begin();
  //Wire.setClock(400000); //400kHz i2C freq change. must be left to default (100kHz) for scanImage to handle the data!
  Serial.begin(500000);
#ifdef MOTION_SENSOR
  motionSensorSetup();
#endif
}

void loop() {
  dataAcquisition();
 // ArduinoPlot();
  i2cDataTransform();
  delay(DELAY_I2C);

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
  ((String)thermRespValue).toCharArray(thermRespI2cArray, 5);
  ((String)lickValue).toCharArray(lickValueI2cArray, 2);
  ((String)lickValveActivationValue).toCharArray(lickValveActivationI2cArray, 2);
  ((String)lickDrainValveActivationValue).toCharArray(lickDrainValveActivationI2cArray, 2);
  ((String)cleanAir).toCharArray(cleanAirI2cArray, 5);
  ((String)OdourFlow).toCharArray(OdourFlowI2cArray, 5);
  ((String)DilutionFlow).toCharArray(DilutionFlowI2cArray, 5);
  ((String)OdourVial).toCharArray(OdourVialI2cArray, 5);
}



void i2cCommunication() {
  Wire.beginTransmission(0x00); // transmits to device with address 0
  Wire.write(strainGaugeI2cArray); Wire.write(" "); // sends strain gauge value
  Wire.write(cameraTriggerI2cArray); Wire.write(" ");// sends trigger, should be always 1
  Wire.write(stimulusI2cArray); Wire.write(" "); // sends final valve
//  Wire.write(motionXI2cArray); Wire.write(" ");
//  Wire.write(motionYI2cArray); Wire.write(" ");
//  Wire.write(motionZI2cArray); Wire.write(" ");
//  Wire.write(thermRespI2cArray); Wire.write(" ");
//  Wire.write(lickValueI2cArray); Wire.write(" ");
//  Wire.write(lickValveActivationI2cArray); Wire.write(" ");
//  Wire.write(lickDrainValveActivationI2cArray); Wire.write(" ");
  Wire.write(cleanAirI2cArray); Wire.write(" ");
  Wire.write(OdourFlowI2cArray); Wire.write(" ");
  Wire.write(DilutionFlowI2cArray); Wire.write(" ");
  Wire.write(OdourVialI2cArray); Wire.write(" ");
  i2cError = Wire.endTransmission();
}

// ===============================
// =====   Arduino plotting   ====
// ===============================
// olf220a not plotted
void ArduinoPlot() {
//  Serial.print("motionX: "); Serial.print(motionSensorValuesArray[0]); Serial.print(" ");
//  Serial.print("motionY: "); Serial.print(-motionSensorValuesArray[1]); Serial.print(" "); //note the negative
//  Serial.print("motionZ: "); Serial.print(motionSensorValuesArray[2]); Serial.print(" ");
  Serial.print("termResp: "); Serial.print(thermRespValue); Serial.print(" ");
  Serial.print("lickValue: "); Serial.print(lickValue * 5000); Serial.print(" ");
  Serial.print("lickValveActivationValue: "); Serial.print(lickValveActivationValue * 5000); Serial.print(" ");
  Serial.print("lickDrainValveActivationValue: "); Serial.print(lickDrainValveActivationValue * 5000); Serial.print(" ");
  Serial.print("cameraTriggerValue: "); Serial.print(cameraTriggerValue * 5000); Serial.print(" ");
  Serial.print("stimulusValue: "); Serial.print(stimulusValue * 5000); Serial.print(" ");
  Serial.print("strainGaugeValue: "); Serial.print(strainGaugeValue); Serial.print(" ");
//  Serial.print("OdourVial: "); Serial.print(OdourVial); Serial.print(" ");
//  Serial.print("cleanAir: "); Serial.print(cleanAir); Serial.print(" ");
//  Serial.print("OdourFlow: "); Serial.print(OdourFlow); Serial.print(" ");
//  Serial.print("DilutionFlow: "); Serial.print(DilutionFlow); Serial.print(" ");
  Serial.print("\n"); //required to plot each time
}

// ===============================
// =====     Acquisition      ====
// ===============================
void dataAcquisition() {
#ifdef MOTION_SENSOR
  motionSensorAcq(&motionSensorValuesArray[0]);
#endif
  thermRespValue = analogRead(thermRespirationPin);
  thermRespValue *= (5000 / 1023.0);
  lickValue = digitalRead(lickPin);
  lickValveActivationValue = digitalRead(lickValveActivationPin);
  lickDrainValveActivationValue = digitalRead(lickDrainValveActivationPin);
  cameraTriggerValue = digitalRead(cameraTriggerPin);
  stimulusValue = digitalRead(stimulusPin);
  strainGaugeValue = analogRead(strainGaugePin);
  strainGaugeValue *= (5000 / 1023.0); //in mV
  cleanAir = analogRead(cleanAirPin);
  OdourFlow = analogRead(OdourFlowPin);
//  OdourFlow *=100;
  DilutionFlow = analogRead(DilutionFlowPin);
  OdSig1 = digitalRead(OdSig1Pin);
  OdSig2 = digitalRead(OdSig2Pin);
  OdSig3 = digitalRead(OdSig3Pin);
  OdSig4 = digitalRead(OdSig4Pin);
  OdSig5 = digitalRead(OdSig5Pin);
  if(OdSig1 == HIGH && OdSig2 ==LOW && OdSig3 ==LOW && OdSig4 ==LOW && OdSig5 == HIGH){
    OdourVial = 0;
  }
  if(OdSig1 == HIGH && OdSig2 ==LOW && OdSig3 ==LOW && OdSig4 ==LOW && OdSig5 == LOW){
    OdourVial = 1;
  }
  if(OdSig1 == LOW && OdSig2 ==HIGH && OdSig3 ==LOW && OdSig4 ==LOW && OdSig5 == LOW){
    OdourVial = 2;
  }
  if(OdSig1 == HIGH && OdSig2 ==HIGH && OdSig3 ==LOW && OdSig4 ==LOW && OdSig5 == LOW){
    OdourVial = 3;
  }
  if(OdSig1 == LOW && OdSig2 ==LOW && OdSig3 ==HIGH && OdSig4 ==LOW && OdSig5 == LOW){
    OdourVial = 4;
  }
  if(OdSig1 == HIGH && OdSig2 ==LOW && OdSig3 ==HIGH && OdSig4 ==LOW && OdSig5 == LOW){
    OdourVial = 5;
  }
  if(OdSig1 == LOW && OdSig2 ==HIGH && OdSig3 ==HIGH && OdSig4 ==LOW && OdSig5 == LOW){
    OdourVial = 6;
  }
  if(OdSig1 == HIGH && OdSig2 ==HIGH && OdSig3 ==HIGH && OdSig4 ==LOW && OdSig5 == LOW){
    OdourVial = 7;
  }
  if(OdSig1 == LOW && OdSig2 ==LOW && OdSig3 ==LOW && OdSig4 ==HIGH && OdSig5 == LOW){
    OdourVial = 8;
  }
  if(OdSig1 == HIGH && OdSig2 ==LOW && OdSig3 ==LOW && OdSig4 ==HIGH && OdSig5 == LOW){
    OdourVial = 9;
  }
  if(OdSig1 == LOW && OdSig2 ==HIGH && OdSig3 ==LOW && OdSig4 ==HIGH && OdSig5 == LOW){
    OdourVial = 10;
  } 
  if(OdSig1 == HIGH && OdSig2 ==HIGH && OdSig3 ==LOW && OdSig4 ==HIGH && OdSig5 == LOW){
    OdourVial = 11;
  }
  if(OdSig1 == LOW && OdSig2 ==LOW && OdSig3 ==HIGH && OdSig4 ==HIGH && OdSig5 == LOW){
    OdourVial = 12;
  }
  if(OdSig1 == HIGH && OdSig2 ==LOW && OdSig3 ==HIGH && OdSig4 ==HIGH && OdSig5 == LOW){
    OdourVial = 13;
  }
  if(OdSig1 == LOW && OdSig2 ==HIGH && OdSig3 ==HIGH && OdSig4 ==HIGH && OdSig5 == LOW){
    OdourVial = 14;
  }
  if(OdSig1 == HIGH && OdSig2 ==HIGH && OdSig3 ==HIGH && OdSig4 ==HIGH && OdSig5 == LOW){
    OdourVial = 15;
  }
  if(OdSig1 == LOW && OdSig2 ==LOW && OdSig3 ==LOW && OdSig4 ==LOW && OdSig5 == HIGH){
    OdourVial = 16;
  }
}
