#ifndef MotionSensor_h
#define MotionSensor_h
#include "PS2Mouse.h"

/*
  PS2 Pinout
  1 : Data
  3 : GND
  4 : VCC
  5 : CLK
*/

extern int clockMouse1Pin;
extern int dataMouse1Pin;
extern int clockMouse2Pin;
extern int dataMouse2Pin;

//representation : mouse moves x and y from normal view -> 2D polar coordinates (r = sqrt(x�+y�), theta = atan(y/x))
// mouse turns around = phi -> angular offset of polar in its reference

#include "coordinates.h"

#define PITCH 0.00318 //resolution in cm of the optical mouse
// we can get an 8-bit (127 values) per communication.
//max speed therefore depends on communication speed -> oscilloscope clock 12.85kHz, actual data rate in this code: 59Hz
// max speed = 59*127*0.00318 = 24cm/sec


void motionSensorSetup();
void motionSensorAcq(int *motionArray);
void toVector(float x, float y, float z);

#endif
