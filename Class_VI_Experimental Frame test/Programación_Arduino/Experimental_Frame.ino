#include <AcceleroMMA7361.h>


// ACCELEROMETER AND TEMPERATURE SENSOR


#include <AcceleroMMA7361.h>

AcceleroMMA7361 acceleroA;
AcceleroMMA7361 acceleroB;
AcceleroMMA7361 acceleroC;

int x; // x axis variable
int y; // y axis variable
int z; // z axis variable
int xb; // x axis variable
int yb; // y axis variable
int zb; // z axis variable
int xc; // x axis variable
int yc; // y axis variable
int zc; // z axis variable
unsigned long time;


//sleepPin = 13, 
//selfTestPin = 12, 
//zeroGPin = 11, 
//gSelectPin = 10, 
//xPin = A3, 
//yPin = A4, 
//zPin = A5.



void setup() {

  Serial.begin(115200); // 9600 is the frequency of the measure
  
  acceleroA.begin(11, 12, 13, 10, A1, A2, A3);
  acceleroA.setARefVoltage(3.3);                   //sets the AREF voltage to 3.3V
  acceleroA.setSensitivity(HIGH);                   //sets the sensitivity to +/-1.5G
  analogReference(EXTERNAL);
//  acceleroA.calibrate();

  acceleroB.begin(7, 8, 9, 10, A5, A6, A7);
  acceleroB.setARefVoltage(3.3);                   //sets the AREF voltage to 3.3V
  acceleroB.setSensitivity(HIGH);                   //sets the sensitivity to +/-1.5G
  analogReference(EXTERNAL);
//  acceleroB.calibrate();

  acceleroC.begin(2, 3, 4, 10, A8, A9, A10);
  acceleroC.setARefVoltage(3.3);                   //sets the AREF voltage to 3.3V
  acceleroC.setSensitivity(HIGH);                   //sets the sensitivity to +/-1.5G
  analogReference(EXTERNAL);
//  acceleroC.calibrate();
}

void loop() {

time = micros();
//prints time since program started
Serial.print(time);
Serial.print(" ");

// TEMPERATURE
//int temp=analogRead(A0);
//float voltage=(temp/1024.0)*3.3;
//float temperature=(voltage-0.5)*100;

//Serial.print(temp);  // ln creates a blanck line
//Serial.print(" ");

//x=analogRead(3);
//y=analogRead(4);
//z=analogRead(5);

//x = acceleroA.getXAccel();
y = acceleroA.getYAccel();
//z = acceleroA.getZAccel();

//xb = acceleroB.getXAccel();
yb = acceleroB.getYAccel();
//zb = acceleroB.getZAccel();

//xc = acceleroC.getXAccel();
yc = acceleroC.getYAccel();
//zc = acceleroC.getZAccel();

//x = accelero.getXRaw();
//y = accelero.getYRaw();
//z = accelero.getZRaw();

//x = map(x, 0, 1023, -500, 500);
//y = map(y, 0, 1023, -500, 500);
//z = map(z, 0, 1023, -500, 500);

//Serial.print(x);  // ln creates a blanck line
//Serial.print(" ");
Serial.print(y);  // LEVEL 1
Serial.print(" ");
//Serial.print(z);  // ln creates a blanck line
//Serial.print(" ");
//Serial.print(xb);  // ln creates a blanck line
//Serial.print(" ");
Serial.print(yb);  // LEVEL 2
Serial.print(" ");
//Serial.print(zb);  // ln creates a blanck line
//Serial.print(" ");
//Serial.print(xc);  // ln creates a blanck line
//Serial.print(" ");
Serial.println(yc);  // LEVEL 3
//Serial.print(" ");
//Serial.println(zc);  // ln creates a blanck line

}
