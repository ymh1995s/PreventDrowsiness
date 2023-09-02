
void setup() {
  Serial.begin(9600);
  pinMode(3, OUTPUT); //+
  pinMode(7, OUTPUT); //-
  pinMode(2, OUTPUT); //test
  pinMode(8, OUTPUT); //test
  pinMode(9, OUTPUT); //test

}

void loop() {



  while (Serial.available() > 0) {
    char c =Serial.read();
    if(c=='y')
     {
    digitalWrite(9,HIGH);
    delay(2000);
    digitalWrite(9,LOW);

    digitalWrite(7,HIGH);
    delay(2000);
    digitalWrite(7,LOW);
    delay(5000);
    digitalWrite(2,HIGH);
    digitalWrite(9,HIGH);
    delay(2000);
    digitalWrite(2,LOW);
    delay(2000);
      }
    if(c=='k')
     {
    digitalWrite(7,HIGH);
    delay(2000);
    digitalWrite(7,LOW);

      }
  }

}
