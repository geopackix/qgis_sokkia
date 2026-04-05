# Send SDR33 Message vie Serial Port to Sokkia Tachymeter

## Motivation

This tool sends Sokkia SDR33 messages to a Sokkia Tachymeter using UART serial port.

## Config in src/index.ts

```ts
let output = sdr.Sdr33Export.fromGeoJson(process.cwd() + "/WMSGrapped.geojson");

let serialmessage = output.getMessage();

await sendData(serialmessage, "COM10", 9600);
```

## Run

```
npm start
```
