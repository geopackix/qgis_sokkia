import sdr from 'sdr33'
import SerialPort from "serialport";


/*
let res = new sdr.Sdr33Export('New Job.');
res.addCoordinate(new sdr.Coordinate('103.0001', 0, 0, 0, ''))
console.log(res.getMessage());
*/

//Async main function
(async () => {

  let output = sdr.Sdr33Export.fromGeoJson(process.cwd() + '/WMSGrapped.geojson');

  let serialmessage = output.getMessage();

  await sendData(serialmessage, 'COM10', 9600);

})();


/**
 * Send Data via serialPort to the total station or any other recipient
 * @param data 
 * @param serialport 
 * @param baud 
 */
async function sendData(data, serialport, baud) {
  let split = data.split("\n");
  console.log(split);

  //const SerialPort = require("serialport");                                   //<-- change this back maybe.
  let port = new SerialPort(serialport, {
    baudRate: baud,
    autoOpen: true,
  });

  for (let line of split) {
    port.write(line);
    port.write([0x0d]);
    port.write([0x0a]);
    console.log('SEND: ' + line);
    await sleep(200);
  }

  port.on("close", function (err) {
    console.log("port closed", err);
  });
}


async function sleep(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}