"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const sdr33_1 = __importDefault(require("sdr33"));
const serialport_1 = __importDefault(require("serialport"));
/*
let res = new sdr.Sdr33Export('New Job.');
res.addCoordinate(new sdr.Coordinate('103.0001', 0, 0, 0, ''))
console.log(res.getMessage());
*/
//Async main function
(() => __awaiter(void 0, void 0, void 0, function* () {
    let output = sdr33_1.default.Sdr33Export.fromGeoJson(process.cwd() + '/WMSGrapped.geojson');
    let serialmessage = output.getMessage();
    yield sendData(serialmessage, 'COM10', 9600);
}))();
/**
 * Send Data via serialPort to the total station or any other recipient
 * @param data
 * @param serialport
 * @param baud
 */
function sendData(data, serialport, baud) {
    return __awaiter(this, void 0, void 0, function* () {
        let split = data.split("\n");
        console.log(split);
        //const SerialPort = require("serialport");                                   //<-- change this back maybe.
        let port = new serialport_1.default(serialport, {
            baudRate: baud,
            autoOpen: true,
        });
        for (let line of split) {
            port.write(line);
            port.write([0x0d]);
            port.write([0x0a]);
            console.log('SEND: ' + line);
            yield sleep(200);
        }
        port.on("close", function (err) {
            console.log("port closed", err);
        });
    });
}
function sleep(ms) {
    return __awaiter(this, void 0, void 0, function* () {
        return new Promise((resolve) => {
            setTimeout(resolve, ms);
        });
    });
}
//# sourceMappingURL=index.js.map