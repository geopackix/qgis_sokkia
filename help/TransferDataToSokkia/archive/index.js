const sdr = require("sdr33").default;

let points = [
  [3523665.383, 5413162.462, 0, "ABS.001"],
  [3523665.472, 5413161.97, 0, "ABS.002"],
  [3523664.98, 5413161.881, 0, "ABS.003"],
  [3523664.891, 5413162.373, 0, "ABS.004"],
];

let result = new sdr.Sdr33Export("Absteckung");

for (P of points) {
  let coordinate = new sdr.Coordinate(P[3], P[1], P[0], P[2], "");
  result.addCoordinate(coordinate);
}

console.log(result.getMessage());
