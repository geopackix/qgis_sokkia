/**
 * Kommunikations-Logik für Sokkia-Tachymeter (WebSerial)
 */
export class SokkiaDevice {
    constructor() {
        this.port = null;
        this.reader = null;
        this.writer = null;
        this.encoder = new TextEncoder();
        this.decoder = new TextDecoder();
        this.buffer = "";
        
        // Aktueller Status des Geräts
        this.state = {
            hz: 0,
            v: 0,
            sd: 0,
            hd: 0,
            vd: 0,
            e: 0,
            n: 0,
            z: 0,
            pointName: "",
            targetHeight: 2.0,
            instrumentHeight: 1.6,
            prismConstant: 0.0,
            mode: 'fine' // 'fine', 'rapid', 'tracking'
        };
    }

    async connect() {
        if (!('serial' in navigator)) {
            throw new Error("WebSerial wird von diesem Browser nicht unterstützt.");
        }

        try {
            this.port = await navigator.serial.requestPort();
            await this.port.open({ baudRate: 9600, dataBits: 8, stopBits: 1, parity: 'none' });

            this.writer = this.port.writable.getWriter();
            this.readLoop();
        } catch (err) {
            console.error("Verbindungsfehler:", err);
            throw err;
        }
    }

    async sendCommand(command) {
        if (!this.writer) return;
        // Sokkia Befehle enden oft mit \r\n
        console.log("Sende:", command);
        await this.writer.write(this.encoder.encode(command + "\r\n"));
    }

    async readLoop() {
        while (this.port.readable) {
            this.reader = this.port.readable.getReader();
            try {
                while (true) {
                    const { value, done } = await this.reader.read();
                    if (done) break;
                    const text = this.decoder.decode(value);
                    this.buffer += text;
                    
                    // Verarbeite vollständige Zeilen (SDR33 / ACK/NAK)
                    let lines = this.buffer.split(/\r?\n/);
                    this.buffer = lines.pop(); // Rest im Buffer behalten
                    
                    for (let line of lines) {
                        if (line.trim()) {
                            this.handleData(line.trim());
                        }
                    }
                }
            } catch (error) {
                console.error("Fehler beim Lesen:", error);
            } finally {
                this.reader.releaseLock();
            }
        }
    }

    handleData(data) {
        console.log("Empfangen:", data);
        
        // SDR33 Parsing (Minimal-Beispiel basierend auf Python-Logik)
        // 08KI: Koordinaten-Datensatz
        // 09MC: Messwert-Datensatz
        if (data.startsWith('09MC')) {
            // Beispiel: 09MC100.000 123.4567 100.2345 50.123
            // Extrahiere SD, Hz, V (Positionen hängen vom SDR33 Dialekt ab)
            // Hier ein vereinfachter Split für die Demo:
            const parts = data.substring(4).split(/\s+/).filter(p => p.length > 0);
            if (parts.length >= 3) {
                this.state.sd = parseFloat(parts[0]);
                this.state.v = parseFloat(parts[1]);
                this.state.hz = parseFloat(parts[2]);
            }
        } else if (data.startsWith('08KI')) {
            // Koordinaten: E, N, Z
            const parts = data.substring(4).split(/\s+/).filter(p => p.length > 0);
            if (parts.length >= 3) {
                this.state.e = parseFloat(parts[0]);
                this.state.n = parseFloat(parts[1]);
                this.state.z = parseFloat(parts[2]);
            }
        }

        // Event auslösen für UI
        const event = new CustomEvent('sokkia-data', { 
            detail: { 
                raw: data,
                state: { ...this.state } 
            } 
        });
        window.dispatchEvent(event);
    }

    /**
     * Messung auslösen
     */
    async measure() {
        // GET MEASUREMENT (SDR33 Format)
        await this.sendCommand("\x0300NM"); // Standard SDR33 Query
    }

    /**
     * Fernsteuerung: Drehen
     */
    async rotateHz(angleGon) {
        await this.sendCommand(`CNH${angleGon.toFixed(4)}`);
    }

    async rotateV(angleGon) {
        await this.sendCommand(`CNV${angleGon.toFixed(4)}`);
    }

    /**
     * Konfiguration
     */
    async setTargetType(isPrism) {
        const cmd = isPrism ? "TSNP" : "TSNR"; 
        await this.sendCommand(cmd);
    }
}
