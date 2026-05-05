import 'ol/ol.css';
import Map from 'ol/Map';
import View from 'ol/View';
import TileLayer from 'ol/layer/Tile';
import OSM from 'ol/source/OSM';
import VectorLayer from 'ol/layer/Vector';
import VectorSource from 'ol/source/Vector';
import Feature from 'ol/Feature';
import Point from 'ol/geom/Point';
import { Style, Icon, Stroke, Fill, Circle } from 'ol/style';
import { SokkiaDevice } from './device.js';

// App State
const device = new SokkiaDevice();
let map;
let vectorSource;
const protocolEntries = [];

// Initialize Map
function initMap() {
    vectorSource = new VectorSource();
    const vectorLayer = new VectorLayer({
        source: vectorSource,
        style: new Style({
            image: new Circle({
                radius: 6,
                fill: new Fill({ color: '#ff9800' }),
                stroke: new Stroke({ color: '#fff', width: 2 })
            })
        })
    });

    map = new Map({
        target: 'map',
        layers: [
            new TileLayer({
                source: new OSM()
            }),
            vectorLayer
        ],
        view: new View({
            center: [0, 0],
            zoom: 18
        })
    });
}

// UI Helpers
function logToProtocol(msg) {
    const timestamp = new Date().toLocaleTimeString();
    const formatted = `[${timestamp}] ${msg}`;
    protocolEntries.push(formatted);
    const output = document.getElementById('protocol-output');
    output.innerText = protocolEntries.join('\n');
    output.scrollTop = output.scrollHeight;
}

function updateUI(state) {
    document.getElementById('live-hz').innerText = (state.hz || 0).toFixed(4);
    document.getElementById('live-v').innerText = (state.v || 0).toFixed(4);
    document.getElementById('live-sd').innerText = (state.sd || 0).toFixed(3);
    document.getElementById('live-hd').innerText = (state.hd || 0).toFixed(3);
    document.getElementById('live-vd').innerText = (state.vd || 0).toFixed(3);
    
    document.getElementById('cur-e').innerText = (state.e || 0).toFixed(3);
    document.getElementById('cur-n').innerText = (state.n || 0).toFixed(3);
    document.getElementById('cur-z').innerText = (state.z || 0).toFixed(3);
}

// Event Listeners: Navigation
document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
        
        btn.classList.add('active');
        document.getElementById(btn.dataset.tab).classList.add('active');
        
        if (btn.dataset.tab === 'map-tab') {
            setTimeout(() => map.updateSize(), 100);
        }
    });
});

// Event Listeners: Connection
document.getElementById('connect-btn').addEventListener('click', async () => {
    try {
        await device.connect();
        const status = document.getElementById('conn-status');
        status.innerText = "● Verbunden";
        status.className = "status-connected";
        document.getElementById('connect-btn').style.display = "none";
        logToProtocol("Erfolgreich mit Tachymeter verbunden.");
    } catch (err) {
        alert("Verbindung fehlgeschlagen: " + err.message);
    }
});

// Event Listeners: Measurement
document.getElementById('measure-trigger-btn').addEventListener('click', async () => {
    logToProtocol("Starte Messung...");
    await device.measure();
});

// Event Listeners: Remote Control
const joyEvents = {
    'joy-up': () => device.rotateV(0.1),
    'joy-down': () => device.rotateV(-0.1),
    'joy-left': () => device.rotateHz(-0.1),
    'joy-right': () => device.rotateHz(0.1),
    'joy-center': () => device.measure()
};

Object.entries(joyEvents).forEach(([id, fn]) => {
    document.getElementById(id).addEventListener('click', fn);
});

// Device Data Handlers
window.addEventListener('sokkia-data', (e) => {
    const { raw, state } = e.detail;
    
    // Update numerical UI
    updateUI(state);
    
    // Map Visualization
    if (state.e !== 0 && state.n !== 0) {
        // Find existing feature for this point or create new one
        const coords = [state.e, state.n];
        const feature = new Feature({
            geometry: new Point(coords),
            name: state.pointName || "Point"
        });
        
        // Style based on point type (Station vs Measurement)
        if (raw.startsWith('02')) { // SDR33 Station Point
            feature.setStyle(new Style({
                image: new Circle({
                    radius: 8,
                    fill: new Fill({ color: '#f44336' }),
                    stroke: new Stroke({ color: '#fff', width: 3 })
                })
            }));
        }
        
        vectorSource.addFeature(feature);
        
        // Auto-center view if it's the first point or requested
        if (vectorSource.getFeatures().length === 1) {
            map.getView().animate({ center: coords, zoom: 20 });
        }
    }
    
    logToProtocol(`RAW: ${raw}`);
});

// Protocol Export
document.getElementById('download-prot').addEventListener('click', () => {
    const blob = new Blob([protocolEntries.join('\n')], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `Sokkia_Protokoll_${new Date().toISOString().slice(0,10)}.txt`;
    a.click();
    logToProtocol("Protokoll exportiert.");
});

document.getElementById('clear-prot').addEventListener('click', () => {
    if (confirm("Protokoll wirklich leeren?")) {
        protocolEntries.length = 0;
        document.getElementById('protocol-output').innerText = "";
        logToProtocol("Protokoll geleert.");
    }
});

// Bootstrap
document.addEventListener('DOMContentLoaded', () => {
    initMap();
    logToProtocol("Web-Interface bereit. Bitte verbinden Sie das Gerät.");
});
