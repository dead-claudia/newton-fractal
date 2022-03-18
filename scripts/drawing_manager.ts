import { PlotScale } from "./plotter";

const WASM_MODULE_SOURCE_PATH = '../pkg/newton_fractal_bg.wasm';
let wasmModule: InitOutput;

const DRAWING_WORKER_SOURCE_PATH = 'drawing_worker.js';

let drawingWorkersCount = navigator.hardwareConcurrency;
let drawingWorkers: Worker[] = [];
let readyWorkersCount = 0;

class DrawingWork {
    bufferPtr: number;
    bufferSize: number;
    promise: Promise<Uint8ClampedArray>;
    promiseResolve: (value: Uint8ClampedArray) => void;

    constructor(bufferPtr: number, bufferSize: number) {
        this.bufferPtr = bufferPtr;
        this.bufferSize = bufferSize;
        this.promise = new Promise((resolve, _) => {
            this.promiseResolve = resolve;
        });
    }
}

let drawingWork: DrawingWork;

const drawingWorkerCallback = async function (ev: MessageEvent<{ workerId: number, doneDrawing: boolean }>) {
    let message = ev.data;
    let { workerId, doneDrawing } = message;

    readyWorkersCount++;

    if (!doneDrawing) {
        console.log(`Worker #${workerId} initialized`);
        return;
    }
    console.log(`Worker #${workerId} done drawing`);

    if (readyWorkersCount == drawingWorkersCount) {
        let data = new Uint8ClampedArray(wasmModule.memory.buffer, drawingWork.bufferPtr, drawingWork.bufferSize);
        console.log('workers data :>> ', data);
        drawingWork.promiseResolve(data);
    }
}

function createDrawingWorker(sourcePath: string | URL) {
    let worker = new Worker(sourcePath);
    worker.onmessage = drawingWorkerCallback;
    return worker;
}

function initializeWorkers(sharedMemory: WebAssembly.Memory) {
    for (let i = 0; i < drawingWorkersCount; i++) {
        let drawingWorker = createDrawingWorker(DRAWING_WORKER_SOURCE_PATH);
        drawingWorkers.push(drawingWorker);
    }

    for (let i = 0; i < drawingWorkersCount; i++) {
        drawingWorkers[i].postMessage({ workerId: i, sharedMemory });
    }
}

async function initializeDrawing() {
    let sharedMemory = new WebAssembly.Memory({ initial: 100, maximum: 1000, shared: true });
    wasmModule = await wasm_bindgen(WASM_MODULE_SOURCE_PATH, sharedMemory);
    initializeWorkers(sharedMemory);
}

function runDrawingWorkers(plotScale: PlotScale, roots: number[][], iterationsCount: number, colors: number[][], concurrency = 1) {
    if (drawingWork != undefined) {
        return false;
    }
    let { x_display_range: width, y_display_range: height } = plotScale;
    let u32BufferSize = width * height;
    let bufferPtr = wasm_bindgen.create_u32_buffer(u32BufferSize);
    drawingWork = new DrawingWork(bufferPtr, u32BufferSize * 4);

    for (let i = 0; i < concurrency; i++) {
        readyWorkersCount--;
        drawingWorkers[i].postMessage({ plotScale, roots, iterationsCount, colors, partOffset: i, partsCount: concurrency, bufferPtr });
    }

    return drawingWork.promise;
}

initializeDrawing();

export { runDrawingWorkers };