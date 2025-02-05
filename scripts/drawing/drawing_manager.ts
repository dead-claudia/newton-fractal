import { PlotScale } from "../math/geometry.js";
const { create_u32_buffer, free_u32_buffer } = wasm_bindgen;

const WASM_MODULE_SOURCE_PATH = '../pkg/newton_fractal_bg.wasm';
let wasmModule: InitOutput;

const DRAWING_WORKER_SOURCE_PATH = './drawing/drawing_worker.js';

let drawingWorkersCount = navigator.hardwareConcurrency;
let drawingWorkers: Worker[] = [];
let readyWorkersCount = 0;

let drawingWorkersInitResolve: (value: unknown) => void;
let drawingWorkersInitPromise = new Promise((resolve, _) => {
    drawingWorkersInitResolve = resolve;
});

enum DrawingModes {
    CpuJsScalar = "CPU-js",
    CpuWasmScalar = "CPU-wasm-scalar",
    CpuWasmSimd = "CPU-wasm-simd",
    GpuGlslScalar = "GPU"
}

type DrawingResult = {
    elapsedMs: number,
    drawingMode: DrawingModes,
    plotScale: PlotScale,
    data: Uint8ClampedArray,
}

class DrawingWork {
    bufferPtr: number;
    bufferSize: number;
    drawingMode: DrawingModes;
    plotScale: PlotScale;

    startTime: number;
    promise: Promise<DrawingResult>;
    promiseResolve: (value: DrawingResult) => void;

    constructor(drawingMode: DrawingModes, plotScale: PlotScale, bufferPtr: number, bufferSize: number) {
        this.drawingMode = drawingMode;
        this.plotScale = plotScale;

        this.bufferPtr = bufferPtr;
        this.bufferSize = bufferSize;
        this.promise = new Promise((resolve, _) => {
            this.promiseResolve = resolve;
        });
    }
}

let drawingWork: DrawingWork;

const drawingWorkerDrawingCallback = async function (ev: MessageEvent<number>) {
    let now = performance.now();
    readyWorkersCount++;

    // let workerId = ev.data;
    // console.log(`Worker #${workerId} done drawing`);

    if (readyWorkersCount == drawingWorkersCount) {
        let data = new Uint8ClampedArray(wasmModule.memory.buffer, drawingWork.bufferPtr, drawingWork.bufferSize);
        let drawingResult: DrawingResult = {
            elapsedMs: now - drawingWork.startTime,
            drawingMode: drawingWork.drawingMode,
            plotScale: drawingWork.plotScale,
            data,
        };
        drawingWork.promiseResolve(drawingResult);

        free_u32_buffer(drawingWork.bufferSize / 4, drawingWork.bufferPtr);
        drawingWork = undefined;
    }
}

const drawingWorkerInitCallback = function (ev: MessageEvent<number>) {
    let workerId = ev.data;
    drawingWorkers[workerId].onmessage = drawingWorkerDrawingCallback;

    readyWorkersCount++;
    console.log(`Worker #${workerId} initialized`);

    if (readyWorkersCount == drawingWorkersCount) {
        console.log(`All workers are initialized`);
        drawingWorkersInitResolve(undefined);
        drawingWorkersInitPromise = undefined;
    }
}

function createDrawingWorker(sourcePath: string | URL) {
    let worker = new Worker(sourcePath);
    worker.onmessage = drawingWorkerInitCallback;
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

async function initializeDrawingManager() {
    let sharedMemory = new WebAssembly.Memory({ initial: 100, maximum: 1000, shared: true });
    wasmModule = await wasm_bindgen(WASM_MODULE_SOURCE_PATH, sharedMemory);
    initializeWorkers(sharedMemory);
}

function runDrawingWorkers(drawingMode: DrawingModes, plotScale: PlotScale, roots: number[][], iterationsCount: number, colors: number[][], threadsCount = drawingWorkersCount) {
    if (drawingWorkersInitPromise != undefined) {
        return drawingWorkersInitPromise;
    }

    if (readyWorkersCount != drawingWorkersCount) {
        return false;
    }

    let drawingModeId = Object.values(DrawingModes).indexOf(drawingMode);
    let { x_display_range: width, y_display_range: height } = plotScale;

    let u32BufferSize = width * height;
    let bufferPtr = create_u32_buffer(u32BufferSize);
    drawingWork = new DrawingWork(drawingMode, plotScale, bufferPtr, u32BufferSize * 4);

    readyWorkersCount -= threadsCount;
    drawingWork.startTime = performance.now();
    for (let i = 0; i < threadsCount; i++) {
        drawingWorkers[i].postMessage({ drawingModeId, plotScale, roots, iterationsCount, colors, partOffset: i, partsCount: threadsCount, bufferPtr });
    }

    return drawingWork.promise;
}

initializeDrawingManager();

export {
    DrawingModes,
    DrawingResult,
    runDrawingWorkers
};