const WASM_MODULE_SOURCE_PATH = '../pkg/newton_fractal_bg.wasm';

importScripts('../pkg/newton_fractal.js');
importScripts('./fractal_calculation.js');

const { fill_pixels_wasm, DrawingModes: WasmDrawingModes, get_wasm_memory } = wasm_bindgen;

function actualCallback(e: MessageEvent<{ drawingModeId: number, plotScale: any, roots: number[][], iterationsCount: number, colors: number[][], partOffset: number, partsCount: number, bufferPtr: number }>) {
    let message = e.data;
    let { drawingModeId, plotScale, roots, iterationsCount, colors, partOffset, partsCount, bufferPtr } = message;

    switch (drawingModeId) {
        case 0:
            fill_pixels_wasm(WasmDrawingModes.Simd, plotScale, roots, iterationsCount, colors, bufferPtr, partOffset, partsCount);
            break;
        case 1:
            fill_pixels_wasm(WasmDrawingModes.Scalar, plotScale, roots, iterationsCount, colors, bufferPtr, partOffset, partsCount);
            break;

        case 2:
            let memory = <WebAssembly.Memory>get_wasm_memory();
            fillPixelsJavascript(<SharedArrayBuffer>memory.buffer, plotScale, roots, iterationsCount, colors, bufferPtr, partOffset, partsCount);
            break;
        default:
            console.log(`Unknown drawing mode, drawing with simds`);
            fill_pixels_wasm(WasmDrawingModes.Simd, plotScale, roots, iterationsCount, colors, bufferPtr, partOffset, partsCount);
            break;
    }

    postMessage(partOffset, undefined);
}

onmessage = async function (e: MessageEvent<{ workerId: number, sharedMemory: WebAssembly.Memory }>) {
    let { workerId, sharedMemory } = e.data;
    await wasm_bindgen(WASM_MODULE_SOURCE_PATH, sharedMemory);

    onmessage = actualCallback;
    postMessage(workerId, undefined);
}