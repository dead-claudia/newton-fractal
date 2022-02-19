import { Dimension, Plotter, Polynomial } from "../pkg/newton_fractal.js";

class Complex32 {
    re: number;
    im: number;
    constructor(re: number, im: number) {
        this.re = re;
        this.im = im;
    }

    add(other: Complex32): Complex32 {
        return new Complex32(this.re + other.re, this.im + other.im);
    }

    subtract(other: Complex32): Complex32 {
        return new Complex32(this.re - other.re, this.im - other.im);
    }

    normSqr(): number {
        return this.re * this.re + this.im * this.im;
    }

    invert(): Complex32 {
        const square_sum = this.normSqr();
        return new Complex32(this.re / square_sum, -this.im / square_sum);
    }

    distance(): number {
        return Math.sqrt(this.re * this.re + this.im * this.im);
    }
}

function newtonMethodApprox(roots: Complex32[], z: Complex32): {
    idx: number;
    z: Complex32;
} {
    let sum = new Complex32(0, 0);
    for (let i in roots) {
        const root = roots[i];
        const diff = z.subtract(root);
        if (diff.re == 0 && diff.im == 0) {
            return { idx: +i, z };
        }
        sum = sum.add(diff.invert());
    }
    return { idx: -1, z: z.subtract(sum.invert()) };
}

function calcDimension(innerWidth: number, innerHeight: number): Dimension {
    const width = Math.round(innerWidth * 0.65 / 4) * 4;
    const height = Math.round(innerHeight * 0.75);
    const k = height / width;
    const x_range = 4;
    const x_offset = -2;
    return new Dimension(width, height, x_range, x_range * k, x_offset, x_offset * k);
}

function mapPoints(p: Plotter, x: number, y: number): [number, number] {
    return <[number, number]>p.canvas_point_to_plot_to_js(x, y);
}

function fillPixelsJavascript(plotter: Plotter, polynom: Polynomial, iterationsCount: number, colors: number[][]): ImageData {
    let { width, height } = plotter.dimension;
    let [w_int, h_int] = [Math.round(width), Math.round(height)];

    let flatColors = new Uint8Array(colors.flat());
    let colorPacks = new Uint32Array(flatColors.buffer);

    let roots: Complex32[] = (<number[][]>polynom.get_roots_to_js()).map((pair) => new Complex32(pair[0], pair[1]));
    let uint32Data = new Uint32Array(w_int * h_int);
    let index = 0;

    for (let y = 0; y < h_int; y++) {
        for (let x = 0; x < w_int; x++) {
            let minDistance = Number.MAX_SAFE_INTEGER;
            let closestRootId = 0;
            let [xp, yp] = mapPoints(plotter, x, y);
            let z = new Complex32(xp, yp);
            let colorId = -1;
            for (let i = 0; i < iterationsCount; i++) {
                let { idx, z: zNew } = newtonMethodApprox(roots, z);
                if (idx != -1) {
                    colorId = idx;
                    break;
                }
                z = zNew;
            }
            if (colorId != -1) {
                uint32Data[index++] = colorPacks[colorId];
                continue;
            }
            for (let i in roots) {
                const root = roots[i];
                let d = z.subtract(root).distance();
                if (d < minDistance) {
                    minDistance = d;
                    closestRootId = +i;
                }
            }
            uint32Data[index++] = colorPacks[closestRootId];
        }
    }
    let uint8Data = new Uint8ClampedArray(uint32Data.buffer);

    let imageData = new ImageData(uint8Data, w_int, h_int);
    console.log('fill pixels from js imageData :>> ', imageData);
    return imageData;
    // plotter.put_image_data_from_js(imageData);
}

export {
    calcDimension,
    mapPoints,
    fillPixelsJavascript
};