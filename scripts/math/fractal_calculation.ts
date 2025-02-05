type PlotScale = {
    x_offset: number,
    y_offset: number,
    x_value_range: number,
    y_value_range: number,
    x_display_range: number,
    y_display_range: number
}

class Complex32 {
    re: number;
    im: number;
    constructor(re: number, im: number) {
        this.re = re;
        this.im = im;
    }

    clone() {
        return new Complex32(this.re, this.im);
    }

    add(other: Complex32) {
        this.re += other.re;
        this.im += other.im;
    }

    subtract(other: Complex32) {
        this.re -= other.re;
        this.im -= other.im;
    }

    invert() {
        const square_sum = this.normSqr();
        this.re /= square_sum;
        this.im /= -square_sum;
    }

    normSqr(): number {
        return this.re * this.re + this.im * this.im;
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
    let i = 0;
    for (const root of roots) {
        let diff = z.clone();
        diff.subtract(root);
        if (Math.abs(diff.re) < 0.001 && Math.abs(diff.im) < 0.001) {
            return { idx: i, z };
        }
        diff.invert();
        sum.add(diff);
        i++;
    }
    sum.invert();
    let result = z.clone();
    result.subtract(sum);
    return { idx: -1, z: result };
}

function transformPointToPlotScale(x: number, y: number, plotScale: PlotScale): number[] {
    return [
        plotScale.x_offset + x * plotScale.x_value_range / plotScale.x_display_range,
        plotScale.y_offset + y * plotScale.y_value_range / plotScale.y_display_range
    ];
}

function transformPointToCanvasScale(x: number, y: number, plotScale: PlotScale): number[] {
    return [
        ((x - plotScale.x_offset) * plotScale.x_display_range / plotScale.x_value_range),
        ((y - plotScale.y_offset) * plotScale.y_display_range / plotScale.y_value_range),
    ];
}

function calculatePartSize(totalSize: number, partsCount: number, offset: number, step: number) {
    return Math.round((totalSize * offset) / (partsCount * step)) * step;
}

function fillPixelsJavascript(buffer: SharedArrayBuffer, plotScale: PlotScale, roots: number[][], iterationsCount: number, colors: number[][], bufferPtr: number, partOffset = 0, partsCount = 1) {
    let {
        x_display_range: width,
        y_display_range: height
    } = plotScale;
    let [w_int, h_int] = [Math.round(width), Math.round(height)];

    let flatColors = new Uint8Array(colors.flat());
    let colorPacks = new Uint32Array(flatColors.buffer);

    let complexRoots: Complex32[] = roots.map((pair) => new Complex32(pair[0], pair[1]));

    const filler = (x: number, y: number) => {
        let minDistance = Number.MAX_SAFE_INTEGER;
        let closestRootId = 0;
        let [xp, yp] = transformPointToPlotScale(x, y, plotScale);
        let z = new Complex32(xp, yp);
        let colorId = -1;
        for (let i = 0; i < iterationsCount; i++) {
            let { idx, z: zNew } = newtonMethodApprox(complexRoots, z);
            if (idx != -1) {
                colorId = idx;
                break;
            }
            z = zNew;
        }
        if (colorId != -1) {
            return colorPacks[colorId];
        }
        let i = 0;
        for(const root of complexRoots) {
            let d = z.clone();
            d.subtract(root);
            let dst = d.distance();
            if (dst < minDistance) {
                minDistance = dst;
                closestRootId = i;
            }
            i++;
        }
        return colorPacks[closestRootId];
    }

    let totalSize = w_int * h_int;
    let thisBorder = calculatePartSize(totalSize, partsCount, partOffset, 1);
    let nextBorder = calculatePartSize(totalSize, partsCount, partOffset + 1, 1);

    let u32BufferView = new Uint32Array(buffer, bufferPtr);

    for (let i = thisBorder; i < nextBorder; i++) {
        u32BufferView[i] = filler(i % w_int, i / w_int);
    }
}