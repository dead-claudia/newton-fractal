class Complex32 {
    constructor(re, im) {
        this.re = re;
        this.im = im;
    }
    clone() {
        return new Complex32(this.re, this.im);
    }
    add(other) {
        this.re += other.re;
        this.im += other.im;
    }
    subtract(other) {
        this.re -= other.re;
        this.im -= other.im;
    }
    invert() {
        const square_sum = this.normSqr();
        this.re /= square_sum;
        this.im /= -square_sum;
    }
    normSqr() {
        return this.re * this.re + this.im * this.im;
    }
    distance() {
        return Math.sqrt(this.re * this.re + this.im * this.im);
    }
}
function newtonMethodApprox(roots, z) {
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
function transformPointToPlotScale(x, y, plotScale) {
    return [
        plotScale.x_offset + x * plotScale.x_value_range / plotScale.x_display_range,
        plotScale.y_offset + y * plotScale.y_value_range / plotScale.y_display_range
    ];
}
function transformPointToCanvasScale(x, y, plotScale) {
    return [
        ((x - plotScale.x_offset) * plotScale.x_display_range / plotScale.x_value_range),
        ((y - plotScale.y_offset) * plotScale.y_display_range / plotScale.y_value_range),
    ];
}
function calculatePartSize(totalSize, partsCount, offset, step) {
    return Math.round((totalSize * offset) / (partsCount * step)) * step;
}
function fillPixelsJavascript(buffer, plotScale, roots, iterationsCount, colors, bufferPtr, partOffset = 0, partsCount = 1) {
    let { x_display_range: width, y_display_range: height } = plotScale;
    let [w_int, h_int] = [Math.round(width), Math.round(height)];
    let flatColors = new Uint8Array(colors.flat());
    let colorPacks = new Uint32Array(flatColors.buffer);
    let complexRoots = roots.map((pair) => new Complex32(pair[0], pair[1]));
    const filler = (x, y) => {
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
        for (const root of complexRoots) {
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
    };
    let totalSize = w_int * h_int;
    let thisBorder = calculatePartSize(totalSize, partsCount, partOffset, 1);
    let nextBorder = calculatePartSize(totalSize, partsCount, partOffset + 1, 1);
    let u32BufferView = new Uint32Array(buffer, bufferPtr);
    for (let i = thisBorder; i < nextBorder; i++) {
        u32BufferView[i] = filler(i % w_int, i / w_int);
    }
}
//# sourceMappingURL=fractal_calculation.js.map