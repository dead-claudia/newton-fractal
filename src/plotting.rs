use crate::approximation::Approximation;

use super::polynomial::Polynomial;
use num_complex::Complex;
use wasm_bindgen::{prelude::*, Clamped};
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement, ImageData};

use super::logger::*;

#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct Dimension {
    pub width: f64,
    pub height: f64,
    pub x_range: f64,
    pub y_range: f64,
    pub x_offset: f64,
    pub y_offset: f64,
}

#[wasm_bindgen]
impl Dimension {
    #[wasm_bindgen(constructor)]
    pub fn new(
        width: f64,
        height: f64,
        x_range: f64,
        y_range: f64,
        x_offset: f64,
        y_offset: f64,
    ) -> Dimension {
        Dimension {
            width,
            height,
            x_range,
            y_range,
            x_offset,
            y_offset,
        }
    }
}

#[wasm_bindgen]
pub struct Plotter {
    pub dimension: Dimension,
    canvas: HtmlCanvasElement,
    context: CanvasRenderingContext2d,
}

#[wasm_bindgen]
impl Plotter {
    #[wasm_bindgen(constructor)]
    pub fn new(
        dimension: Dimension,
        canvas: HtmlCanvasElement,
        context: CanvasRenderingContext2d,
    ) -> Plotter {
        Plotter {
            dimension,
            canvas,
            context,
        }
    }

    #[wasm_bindgen]
    pub fn canvas_to_plot_to_js(&self, x: f64, y: f64) -> JsValue {
        let p = self.canvas_to_plot(x, y);
        JsValue::from_serde(&p).unwrap()
    }

    #[wasm_bindgen]
    pub fn resize_canvas(&self) {
        self.canvas.set_width(self.dimension.width as u32);
        self.canvas.set_height(self.dimension.height as u32);
    }

    #[wasm_bindgen]
    pub fn plot_point(&self, x: f64, y: f64, color: &JsValue, size: f64) {
        let ctx = &self.context;
        let (canvas_x, canvas_y) = self.plot_to_canvas(x, y);
        ctx.move_to(canvas_x, canvas_y);
        ctx.begin_path();
        ctx.arc(canvas_x, canvas_y, size, 0f64, 2f64 * std::f64::consts::PI)
            .unwrap();
        ctx.set_fill_style(color);
        ctx.fill();
        ctx.stroke();
        ctx.close_path();
    }

    #[wasm_bindgen]
    pub fn plot_points(
        &self,
        step_x: f64,
        step_y: f64,
        polynom: &Polynomial,
        approximation: &Approximation,
        point_size: Option<f64>,
    ) {
        if polynom.get_roots().len() == 0 {
            return;
        }

        let (width, height) = (self.dimension.width, self.dimension.height);

        let point_size = match point_size {
            Some(v) => v,
            None => 0.5,
        };

        let ctx = &self.context;
        let canvas = &self.canvas;
        ctx.clear_rect(0f64, 0f64, canvas.width().into(), canvas.height().into());

        let (x_range, y_range) = (self.dimension.x_range, self.dimension.y_range);
        log!("x_range: {}, y_range: {}", x_range, y_range);
        let size = ((step_x + step_y) / 2.0) * ((width + height) / 2.0) * point_size / 6.0;
        ctx.set_fill_style(&"grey".into());

        let mut y = self.dimension.y_offset + step_y / 2.0;
        while y < y_range + step_y / 2.0 {
            let mut x = self.dimension.x_offset + step_x / 2.0;
            while x < x_range + step_x / 2.0 {
                let z = Complex::<f64>::new(x, y);
                // log!("Original point: {:?}", z);
                let z = polynom.calculate(z).unwrap();
                // log!("Result point: {:?}", z);
                let (canvas_x, canvas_y) = self.plot_to_canvas(z.re, z.im);
                // log!("Remapped point: ({}, {})", canvas_x, canvas_y);

                ctx.move_to(canvas_x, canvas_y);
                ctx.begin_path();
                match ctx.arc(canvas_x, canvas_y, size, 0f64, 2f64 * std::f64::consts::PI) {
                    Ok(_) => (),
                    Err(_) => (),
                };
                ctx.fill();
                ctx.stroke();
                ctx.close_path();

                x += step_x;
            }
            y += step_y;
        }

        for root in polynom.get_roots().iter() {
            let p = root.clone();
            self.plot_point(p.re, p.im, &"red".into(), 4.0 * size);
        }

        for point in approximation.get_points().iter() {
            let p = point.clone();
            self.plot_point(p.re, p.im, &"blue".into(), 3.0 * size);
        }
    }

    #[wasm_bindgen]
    pub fn reverse_colors(&self) {
        let (width, height) = (self.dimension.width, self.dimension.height);
        match self.context.get_image_data(0.0, 0.0, width, height) {
            Ok(image_data) => {
                let mut data = image_data.data();
                for i in (0..data.len()).step_by(4) {
                    data[i] ^= 255;
                    data[i + 1] ^= 255;
                    data[i + 2] ^= 255;
                }
                let new_image_data = match ImageData::new_with_u8_clamped_array_and_sh(
                    Clamped(data.as_slice()),
                    width as u32,
                    height as u32,
                ) {
                    Ok(v) => v,
                    Err(e) => {
                        log!(
                            "Error creating new image data of size {}x{}: {:?}",
                            width,
                            height,
                            e
                        );
                        return;
                    }
                };
                match self.context.put_image_data(&new_image_data, 0.0, 0.0) {
                    Ok(_) => log!("Successfully modified values in image data and applied them."),
                    Err(e) => log!("Error applying modified image: {:?}", e),
                }
            }
            Err(e) => log!("Error getting canvas image data: {:?}", e),
        }
    }
}

impl Plotter {
    pub fn canvas_to_plot(&self, x: f64, y: f64) -> (f64, f64) {
        (
            self.dimension.x_offset + x * self.dimension.x_range / self.dimension.width,
            self.dimension.y_offset + y * self.dimension.y_range / self.dimension.height,
        )
    }
    pub fn plot_to_canvas(&self, x: f64, y: f64) -> (f64, f64) {
        (
            (x - self.dimension.x_offset) * self.dimension.width / self.dimension.x_range,
            (y - self.dimension.y_offset) * self.dimension.height / self.dimension.y_range,
        )
    }
}
