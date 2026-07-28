declare module "gifenc" {
  type PixelFormat = "rgb444" | "rgb565" | "rgba4444";
  type Palette = number[][];

  interface QuantizeOptions {
    format?: PixelFormat;
    oneBitAlpha?: boolean;
    clearAlpha?: boolean;
    clearAlphaColor?: number;
  }

  interface FrameOptions {
    palette?: Palette;
    delay?: number;
    repeat?: number;
    transparent?: boolean;
    transparentIndex?: number;
    colorDepth?: number;
    dispose?: number;
  }

  interface Encoder {
    reset(): void;
    finish(): void;
    bytes(): number[];
    bytesView(): Uint8Array;
    writeHeader(): void;
    writeFrame(
      pixels: Uint8Array,
      width: number,
      height: number,
      options?: FrameOptions,
    ): void;
  }

  export function GIFEncoder(options?: {
    initialCapacity?: number;
    auto?: boolean;
  }): Encoder;

  export function quantize(
    rgba: Uint8ClampedArray | Uint8Array,
    maxColors: number,
    options?: QuantizeOptions,
  ): Palette;

  export function applyPalette(
    rgba: Uint8ClampedArray | Uint8Array,
    palette: Palette,
    format?: PixelFormat,
  ): Uint8Array;
}
