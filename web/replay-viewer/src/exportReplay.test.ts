import { describe, expect, it } from "vitest";

import { sampledFrameIndices } from "./exportReplay";

describe("replay export helpers", () => {
  it("samples the complete short replay", () => {
    expect(sampledFrameIndices(4, 10)).toEqual([0, 1, 2, 3]);
  });

  it("preserves first and last frames while bounding long exports", () => {
    const samples = sampledFrameIndices(10_000, 600);
    expect(samples).toHaveLength(600);
    expect(samples[0]).toBe(0);
    expect(samples.at(-1)).toBe(9_999);
    expect(new Set(samples).size).toBe(samples.length);
  });

  it("rejects empty sampling domains without throwing", () => {
    expect(sampledFrameIndices(0, 30)).toEqual([]);
    expect(sampledFrameIndices(30, 0)).toEqual([]);
  });
});
