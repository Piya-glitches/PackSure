import { useEffect, useRef } from "react";
import type { ComplianceReport } from "../types";
import { FIELD_LABELS } from "../types";

const STATUS_COLOR: Record<string, string> = {
  PASS: "#2F6B4F",
  FAIL: "#A3402B",
  WARN: "#C97A2B",
  NOT_FOUND: "#9AA3B5",
};

export default function ImageWithOverlay({ imageDataUrl, report }: { imageDataUrl: string; report: ComplianceReport }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const img = new Image();
    img.onload = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d")!;
      const displayWidth = 420;
      const scale = displayWidth / img.width;
      canvas.width = img.width * scale;
      canvas.height = img.height * scale;
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      for (const f of report.fields) {
        if (!f.bbox) continue;
        const color = STATUS_COLOR[f.status] ?? "#9AA3B5";
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(f.bbox.x * scale, f.bbox.y * scale, f.bbox.w * scale, f.bbox.h * scale);

        const label = FIELD_LABELS[f.field_key]?.split("/")[0].trim() ?? f.field_key;
        ctx.font = "10px monospace";
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = color;
        ctx.fillRect(f.bbox.x * scale, f.bbox.y * scale - 14, tw + 8, 14);
        ctx.fillStyle = "#fff";
        ctx.fillText(label, f.bbox.x * scale + 4, f.bbox.y * scale - 3);
      }

      if (report.calibration.found && report.calibration.bbox) {
        const b = report.calibration.bbox;
        ctx.strokeStyle = "#3B6FB0";
        ctx.setLineDash([4, 3]);
        ctx.lineWidth = 2;
        ctx.strokeRect(b.x * scale, b.y * scale, b.w * scale, b.h * scale);
        ctx.setLineDash([]);
      }
    };
    img.src = imageDataUrl;
  }, [imageDataUrl, report]);

  return <canvas ref={canvasRef} className="rounded-lg border border-line max-w-full" />;
}
