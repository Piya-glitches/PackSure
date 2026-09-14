import { useRef, useState } from "react";

export default function Uploader({ onImage }: { onImage: (dataUrl: string) => void }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);

  function loadFile(file: File) {
    const reader = new FileReader();
    reader.onload = () => onImage(reader.result as string);
    reader.readAsDataURL(file);
  }

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
      onDragLeave={() => setDragActive(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragActive(false);
        const file = e.dataTransfer.files?.[0];
        if (file) loadFile(file);
      }}
      onClick={() => inputRef.current?.click()}
      className={`rounded-lg border-2 border-dashed p-10 text-center cursor-pointer transition-colors ${
        dragActive ? "border-brand-600 bg-brand-50" : "border-line hover:border-brand-600/50 bg-white"
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        capture="environment"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) loadFile(file);
        }}
      />
      <p className="text-sm text-ink mb-1">Drop a label photo here, or click to choose a file</p>
      <p className="text-xs text-muted">On mobile, this opens your camera directly.</p>
    </div>
  );
}
