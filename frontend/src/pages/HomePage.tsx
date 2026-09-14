import { Link } from "react-router-dom";

const FIELDS = [
  "Manufacturer / packer / importer address",
  "Common or generic name",
  "Net quantity in standard units",
  "Month & year of manufacture",
  "MRP inclusive of all taxes",
  "Consumer care details",
  "Country of origin",
  "Unit sale price",
];

const LAYERS = [
  { name: "Barcode calibration", detail: "EAN-13 as a self-calibrating physical ruler" },
  { name: "PDP detection", detail: "YOLOv8, with a classical fallback when unfine-tuned" },
  { name: "CRAFT + CRNN OCR", detail: "Multi-script text detection and recognition" },
  { name: "NER field classification", detail: "DistilBERT backbone + rule-augmented practical layer" },
  { name: "Rule validation", detail: "Format checks, font-size-in-mm, responsible-party resolution" },
];

export default function HomePage() {
  return (
    <div>
      <section className="mx-auto max-w-6xl px-6 pt-20 pb-16 grid md:grid-cols-[1.3fr_1fr] gap-14 items-start">
        <div>
          <p className="text-brand-700 text-sm font-medium mb-4">SIH26034 · Ministry of Consumer Affairs, Food &amp; Public Distribution</p>
          <h1 className="text-4xl md:text-5xl font-semibold leading-tight text-ink mb-6">
            Read a label the way<br />the law reads it.
          </h1>
          <p className="text-muted text-base leading-relaxed max-w-md mb-8">
            Every pre-packaged commodity sold in India must carry eight mandatory declarations under the Legal
            Metrology (Packaged Commodities) Rules, 2011. PackSure scans a photo of any package and checks all
            eight — including whether the print is physically large enough to read.
          </p>
          <div className="flex gap-3">
            <Link to="/scan" className="btn-primary">Scan a label</Link>
            <Link to="/dashboard" className="btn-secondary">View dashboard</Link>
          </div>
        </div>

        <div className="card p-6">
          <p className="text-xs font-medium text-muted mb-4 uppercase tracking-wide">Mandatory declarations checked</p>
          <ul className="space-y-2.5">
            {FIELDS.map((f, i) => (
              <li key={f} className="flex items-start gap-3 text-sm text-ink">
                <span className="mono text-brand-700 text-xs mt-0.5 w-4">{String(i + 1).padStart(2, "0")}</span>
                <span>{f}</span>
              </li>
            ))}
          </ul>
        </div>
      </section>

      <section className="border-t border-line bg-white">
        <div className="mx-auto max-w-6xl px-6 py-16">
          <h2 className="text-xl font-semibold text-ink mb-8">Pipeline</h2>
          <div className="grid md:grid-cols-5 gap-4">
            {LAYERS.map((l, i) => (
              <div key={l.name} className="card p-4">
                <p className="mono text-xs text-brand-700 mb-2">{String(i).padStart(2, "0")}</p>
                <p className="text-sm font-medium text-ink mb-1">{l.name}</p>
                <p className="text-xs text-muted leading-relaxed">{l.detail}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 py-16 grid md:grid-cols-3 gap-8">
        <div>
          <h3 className="font-semibold text-ink mb-3">The gap this fills</h3>
          <p className="text-sm text-muted leading-relaxed">
            The government&apos;s own eMaap portal handles registration and licensing — it does not scan a package
            image and check what&apos;s printed on it. Industrial vision systems compare a label against that
            brand&apos;s own approved template on a fixed camera rig. Neither works for an arbitrary product,
            photographed in the field.
          </p>
        </div>
        <div>
          <h3 className="font-semibold text-ink mb-3">Barcode as a ruler</h3>
          <p className="text-sm text-muted leading-relaxed">
            Font-size compliance is normally impossible to check from an uncalibrated photo. PackSure uses the
            EAN-13 barcode already on the package — a fixed, standardised physical width — as a built-in ruler,
            deriving a real pixels-per-millimetre factor for that exact photo.
          </p>
        </div>
        <div>
          <h3 className="font-semibold text-ink mb-3">Beyond keyword matching</h3>
          <p className="text-sm text-muted leading-relaxed">
            The Rules distinguish manufacturer, packer, importer, and brand owner — only some of those roles
            discharge the legal obligation. PackSure&apos;s responsible-party engine resolves which role is
            declared and flags labels that name only a brand owner.
          </p>
        </div>
      </section>
    </div>
  );
}
