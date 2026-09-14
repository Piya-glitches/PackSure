from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Scan
from app.pdf_report import generate_report_pdf

router = APIRouter(prefix="/scans", tags=["report"])


@router.get("/{scan_id}/report")
def download_report(scan_id: str, db: Session = Depends(get_db)):
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found.")

    pdf_bytes = generate_report_pdf(scan)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="packsure-report-{scan.id[:8]}.pdf"'},
    )
