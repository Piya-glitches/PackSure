from collections import Counter
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Scan, Violation

router = APIRouter(prefix="/stats", tags=["stats"])


@router.get("")
def get_stats(db: Session = Depends(get_db)):
    scans = db.query(Scan).order_by(Scan.created_at.asc()).all()
    violations = db.query(Violation).all()

    total = len(scans)
    compliant = sum(1 for s in scans if s.overall_status == "COMPLIANT")
    non_compliant = sum(1 for s in scans if s.overall_status == "NON_COMPLIANT")
    needs_review = sum(1 for s in scans if s.overall_status == "NEEDS_REVIEW")
    avg_score = round(sum(s.compliance_score for s in scans) / total) if total else 0

    trend = [{"date": s.created_at.strftime("%Y-%m-%d"), "score": s.compliance_score, "status": s.overall_status} for s in scans]

    counter = Counter((v.rule_code, v.severity) for v in violations)
    violation_breakdown = [
        {"rule_code": code, "severity": sev, "count": count}
        for (code, sev), count in sorted(counter.items(), key=lambda x: -x[1])[:10]
    ]

    return {
        "total": total,
        "compliant": compliant,
        "non_compliant": non_compliant,
        "needs_review": needs_review,
        "avg_score": avg_score,
        "trend": trend,
        "violation_breakdown": violation_breakdown,
    }
