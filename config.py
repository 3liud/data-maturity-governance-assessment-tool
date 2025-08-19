# --- Configuration --------------------------------------------------------------------------------

DOMAINS = [
    "Data Governance",
    "Data Quality",
    "Metadata & Catalog",
    "Privacy & Security",
    "Data Architecture",
    "AI Governance",
]

LIKERT = [
    {"label": "1 • Ad hoc", "value": 1},
    {"label": "2 • Repeatable", "value": 2},
    {"label": "3 • Defined", "value": 3},
    {"label": "4 • Managed", "value": 4},
    {"label": "5 • Optimized", "value": 5},
]

# 18 compact, high-signal questions (3 per domain). Each has weight and framework tags for coverage.
QUESTIONS = [
    # Data Governance
    {
        "id": "DG1",
        "domain": "Data Governance",
        "text": "Executive-sponsored Data Governance Council exists and meets with decisions tracked.",
        "weight": 1.5,
        "tags": ["DAMA", "Accountability"],
    },
    {
        "id": "DG2",
        "domain": "Data Governance",
        "text": "Data ownership & RACI for key datasets are documented and socialized.",
        "weight": 1.2,
        "tags": ["DAMA", "Ownership"],
    },
    {
        "id": "DG3",
        "domain": "Data Governance",
        "text": "Enterprise data policies (access, retention, classification) are approved and enforced.",
        "weight": 1.3,
        "tags": ["GDPR", "KE_DPA"],
    },
    # Data Quality
    {
        "id": "DQ1",
        "domain": "Data Quality",
        "text": "Quality dimensions (accuracy, completeness, timeliness) have SLAs and targets.",
        "weight": 1.4,
        "tags": ["DAMA"],
    },
    {
        "id": "DQ2",
        "domain": "Data Quality",
        "text": "Automated profiling/monitoring with alerting on critical pipelines.",
        "weight": 1.5,
        "tags": ["DAMA"],
    },
    {
        "id": "DQ3",
        "domain": "Data Quality",
        "text": "Issue triage workflow exists with steward ownership and MTTR tracked.",
        "weight": 1.1,
        "tags": ["DAMA"],
    },
    # Metadata & Catalog
    {
        "id": "MD1",
        "domain": "Metadata & Catalog",
        "text": "Central catalog documents business/technical lineage for critical assets.",
        "weight": 1.4,
        "tags": ["DAMA"],
    },
    {
        "id": "MD2",
        "domain": "Metadata & Catalog",
        "text": "Glossary terms have owners and are linked to datasets/dashboards.",
        "weight": 1.2,
        "tags": ["DAMA"],
    },
    {
        "id": "MD3",
        "domain": "Metadata & Catalog",
        "text": "Active metadata drives discovery, impact analysis, and approvals.",
        "weight": 1.4,
        "tags": ["DAMA"],
    },
    # Privacy & Security
    {
        "id": "PS1",
        "domain": "Privacy & Security",
        "text": "Data classification & retention schedules applied and auditable.",
        "weight": 1.5,
        "tags": ["GDPR", "KE_DPA"],
    },
    {
        "id": "PS2",
        "domain": "Privacy & Security",
        "text": "PII handling: DPIAs, consent, minimization, and subject rights processes.",
        "weight": 1.6,
        "tags": ["GDPR", "KE_DPA"],
    },
    {
        "id": "PS3",
        "domain": "Privacy & Security",
        "text": "Security controls: encryption, key management, IAM least-privilege, monitoring.",
        "weight": 1.5,
        "tags": ["Security"],
    },
    # Data Architecture
    {
        "id": "DA1",
        "domain": "Data Architecture",
        "text": "Documented reference architecture (ingest→storage→processing→serve) with standards.",
        "weight": 1.3,
        "tags": ["Architecture"],
    },
    {
        "id": "DA2",
        "domain": "Data Architecture",
        "text": "CI/CD for data (tests, deploys, rollback), infra as code, reproducibility.",
        "weight": 1.5,
        "tags": ["Architecture"],
    },
    {
        "id": "DA3",
        "domain": "Data Architecture",
        "text": "Cost/perf governance (tiering, query policies, SLOs) with regular review.",
        "weight": 1.2,
        "tags": ["Architecture"],
    },
    # AI Governance
    {
        "id": "AI1",
        "domain": "AI Governance",
        "text": "AI use-case intake & risk classification process (incl. prohibitions).",
        "weight": 1.5,
        "tags": ["EU_AI_Act"],
    },
    {
        "id": "AI2",
        "domain": "AI Governance",
        "text": "Model lifecycle controls: data provenance, evals, bias testing, human oversight.",
        "weight": 1.6,
        "tags": ["EU_AI_Act", "GDPR"],
    },
    {
        "id": "AI3",
        "domain": "AI Governance",
        "text": "Transparency & incident response: model cards, logs, rollback, user notices.",
        "weight": 1.4,
        "tags": ["EU_AI_Act"],
    },
]

FRAMEWORK_ROWS = ["DAMA", "GDPR", "KE_DPA", "EU_AI_Act", "Security", "Architecture"]

RECS = {
    "Data Governance": {
        1: [
            "Stand up a DG Council with charter and quarterly OKRs.",
            "Publish data ownership map with RACI per domain.",
        ],
        2: [
            "Approve baseline access/retention policies; roll out mandatory training.",
            "Launch monthly governance review with decision log.",
        ],
        3: [
            "Automate policy checks in CI/CD (schemas, PII gates).",
            "Add governance KPIs to exec scorecards.",
        ],
        4: [
            "Embed policy-as-code across pipelines; auto-exceptions with expiry.",
            "External audit of governance effectiveness.",
        ],
        5: [
            "Benchmark vs peers; tie incentives to data outcomes.",
            "Continuous improvement program with hypothesis tests.",
        ],
    },
    "Data Quality": {
        1: [
            "Define critical data elements and SLAs.",
            "Deploy profiling on top 10 pipelines.",
        ],
        2: [
            "Alerting on SLA breaches; steward triage workflow.",
            "Capture MTTR and recurring defect classes.",
        ],
        3: [
            "Contract tests at data product boundaries.",
            "Quarantine bad records with self-service fixes.",
        ],
        4: [
            "Predictive DQ using drift/anomaly detection.",
            "Quality KPIs on exec dashboards.",
        ],
        5: [
            "Closed-loop prevention at source; automated RCA & fix PRs.",
            "Continuous risk-based sampling at scale.",
        ],
    },
    "Metadata & Catalog": {
        1: [
            "Stand up catalog; index top 100 assets.",
            "Publish glossary for top metrics.",
        ],
        2: [
            "Link lineage to pipelines/dashboards.",
            "Assign glossary owners and review cadence.",
        ],
        3: [
            "Approval workflows using active metadata.",
            "Impact analysis before breaking changes.",
        ],
        4: [
            "Contextual recommendations in BI & notebooks.",
            "Federated catalog governance.",
        ],
        5: [
            "Auto-harvest semantics; policy-driven access from metadata.",
            "Unified search across data+ML assets.",
        ],
    },
    "Privacy & Security": {
        1: [
            "Classify data; define retention matrix.",
            "Baseline encryption & IAM least-privilege.",
        ],
        2: [
            "DPIA templates; consent/minimization flows.",
            "Subject rights playbooks (access/erase).",
        ],
        3: [
            "Key rotation, HSM/KMS; centralized logging.",
            "Privacy reviews in change management.",
        ],
        4: [
            "Data masking/tokenization at source; DLP everywhere.",
            "Continuous control monitoring with alerts.",
        ],
        5: [
            "Formal privacy assurance; tabletop incident drills.",
            "Privacy engineering patterns embedded in SDLC.",
        ],
    },
    "Data Architecture": {
        1: [
            "Document target state; retire anti-patterns.",
            "Standardize storage layers and formats.",
        ],
        2: [
            "Infra as code; unit/integration tests for pipelines.",
            "Versioned data products & schemas.",
        ],
        3: [
            "Cost/perf SLOs; auto-tuning & tiering.",
            "Backfills and reproducibility policies.",
        ],
        4: [
            "Multi-region DR; blue/green for data jobs.",
            "Query governance & workload isolation.",
        ],
        5: [
            "Self-service platform with golden paths.",
            "Continuous architecture fitness functions.",
        ],
    },
    "AI Governance": {
        1: [
            "Register all AI use cases; define prohibited/high-risk list.",
            "Form AI risk committee; intake workflow.",
        ],
        2: [
            "Bias/fairness tests; evaluation gates; human-in-the-loop.",
            "Model cards & data sheets baseline.",
        ],
        3: [
            "Monitoring for drift & incidents; rollback plans.",
            "Third-party model/vendor due diligence.",
        ],
        4: [
            "Policy-as-code for AI; red-teaming; audit trails.",
            "User disclosure & appeal mechanisms.",
        ],
        5: [
            "Assurance reports; external audits.",
            "Continuous stress tests and chaos experiments for AI.",
        ],
    },
}
