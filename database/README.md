# MySQL clinical database setup

This folder contains a MySQL 8.0+ schema for the biomechanical system.

## Quick start

Run from project root:

```bash
python database/init_mysql.py --user cesar --password cesar123
```

If your MySQL is remote/custom:

```bash
python database/init_mysql.py --host 127.0.0.1 --port 3306 --user cesar --password cesar123
```

## What the schema enforces

- Unique patient identifier in FHIR-like format: `patient-[alphanumeric]`
- DICOM-like study UID: `1.2.840.113619.2.55.3.[timestamp]`
- Separate `study_date` (`YYYYMMDD`) and `study_time` (`HH.MM.SS`)
- CDA document format: `CDA-[YEAR]-[FOLIO5]`
- Internal message format: `MSG[alphanumeric]`
- Identity data and technical data are dissociated:
  - `patient_identity` stores encrypted name/contact bytes
  - `patients`, `capture_sessions`, `analysis_records` store technical records
  - Link is anonymous `patient_uuid`
- Informed consent is mandatory-capable through table `informed_consents`
- Optional full traceability table `access_audit_log`
- Retention rule by trigger: blocks delete of a patient before 5 years after last medical act

## Security notes

- At-rest table encryption is prepared as an optional hardening step.
- For real at-rest encryption, enable MySQL keyring plugin and then run:
  - `ALTER TABLE patient_identity ENCRYPTION='Y';`
  - `ALTER TABLE informed_consents ENCRYPTION='Y';`
- Replace default app credential (`foot_app_change_me`) after deployment.

## Main tables

- `patients`
- `patient_identity`
- `capture_sessions`
- `analysis_records`
- `cda_reports`
- `informed_consents`
- `access_audit_log`

## App integration (CLI)

The CLI now writes patient/session metadata and analysis metrics to MySQL automatically.

Example:

```bash
python main.py --mode cli \
  --foot-image /ruta/pie.jpg \
  --knee-image /ruta/rodilla.jpg \
  --posture-image /ruta/postura.jpg \
  --chains-image /ruta/cadenas.jpg
```

Optional DB flags:

- `--db-host`, `--db-port`, `--db-user`, `--db-password`, `--db-name`
- `--patient-uuid`, `--patient-fhir-id`
- `--no-db` (disable database writes)

## App integration (Tkinter UI)

In the top toolbar you now have:

- `Config DB`: configure and connect MySQL + patient/session context
- `Consentimiento`: save informed consent (`informed_consents`)
- `Nota clinica`: save HL7 CDA-like clinical notes (`cda_reports`)

Behavior:

- UI analyses (`pie`, `rodilla`, `postura`, `cadenas`) are stored in `analysis_records` when DB is enabled.
- Before processing images, the UI checks that at least one informed consent exists for the active patient.