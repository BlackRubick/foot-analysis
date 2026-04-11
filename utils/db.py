from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Optional, Tuple

try:
    import mysql.connector as mysql_connector
except Exception as exc:  # pragma: no cover - handled at runtime
    mysql_connector = None
    _MYSQL_IMPORT_ERROR = exc
else:
    _MYSQL_IMPORT_ERROR = None


class DatabaseClient:
    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str = "foot_analysis_db",
    ) -> None:
        if mysql_connector is None:
            raise RuntimeError(
                "mysql-connector-python no esta disponible. "
                "Instala dependencias con: pip install -r requirements.txt"
            ) from _MYSQL_IMPORT_ERROR

        self.connection = mysql_connector.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            autocommit=False,
        )

        # Extensiones de esquema opcionales (p.ej. imagen de firma en consentimientos)
        self._has_signature_image_col = False
        self._ensure_optional_columns()

    def _ensure_optional_columns(self) -> None:
        """Asegura columnas opcionales sin romper instalaciones existentes."""
        try:
            cursor = self.connection.cursor()
            # Comprobar si existe la columna para imagen de firma
            cursor.execute("SHOW COLUMNS FROM informed_consents LIKE 'signature_image'")
            exists = cursor.fetchone() is not None
            if not exists:
                try:
                    cursor.execute("ALTER TABLE informed_consents ADD COLUMN signature_image LONGBLOB NULL")
                except Exception:
                    # Si falla (sin permisos, columna ya existe, etc.), continuamos en modo compatible
                    pass
                cursor.execute("SHOW COLUMNS FROM informed_consents LIKE 'signature_image'")
                exists = cursor.fetchone() is not None
            self._has_signature_image_col = bool(exists)

            # Asegurar que el documento de consentimiento se almacene como BLOB
            try:
                cursor.execute("SHOW COLUMNS FROM informed_consents LIKE 'consent_document'")
                col = cursor.fetchone()
                if col is not None:
                    col_type = str(col[1]).lower()
                    # Si no es un tipo BLOB, intentar convertirlo a LONGBLOB para evitar decodificación UTF-8
                    if "blob" not in col_type:
                        try:
                            cursor.execute("ALTER TABLE informed_consents MODIFY consent_document LONGBLOB NULL")
                        except Exception:
                            # Si falla (permisos, etc.), dejamos el tipo como está
                            pass
            except Exception:
                # Si no podemos inspeccionar/modificar la columna, continuamos en modo compatible
                pass
        except Exception:
            self._has_signature_image_col = False
        finally:
            try:
                cursor.close()
            except Exception:
                pass

    def close(self) -> None:
        self.connection.close()

    def ensure_patient(
        self,
        patient_uuid: Optional[str] = None,
        patient_fhir_id: Optional[str] = None,
    ) -> Tuple[str, str]:
        patient_uuid = patient_uuid or str(uuid.uuid4())
        patient_fhir_id = patient_fhir_id or f"patient-{uuid.uuid4().hex.upper()}"

        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT patient_uuid, patient_fhir_id FROM patients WHERE patient_uuid = %s",
            (patient_uuid,),
        )
        row = cursor.fetchone()
        if row:
            cursor.close()
            return row[0], row[1]

        cursor.execute(
            "INSERT INTO patients (patient_uuid, patient_fhir_id) VALUES (%s, %s)",
            (patient_uuid, patient_fhir_id),
        )
        self.connection.commit()
        cursor.close()
        return patient_uuid, patient_fhir_id

    def create_capture_session(self, patient_uuid: str) -> Dict[str, Any]:
        cursor = self.connection.cursor()
        cursor.execute(
            (
                "INSERT INTO capture_sessions "
                "(patient_uuid, study_date, study_time, study_instance_uid) "
                "VALUES (%s, %s, %s, %s)"
            ),
            (patient_uuid, "", "", ""),
        )
        session_id = int(cursor.lastrowid)

        cursor.execute(
            (
                "SELECT session_id, study_instance_uid, study_date, study_time "
                "FROM capture_sessions WHERE session_id = %s"
            ),
            (session_id,),
        )
        row = cursor.fetchone()
        self.connection.commit()
        cursor.close()

        return {
            "session_id": int(row[0]),
            "study_instance_uid": row[1],
            "study_date": row[2],
            "study_time": row[3],
        }

    def save_analysis(
        self,
        patient_uuid: str,
        session_id: int,
        analysis_type: str,
        metrics: Dict[str, Any],
        notes_text: str = "",
    ) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            (
                "INSERT INTO analysis_records "
                "(patient_uuid, session_id, analysis_type, metrics_json, notes_text, study_date, study_time) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)"
            ),
            (
                patient_uuid,
                session_id,
                analysis_type,
                json.dumps(metrics, ensure_ascii=False),
                notes_text,
                "",
                "",
            ),
        )
        self.connection.commit()
        cursor.close()

    def upsert_patient_identity(self, patient_uuid: str, full_name: str, contact_data: str = "") -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            (
                "INSERT INTO patient_identity (patient_uuid, full_name, contact_data) "
                "VALUES (%s, %s, %s) "
                "ON DUPLICATE KEY UPDATE full_name = VALUES(full_name), contact_data = VALUES(contact_data)"
            ),
            (
                patient_uuid,
                full_name.encode("utf-8"),
                contact_data.encode("utf-8"),
            ),
        )
        self.connection.commit()
        cursor.close()

    def has_consent(self, patient_uuid: str) -> bool:
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM informed_consents WHERE patient_uuid = %s",
            (patient_uuid,),
        )
        row = cursor.fetchone()
        cursor.close()
        return bool(row and int(row[0]) > 0)

    def save_informed_consent(
        self,
        patient_uuid: str,
        consent_text: str,
        signed_by: str,
        signature_digital_hash: str,
        consent_document_bytes: bytes,
        signature_image_bytes: Optional[bytes] = None,
    ) -> None:
        # Asegurar que los documentos binarios se envíen como BLOB y no se intenten decodificar como UTF-8
        doc_param = consent_document_bytes
        sig_img_param = signature_image_bytes
        try:
            if consent_document_bytes is not None and hasattr(mysql_connector, "Binary"):
                doc_param = mysql_connector.Binary(consent_document_bytes)
            if signature_image_bytes is not None and hasattr(mysql_connector, "Binary"):
                sig_img_param = mysql_connector.Binary(signature_image_bytes)
        except Exception:
            # Si por alguna razón falla, usamos los bytes crudos (comportamiento anterior)
            doc_param = consent_document_bytes
            sig_img_param = signature_image_bytes

        cursor = self.connection.cursor()
        if getattr(self, "_has_signature_image_col", False):
            cursor.execute(
                (
                    "INSERT INTO informed_consents "
                    "(patient_uuid, consent_date, consent_time, consent_document, consent_text, signed_by, signature_digital_hash, signature_image) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
                ),
                (
                    patient_uuid,
                    "",
                    "",
                    doc_param,
                    consent_text,
                    signed_by,
                    signature_digital_hash,
                    sig_img_param,
                ),
            )
        else:
            cursor.execute(
                (
                    "INSERT INTO informed_consents "
                    "(patient_uuid, consent_date, consent_time, consent_document, consent_text, signed_by, signature_digital_hash) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)"
                ),
                (
                    patient_uuid,
                    "",
                    "",
                    doc_param,
                    consent_text,
                    signed_by,
                    signature_digital_hash,
                ),
            )
        self.connection.commit()
        cursor.close()

    def save_cda_report(
        self,
        patient_uuid: str,
        session_id: Optional[int],
        report_kind: str,
        clinician_full_name: str,
        signature_digital_hash: str,
        body_text: str,
    ) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            (
                "INSERT INTO cda_reports "
                "(patient_uuid, session_id, cda_document_id, message_id, report_kind, note_date, note_time, clinician_full_name, signature_digital_hash, body_text) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            ),
            (
                patient_uuid,
                session_id,
                "",
                "",
                report_kind,
                "",
                "",
                clinician_full_name,
                signature_digital_hash,
                body_text,
            ),
        )
        self.connection.commit()
        cursor.close()

    def fetch_patient_history(self, patient_uuid: str, limit: int = 10) -> Dict[str, Any]:
        cursor = self.connection.cursor(dictionary=True)

        cursor.execute(
            (
                "SELECT session_id, study_instance_uid, study_date, study_time, created_at "
                "FROM capture_sessions WHERE patient_uuid = %s ORDER BY created_at DESC LIMIT %s"
            ),
            (patient_uuid, limit),
        )
        sessions = cursor.fetchall()

        cursor.execute(
            (
                "SELECT analysis_id, analysis_type, study_date, study_time, created_at, metrics_json, notes_text "
                "FROM analysis_records WHERE patient_uuid = %s ORDER BY created_at DESC LIMIT %s"
            ),
            (patient_uuid, limit),
        )
        analyses = cursor.fetchall()

        cursor.execute(
            (
                "SELECT report_folio, report_kind, cda_document_id, message_id, note_date, note_time, "
                "clinician_full_name, created_at, body_text "
                "FROM cda_reports WHERE patient_uuid = %s ORDER BY created_at DESC LIMIT %s"
            ),
            (patient_uuid, limit),
        )
        notes = cursor.fetchall()

        cursor.close()
        return {
            "sessions": sessions,
            "analyses": analyses,
            "notes": notes,
        }

    def fetch_latest_consent(self, patient_uuid: str) -> Optional[Dict[str, Any]]:
        """Devuelve el consentimiento más reciente del paciente (o None si no hay)."""
        cursor = self.connection.cursor(dictionary=True)
        try:
            cursor.execute(
                (
                    "SELECT consent_id, consent_date, consent_time, created_at, consent_document "
                    "FROM informed_consents WHERE patient_uuid = %s ORDER BY created_at DESC LIMIT 1"
                ),
                (patient_uuid,),
            )
            row = cursor.fetchone()
            return row if row else None
        finally:
            cursor.close()