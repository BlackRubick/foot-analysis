-- MySQL 8.0+ schema for biomechanical analysis records
-- Security note: enable keyring_file/keyring_encrypted_file in MySQL for true at-rest encryption.

CREATE DATABASE IF NOT EXISTS foot_analysis_db
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE foot_analysis_db;

-- Technical patient table. Do not store name/contact here.
CREATE TABLE IF NOT EXISTS patients (
  patient_uuid CHAR(36) NOT NULL,
  patient_fhir_id VARCHAR(64) NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6),
  last_medical_act_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (patient_uuid),
  UNIQUE KEY uq_patients_fhir (patient_fhir_id),
  CONSTRAINT ck_patient_fhir_format
    CHECK (patient_fhir_id REGEXP '^patient-[A-Za-z0-9]+$')
) ENGINE=InnoDB;

-- Identity table separated from technical table, linked only by anonymous UUID.
CREATE TABLE IF NOT EXISTS patient_identity (
  identity_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  patient_uuid CHAR(36) NOT NULL,
  full_name VARBINARY(512) NOT NULL,
  contact_data VARBINARY(1024) NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (identity_id),
  UNIQUE KEY uq_identity_patient_uuid (patient_uuid),
  CONSTRAINT fk_identity_patient_uuid
    FOREIGN KEY (patient_uuid) REFERENCES patients (patient_uuid)
      ON UPDATE CASCADE ON DELETE RESTRICT
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS capture_sessions (
  session_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  patient_uuid CHAR(36) NOT NULL,
  study_root VARCHAR(32) NOT NULL DEFAULT '1.2.840.113619.2.55.3',
  study_datetime DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  study_date CHAR(8) NOT NULL,
  study_time CHAR(8) NOT NULL,
  study_instance_uid VARCHAR(96) NOT NULL,
  extremity_image_uid VARCHAR(96) NULL,
  trunk_image_uid VARCHAR(96) NULL,
  head_image_uid VARCHAR(96) NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (session_id),
  UNIQUE KEY uq_capture_study_uid (study_instance_uid),
  CONSTRAINT fk_capture_patient_uuid
    FOREIGN KEY (patient_uuid) REFERENCES patients (patient_uuid)
      ON UPDATE CASCADE ON DELETE RESTRICT,
  CONSTRAINT ck_capture_date_format
    CHECK (study_date REGEXP '^[0-9]{8}$'),
  CONSTRAINT ck_capture_time_format
    CHECK (study_time REGEXP '^[0-9]{2}\.[0-9]{2}\.[0-9]{2}$'),
  CONSTRAINT ck_capture_uid_format
    CHECK (study_instance_uid REGEXP '^1\\.2\\.840\\.113619\\.2\\.55\\.3\\.[0-9]+$')
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS analysis_records (
  analysis_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  patient_uuid CHAR(36) NOT NULL,
  session_id BIGINT UNSIGNED NOT NULL,
  analysis_type ENUM('foot', 'knee', 'posture', 'chains', 'lever') NOT NULL,
  metrics_json JSON NOT NULL,
  notes_text TEXT NULL,
  study_date CHAR(8) NOT NULL,
  study_time CHAR(8) NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (analysis_id),
  KEY ix_analysis_patient_date (patient_uuid, created_at),
  CONSTRAINT fk_analysis_patient_uuid
    FOREIGN KEY (patient_uuid) REFERENCES patients (patient_uuid)
      ON UPDATE CASCADE ON DELETE RESTRICT,
  CONSTRAINT fk_analysis_session
    FOREIGN KEY (session_id) REFERENCES capture_sessions (session_id)
      ON UPDATE CASCADE ON DELETE RESTRICT,
  CONSTRAINT ck_analysis_date_format
    CHECK (study_date REGEXP '^[0-9]{8}$'),
  CONSTRAINT ck_analysis_time_format
    CHECK (study_time REGEXP '^[0-9]{2}\.[0-9]{2}\.[0-9]{2}$'),
  CONSTRAINT ck_analysis_metrics_json_valid
    CHECK (JSON_VALID(metrics_json))
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS cda_reports (
  report_folio BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  patient_uuid CHAR(36) NOT NULL,
  session_id BIGINT UNSIGNED NULL,
  cda_document_id VARCHAR(32) NOT NULL,
  message_id VARCHAR(32) NOT NULL,
  report_kind ENUM('mechanical_advantage', 'force_moment', 'clinical_note') NOT NULL,
  note_datetime DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  note_date CHAR(8) NOT NULL,
  note_time CHAR(8) NOT NULL,
  clinician_full_name VARCHAR(255) NOT NULL,
  signature_digital_hash VARCHAR(256) NOT NULL,
  body_text TEXT NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (report_folio),
  UNIQUE KEY uq_cda_document_id (cda_document_id),
  UNIQUE KEY uq_cda_message_id (message_id),
  KEY ix_cda_patient_created (patient_uuid, created_at),
  CONSTRAINT fk_cda_patient_uuid
    FOREIGN KEY (patient_uuid) REFERENCES patients (patient_uuid)
      ON UPDATE CASCADE ON DELETE RESTRICT,
  CONSTRAINT fk_cda_session
    FOREIGN KEY (session_id) REFERENCES capture_sessions (session_id)
      ON UPDATE CASCADE ON DELETE SET NULL,
  CONSTRAINT ck_cda_document_format
    CHECK (cda_document_id REGEXP '^CDA-[0-9]{4}-[0-9]{5}$'),
  CONSTRAINT ck_cda_message_format
    CHECK (message_id REGEXP '^MSG[A-Za-z0-9]{8,20}$')
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS informed_consents (
  consent_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  patient_uuid CHAR(36) NOT NULL,
  consent_datetime DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  consent_date CHAR(8) NOT NULL,
  consent_time CHAR(8) NOT NULL,
  consent_document LONGBLOB NOT NULL,
  consent_text TEXT NULL,
  signed_by VARCHAR(255) NOT NULL,
  signature_digital_hash VARCHAR(256) NOT NULL,
  created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  PRIMARY KEY (consent_id),
  KEY ix_consent_patient_created (patient_uuid, created_at),
  CONSTRAINT fk_consent_patient_uuid
    FOREIGN KEY (patient_uuid) REFERENCES patients (patient_uuid)
      ON UPDATE CASCADE ON DELETE RESTRICT,
  CONSTRAINT ck_consent_date_format
    CHECK (consent_date REGEXP '^[0-9]{8}$'),
  CONSTRAINT ck_consent_time_format
    CHECK (consent_time REGEXP '^[0-9]{2}\.[0-9]{2}\.[0-9]{2}$')
) ENGINE=InnoDB;

-- If keyring is configured in your MySQL server, you can enforce encrypted tablespace later:
-- ALTER TABLE patient_identity ENCRYPTION='Y';
-- ALTER TABLE informed_consents ENCRYPTION='Y';

-- Optional full traceability table.
CREATE TABLE IF NOT EXISTS access_audit_log (
  audit_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  event_datetime DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
  db_user VARCHAR(128) NOT NULL,
  event_type ENUM('SELECT', 'INSERT', 'UPDATE', 'DELETE', 'LOGIN', 'EXPORT') NOT NULL,
  table_name VARCHAR(128) NOT NULL,
  record_ref VARCHAR(128) NULL,
  event_details JSON NULL,
  PRIMARY KEY (audit_id),
  KEY ix_audit_date_user (event_datetime, db_user)
) ENGINE=InnoDB;

DELIMITER $$

DROP TRIGGER IF EXISTS trg_patients_before_insert $$
CREATE TRIGGER trg_patients_before_insert
BEFORE INSERT ON patients
FOR EACH ROW
BEGIN
  IF NEW.patient_uuid IS NULL OR NEW.patient_uuid = '' THEN
    SET NEW.patient_uuid = UUID();
  END IF;

  IF NEW.patient_fhir_id IS NULL OR NEW.patient_fhir_id = '' THEN
    SET NEW.patient_fhir_id = CONCAT('patient-', UPPER(REPLACE(UUID(), '-', '')));
  END IF;
END $$

DROP TRIGGER IF EXISTS trg_capture_sessions_before_insert $$
CREATE TRIGGER trg_capture_sessions_before_insert
BEFORE INSERT ON capture_sessions
FOR EACH ROW
BEGIN
  IF NEW.study_datetime IS NULL THEN
    SET NEW.study_datetime = CURRENT_TIMESTAMP(6);
  END IF;

  IF NEW.study_date IS NULL OR NEW.study_date = '' THEN
    SET NEW.study_date = DATE_FORMAT(NEW.study_datetime, '%Y%m%d');
  END IF;

  IF NEW.study_time IS NULL OR NEW.study_time = '' THEN
    SET NEW.study_time = DATE_FORMAT(NEW.study_datetime, '%H.%i.%s');
  END IF;

  IF NEW.study_instance_uid IS NULL OR NEW.study_instance_uid = '' THEN
    SET NEW.study_instance_uid = CONCAT(
      NEW.study_root,
      '.',
      DATE_FORMAT(NEW.study_datetime, '%Y%m%d%H%i%s%f')
    );
  END IF;
END $$

DROP TRIGGER IF EXISTS trg_analysis_before_insert $$
CREATE TRIGGER trg_analysis_before_insert
BEFORE INSERT ON analysis_records
FOR EACH ROW
BEGIN
  IF NEW.study_date IS NULL OR NEW.study_date = '' THEN
    SET NEW.study_date = DATE_FORMAT(CURRENT_TIMESTAMP(6), '%Y%m%d');
  END IF;

  IF NEW.study_time IS NULL OR NEW.study_time = '' THEN
    SET NEW.study_time = DATE_FORMAT(CURRENT_TIMESTAMP(6), '%H.%i.%s');
  END IF;
END $$

DROP TRIGGER IF EXISTS trg_cda_before_insert $$
CREATE TRIGGER trg_cda_before_insert
BEFORE INSERT ON cda_reports
FOR EACH ROW
BEGIN
  IF NEW.note_datetime IS NULL THEN
    SET NEW.note_datetime = CURRENT_TIMESTAMP(6);
  END IF;

  IF NEW.note_date IS NULL OR NEW.note_date = '' THEN
    SET NEW.note_date = DATE_FORMAT(NEW.note_datetime, '%Y%m%d');
  END IF;

  IF NEW.note_time IS NULL OR NEW.note_time = '' THEN
    SET NEW.note_time = DATE_FORMAT(NEW.note_datetime, '%H.%i.%s');
  END IF;

  IF NEW.cda_document_id IS NULL OR NEW.cda_document_id = '' THEN
    SET NEW.cda_document_id = CONCAT(
      'CDA-',
      DATE_FORMAT(NEW.note_datetime, '%Y'),
      '-',
      LPAD(CAST(FLOOR(RAND() * 100000) AS CHAR), 5, '0')
    );
  END IF;

  IF NEW.message_id IS NULL OR NEW.message_id = '' THEN
    SET NEW.message_id = CONCAT('MSG', UPPER(SUBSTRING(REPLACE(UUID(), '-', ''), 1, 12)));
  END IF;
END $$

DROP TRIGGER IF EXISTS trg_consent_before_insert $$
CREATE TRIGGER trg_consent_before_insert
BEFORE INSERT ON informed_consents
FOR EACH ROW
BEGIN
  IF NEW.consent_datetime IS NULL THEN
    SET NEW.consent_datetime = CURRENT_TIMESTAMP(6);
  END IF;

  IF NEW.consent_date IS NULL OR NEW.consent_date = '' THEN
    SET NEW.consent_date = DATE_FORMAT(NEW.consent_datetime, '%Y%m%d');
  END IF;

  IF NEW.consent_time IS NULL OR NEW.consent_time = '' THEN
    SET NEW.consent_time = DATE_FORMAT(NEW.consent_datetime, '%H.%i.%s');
  END IF;
END $$

DROP TRIGGER IF EXISTS trg_medical_act_analysis $$
CREATE TRIGGER trg_medical_act_analysis
AFTER INSERT ON analysis_records
FOR EACH ROW
BEGIN
  UPDATE patients
     SET last_medical_act_at = GREATEST(last_medical_act_at, CURRENT_TIMESTAMP(6))
   WHERE patient_uuid = NEW.patient_uuid;
END $$

DROP TRIGGER IF EXISTS trg_medical_act_cda $$
CREATE TRIGGER trg_medical_act_cda
AFTER INSERT ON cda_reports
FOR EACH ROW
BEGIN
  UPDATE patients
     SET last_medical_act_at = GREATEST(last_medical_act_at, CURRENT_TIMESTAMP(6))
   WHERE patient_uuid = NEW.patient_uuid;
END $$

DROP TRIGGER IF EXISTS trg_patients_before_delete $$
CREATE TRIGGER trg_patients_before_delete
BEFORE DELETE ON patients
FOR EACH ROW
BEGIN
  IF CURRENT_TIMESTAMP(6) < DATE_ADD(OLD.last_medical_act_at, INTERVAL 5 YEAR) THEN
    SIGNAL SQLSTATE '45000'
      SET MESSAGE_TEXT = 'Deletion blocked: records must be retained for at least 5 years after last medical act.';
  END IF;
END $$

DELIMITER ;

CREATE OR REPLACE VIEW v_patient_technical_summary AS
SELECT
  p.patient_uuid,
  p.patient_fhir_id,
  p.created_at,
  p.last_medical_act_at,
  COUNT(DISTINCT cs.session_id) AS sessions_total,
  COUNT(DISTINCT ar.analysis_id) AS analysis_total,
  COUNT(DISTINCT cr.report_folio) AS cda_reports_total
FROM patients p
LEFT JOIN capture_sessions cs ON cs.patient_uuid = p.patient_uuid
LEFT JOIN analysis_records ar ON ar.patient_uuid = p.patient_uuid
LEFT JOIN cda_reports cr ON cr.patient_uuid = p.patient_uuid
GROUP BY p.patient_uuid, p.patient_fhir_id, p.created_at, p.last_medical_act_at;

-- Minimal security baseline. Adjust hosts and passwords in production.
CREATE USER IF NOT EXISTS 'foot_app'@'localhost' IDENTIFIED BY 'foot_app_change_me';
GRANT SELECT, INSERT, UPDATE ON foot_analysis_db.* TO 'foot_app'@'localhost';
FLUSH PRIVILEGES;