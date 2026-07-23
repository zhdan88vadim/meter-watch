CREATE TYPE SOURCEENUM AS ENUM ('METER', 'PERSON_DETECTOR');
CREATE TYPE EVENTTYPEENUM AS ENUM ('READING', 'PERSON_DETECTED', 'PERSON_LEFT');

-- Create activity_logs table
CREATE TABLE IF NOT EXISTS activity_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    source SOURCEENUM NOT NULL,
    event_type EVENTTYPEENUM NOT NULL,
    data TEXT,
    meter_reading DOUBLE PRECISION
);

-- Create meter_readings table
CREATE TABLE IF NOT EXISTS meter_readings (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    value DOUBLE PRECISION NOT NULL,
    min_conf DOUBLE PRECISION
);

-- Create indexes for better performance
CREATE INDEX idx_activity_logs_timestamp ON activity_logs(timestamp);
CREATE INDEX idx_activity_logs_source ON activity_logs(source);
CREATE INDEX idx_activity_logs_event_type ON activity_logs(event_type);
CREATE INDEX idx_meter_readings_timestamp ON meter_readings(timestamp);

-- Add comments
COMMENT ON TABLE activity_logs IS 'Logs all activities from meter and person detector';
COMMENT ON COLUMN activity_logs.source IS 'Source of the activity: METER or PERSON_DETECTOR';
COMMENT ON COLUMN activity_logs.event_type IS 'Type of event: READING, PERSON_DETECTED, or PERSON_LEFT';
COMMENT ON COLUMN activity_logs.data IS 'JSON data with additional information';
COMMENT ON COLUMN activity_logs.meter_reading IS 'Meter reading value if event is a reading';

COMMENT ON TABLE meter_readings IS 'Stores meter readings with confidence scores';
COMMENT ON COLUMN meter_readings.value IS 'Meter reading value';
COMMENT ON COLUMN meter_readings.min_conf IS 'Minimum confidence score for the reading';