-- Medical Inventory Database Initialization

-- Create database schema
CREATE SCHEMA IF NOT EXISTS medical_inventory;

-- Request logs table
CREATE TABLE IF NOT EXISTS request_logs (
    id SERIAL PRIMARY KEY,
    request_id VARCHAR(36) UNIQUE NOT NULL,
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    user_id VARCHAR(255),
    model_version VARCHAR(50),
    processing_time FLOAT,
    object_count INTEGER,
    confidence_threshold FLOAT,
    iou_threshold FLOAT,
    success BOOLEAN NOT NULL,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ip_address INET,
    user_agent TEXT,
    file_size BIGINT,
    image_dimensions VARCHAR(20)
);

-- API keys table
CREATE TABLE IF NOT EXISTS api_keys (
    id SERIAL PRIMARY KEY,
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    name VARCHAR(255),
    rate_limit INTEGER DEFAULT 1000,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    last_used_at TIMESTAMP WITH TIME ZONE
);

-- Detection results table
CREATE TABLE IF NOT EXISTS detection_results (
    id SERIAL PRIMARY KEY,
    request_id VARCHAR(36) REFERENCES request_logs(request_id),
    class_name VARCHAR(100) NOT NULL,
    confidence FLOAT NOT NULL,
    bbox_x1 FLOAT NOT NULL,
    bbox_y1 FLOAT NOT NULL,
    bbox_x2 FLOAT NOT NULL,
    bbox_y2 FLOAT NOT NULL,
    segmentation_mask BYTEA,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Model performance metrics
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    dataset_name VARCHAR(100),
    evaluation_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    additional_info JSONB
);

-- User sessions table
CREATE TABLE IF NOT EXISTS user_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);

-- System health metrics
CREATE TABLE IF NOT EXISTS system_health (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    unit VARCHAR(20),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    additional_data JSONB
);

-- Indexes for better performance
CREATE INDEX IF NOT EXISTS idx_request_logs_created_at ON request_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_request_logs_user_id ON request_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_request_logs_success ON request_logs(success);
CREATE INDEX IF NOT EXISTS idx_request_logs_endpoint ON request_logs(endpoint);

CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_is_active ON api_keys(is_active);
CREATE INDEX IF NOT EXISTS idx_api_keys_expires_at ON api_keys(expires_at);

CREATE INDEX IF NOT EXISTS idx_detection_results_request_id ON detection_results(request_id);
CREATE INDEX IF NOT EXISTS idx_detection_results_class_name ON detection_results(class_name);
CREATE INDEX IF NOT EXISTS idx_detection_results_confidence ON detection_results(confidence);

CREATE INDEX IF NOT EXISTS idx_model_metrics_model_version ON model_metrics(model_version);
CREATE INDEX IF NOT EXISTS idx_model_metrics_evaluation_date ON model_metrics(evaluation_date);

CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_user_sessions_is_active ON user_sessions(is_active);

CREATE INDEX IF NOT EXISTS idx_system_health_metric_name ON system_health(metric_name);
CREATE INDEX IF NOT EXISTS idx_system_health_timestamp ON system_health(timestamp);

-- Create views for common queries
CREATE OR REPLACE VIEW daily_request_stats AS
SELECT 
    DATE(created_at) as date,
    COUNT(*) as total_requests,
    COUNT(*) FILTER (WHERE success = true) as successful_requests,
    COUNT(*) FILTER (WHERE success = false) as failed_requests,
    AVG(processing_time) as avg_processing_time,
    SUM(object_count) as total_objects_detected
FROM request_logs 
GROUP BY DATE(created_at)
ORDER BY date DESC;

CREATE OR REPLACE VIEW model_usage_stats AS
SELECT 
    model_version,
    COUNT(*) as usage_count,
    AVG(processing_time) as avg_processing_time,
    AVG(object_count) as avg_objects_detected,
    COUNT(*) FILTER (WHERE success = true) as success_count,
    COUNT(*) FILTER (WHERE success = false) as failure_count
FROM request_logs 
WHERE model_version IS NOT NULL
GROUP BY model_version
ORDER BY usage_count DESC;

CREATE OR REPLACE VIEW top_detected_classes AS
SELECT 
    class_name,
    COUNT(*) as detection_count,
    AVG(confidence) as avg_confidence,
    MIN(confidence) as min_confidence,
    MAX(confidence) as max_confidence
FROM detection_results
GROUP BY class_name
ORDER BY detection_count DESC;

-- Sample data for testing (optional)
-- INSERT INTO api_keys (key_hash, user_id, name, rate_limit) VALUES 
-- ('test_key_hash', 'test_user', 'Test API Key', 1000);

-- Add comments to tables
COMMENT ON TABLE request_logs IS 'Logs all API requests for monitoring and analytics';
COMMENT ON TABLE api_keys IS 'Stores API key information and rate limiting settings';
COMMENT ON TABLE detection_results IS 'Stores individual object detection results';
COMMENT ON TABLE model_metrics IS 'Stores model performance metrics and evaluation results';
COMMENT ON TABLE user_sessions IS 'Tracks user sessions and activity';
COMMENT ON TABLE system_health IS 'System health and performance metrics';

-- Grant permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO inventory_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO inventory_user;