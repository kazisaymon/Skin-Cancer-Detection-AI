-- Databased 
CREATE DATABASE IF NOT EXISTS skin_db;
USE skin_db;

--  User Login(table)
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    role ENUM('Doctor', 'Patient') NOT NULL
);

--  DATA insert
INSERT INTO users (username, password, role) VALUES 
('admin', '1234', 'Doctor'),
('patient01', '1234', 'Patient');

-- report save
CREATE TABLE IF NOT EXISTS analysis_reports (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_name VARCHAR(100),
    diagnosis VARCHAR(100),
    confidence VARCHAR(20),
    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
