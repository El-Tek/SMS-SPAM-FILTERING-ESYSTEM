-- Create the SpamRepository database
CREATE DATABASE IF NOT EXISTS SpamRepositoryDB;

-- Use the SpamRepositoryDB database
USE SpamRepositoryDB;

-- Create the table to store spam messages
CREATE TABLE IF NOT EXISTS SpamRepository (
    ID INT AUTO_INCREMENT PRIMARY KEY,  -- Unique ID for each message
    MessageText TEXT NOT NULL,          -- The actual content of the SMS message
    SpamLabel BOOLEAN NOT NULL,         -- Spam label (1 for spam, 0 for not spam)
    DateAdded TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- Date when the message was added
    MessageType VARCHAR(100)            -- Type of spam (e.g., Phishing, Advertisement, etc.)
);

-- Insert sample data into SpamRepository
INSERT INTO SpamRepository (MessageText, SpamLabel, MessageType)
VALUES
('Congratulations! You have won a lottery of $1000!', TRUE, 'Phishing'),
('Your order has been shipped and will arrive soon.', FALSE, 'Transactional'),
('Win a brand new car! Click here to claim your prize!', TRUE, 'Advertisement'),
('Reminder: Your appointment is scheduled for tomorrow.', FALSE, 'Reminder'),
('Get 50% off on all electronics this weekend only!', TRUE, 'Promotion'),
('Important: Your account has been compromised. Click here to secure it.', TRUE, 'Phishing'),
('Don’t miss out on our latest offers! Up to 70% off.', TRUE, 'Promotion'),
('Thank you for your purchase! Your order will be delivered soon.', FALSE, 'Transactional');

-- Query to view the contents of the SpamRepository table
SELECT * FROM SpamRepository;
